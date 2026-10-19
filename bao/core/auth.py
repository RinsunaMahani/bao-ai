"""Bao AI - Sign-in for the web app.

Local accounts, because the app is shown offline: a sign-in through
Google or Microsoft needs the internet, and a demo with Wi-Fi off would
lock everyone out. Accounts live in a TOML file outside version control
(`users.toml` by default, see [auth] in config.toml) and are made with
`python scripts/add_user.py <name>`, which asks for the password without
echoing it.

Passwords are never stored: each is hashed with scrypt and a random salt,
from the standard library, so this adds no dependency. Verification runs
in constant time, and an unknown name costs the same work as a wrong
password, so the response does not reveal which names exist.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
import tomllib

# scrypt cost: about 50 ms and 16 MB per check on a laptop, which is
# nothing to a person signing in and a great deal to someone guessing.
_N, _R, _P, _DKLEN = 2**14, 8, 1, 32

USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,32}")
# Shortest password accepted. Named without "password" so that
# printing the number is not mistaken for printing a password.
MIN_CHARS = 8

# Wrong passwords allowed per name before it is locked, and for how long.
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return "$".join([
        "scrypt", str(_N), str(_R), str(_P),
        base64.b64encode(salt).decode(), base64.b64encode(digest).decode(),
    ])


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Checked against when the name is unknown, so that case takes as long as
# a wrong password and the timing does not tell names apart.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


def load_users(path: str) -> dict[str, str]:
    """Name -> password hash. Empty when the file does not exist yet."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        users = tomllib.load(f).get("users", {})
    return {str(name): str(value) for name, value in users.items()}


def users_from_env(value: str | None) -> dict[str, str]:
    """Accounts from the BAO_USERS environment variable, as "name:hash"
    entries separated by commas or new lines. For hosts where a file
    cannot be placed beside the app but secrets become environment
    variables (Streamlit Community Cloud). Hashes contain neither ":" nor
    ",", so the format is unambiguous; malformed entries are skipped.
    """
    users = {}
    for entry in re.split(r"[,\n]", value or ""):
        name, sep, stored = entry.strip().partition(":")
        if sep and USERNAME_RE.fullmatch(name) and stored.startswith("scrypt$"):
            users[name] = stored
    return users


def load_accounts(path: str) -> dict[str, str]:
    """Every account the app accepts: the file, then BAO_USERS on top."""
    return {**load_users(path), **users_from_env(os.getenv("BAO_USERS"))}


def save_users(path: str, users: dict[str, str]) -> None:
    """Writes the accounts file. Names are validated by the caller, and
    hashes contain only base64 and "$", so plain quoting is safe.
    """
    lines = [
        "# Bao sign-in accounts. Password hashes only - no passwords.",
        "# Manage with: python scripts/add_user.py <name>",
        "# Never commit this file (it is in .gitignore and .dockerignore).",
        "",
        "[users]",
    ]
    lines += [f'"{name}" = "{users[name]}"' for name in sorted(users)]
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


class LoginGuard:
    """Checks a sign-in and locks a name after repeated wrong passwords.

    One guard serves the whole server, so the count follows the name
    across browser tabs; a new tab must not mean five fresh guesses.
    """

    def __init__(self, max_failures: int = MAX_FAILURES, lockout_seconds: float = LOCKOUT_SECONDS,
                 clock=time.monotonic):
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self._clock = clock
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def seconds_locked(self, username: str) -> int:
        """How long `username` stays locked; 0 when it is not."""
        key = username.strip().lower()
        with self._lock:
            recent = [t for t in self._failures.get(key, [])
                      if self._clock() - t < self.lockout_seconds]
            self._failures[key] = recent
            if len(recent) < self.max_failures:
                return 0
            return max(1, int(self.lockout_seconds - (self._clock() - recent[-1])) + 1)

    def check(self, users: dict[str, str], username: str, password: str) -> str | None:
        """The account name when the sign-in is right, otherwise None.
        A locked name is refused without checking the password at all.
        """
        username = username.strip()
        if self.seconds_locked(username):
            return None
        stored = users.get(username)
        ok = verify_password(password, stored if stored else _DUMMY_HASH) and stored is not None
        key = username.lower()
        with self._lock:
            if ok:
                self._failures.pop(key, None)
            else:
                self._failures.setdefault(key, []).append(self._clock())
        return username if ok else None
