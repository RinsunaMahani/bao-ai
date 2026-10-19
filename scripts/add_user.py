"""Manage the web app's sign-in accounts.

    python scripts/add_user.py rinsuna            # add, or change the password
    python scripts/add_user.py rinsuna --remove   # delete the account
    python scripts/add_user.py --list             # names only
    python scripts/add_user.py rinsuna --secret   # print its BAO_USERS line for a host

The password is asked for twice and never echoed or written down: only
its scrypt hash goes into the accounts file ([auth] users_file in
config.toml, users.toml by default), which is never committed.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bao.core.auth import (  # noqa: E402
    MIN_CHARS,
    USERNAME_RE,
    hash_password,
    load_users,
    save_users,
)
from bao.core.config import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("username", nargs="?")
    parser.add_argument("--remove", action="store_true", help="delete the account")
    parser.add_argument("--list", action="store_true", help="list account names")
    parser.add_argument("--secret", action="store_true",
                        help="print the saved account as a name:hash line for the BAO_USERS secret")
    args = parser.parse_args()

    path = Settings().users_file
    users = load_users(path)

    if args.list:
        print("\n".join(sorted(users)) or "(no accounts yet)")
        return 0
    if not args.username:
        parser.error("give a username, or --list")
    name = args.username.strip()
    if not USERNAME_RE.fullmatch(name):
        parser.error("usernames are 1-32 letters, digits, '.', '_' or '-'")

    if args.remove:
        if users.pop(name, None) is None:
            print(f"No account named {name}.")
            return 1
        save_users(path, users)
        print(f"Removed {name}.")
        return 0

    if args.secret:
        # The hash already saved, for a host's secret settings (BAO_USERS).
        # No password is asked for, so none can reach the output.
        if name not in users:
            print(f"No account named {name}. Make it first: python scripts/add_user.py {name}")
            return 1
        print(f"{name}:{users[name]}")
        return 0

    password = getpass.getpass(f"Password for {name}: ")
    if len(password) < MIN_CHARS:
        print(f"Use at least {MIN_CHARS} characters.")
        return 1
    if getpass.getpass("Again: ") != password:
        print("The two passwords differ; nothing was changed.")
        return 1
    existed = name in users
    users[name] = hash_password(password)
    save_users(path, users)
    print(f"{'Changed the password for' if existed else 'Added'} {name} ({path}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
