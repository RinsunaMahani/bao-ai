"""Refuses to build a distributable archive that contains secrets.

WHY: `.gitignore` and `.dockerignore` both correctly exclude `.env`, and
both were correctly configured — yet a hand-built zip of this project was
distributed with a live GEMINI_API_KEY and HF_TOKEN inside it. Neither
ignore file governs `zip -r`. The gap was never in the configuration; it
was that nothing checked the artefact actually being handed over.

Run before sharing a build:

    python scripts/check_archive.py            # check the working tree
    python scripts/check_archive.py bao.zip    # check a built archive

Two kinds of check, because a secret can hide in two ways:

  BY NAME — files that are credentials by definition: `.env`,
  `credentials.json`, `id_rsa`, a `.p12` keystore. A `.pem` or `.key` is
  flagged only when it actually holds a PRIVATE key. The name alone used to
  be enough, so pip's public certificate-authority bundle (`cacert.pem`)
  failed the check on every machine with a virtualenv — and a check that
  always fails is one people learn to ignore.

  BY CONTENT — the credential formats this project uses, anywhere in a text
  file. The incident above was caught by name, but the same key pasted into
  config.toml, a notebook or a README would have passed. The output names
  the file and the kind of credential, never the value.
"""

from __future__ import annotations

import re
import sys
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path, PurePath

FORBIDDEN_NAMES = {".env", "credentials.json", "service-account.json", "id_rsa"}
ALLOWED = {".env.example", ".env.template"}
# Keystores that are always private. .pem and .key are judged by content.
ALWAYS_SECRET_SUFFIXES = (".p12", ".pfx")
KEY_FILE_SUFFIXES = (".pem", ".key")

# Directories that are never part of the project's own content. Matched on
# path PARTS rather than on a "/.git/" substring: the earlier substring test
# never matched on Windows, where the separator is a backslash, so the whole
# of .git was scanned there.
SKIP_DIRS = {".git"}
# Third-party code: checked by name, not by content. Test fixtures in
# installed packages routinely contain realistic-looking fake keys.
THIRD_PARTY_DIRS = {".venv", "venv", "site-packages", "node_modules", "__pycache__"}

MAX_SCAN_BYTES = 1_000_000

# Returns a file's bytes, or None when it is too large or unreadable.
Reader = Callable[[], bytes | None]

CONTENT_PATTERNS = {
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    "Google API key": re.compile(rb"\bAIza[0-9A-Za-z_\-]{35}\b"),
    # An assignment of this project's credential variables to a real-looking
    # value, in whatever format the provider currently issues. The template
    # leaves these empty, so it does not match.
    "assigned credential": re.compile(
        rb"(?:GEMINI_API_KEY|GOOGLE_API_KEY|HF_TOKEN)\s*[=:]\s*[\"']?[A-Za-z0-9._\-]{20,}"
    ),
}
PRIVATE_KEY_MARKER = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:1024]


def findings(path: str, read: Reader) -> list[str]:
    """Why this path must not ship, or [] if it may. `read` returns the
    file's bytes and is called only when the content matters.
    """
    pure = PurePath(path.replace("\\", "/"))
    name, parts = pure.name, set(pure.parts)

    if name not in ALLOWED:
        if name in FORBIDDEN_NAMES or name.startswith(".env."):
            return ["credential file"]
        if name.endswith(ALWAYS_SECRET_SUFFIXES):
            return ["keystore"]
    reasons: list[str] = []

    if name.endswith(KEY_FILE_SUFFIXES):
        data = read()
        if data is not None and PRIVATE_KEY_MARKER.search(data):
            reasons.append("private key")
        return reasons

    if parts & THIRD_PARTY_DIRS:
        return reasons

    data = read()
    if data is None or _looks_binary(data):
        return reasons
    for kind, pattern in CONTENT_PATTERNS.items():
        if pattern.search(data):
            reasons.append(kind)
    return reasons


def _working_tree(root: Path) -> Iterator[tuple[str, Reader]]:
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if set(rel.parts) & SKIP_DIRS or not p.is_file():
            continue

        def read(p=p):
            try:
                if p.stat().st_size > MAX_SCAN_BYTES:
                    return None
                return p.read_bytes()
            except OSError:
                return None

        yield str(rel), read


def _archive(zf: zipfile.ZipFile) -> Iterator[tuple[str, Reader]]:
    for info in zf.infolist():
        if info.is_dir():
            continue

        def read(info=info):
            return None if info.file_size > MAX_SCAN_BYTES else zf.read(info)

        yield info.filename, read


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        archive = Path(sys.argv[1])
        if not archive.exists():
            print(f"No such archive: {archive}", file=sys.stderr)
            return 2
        zf = zipfile.ZipFile(archive)
        entries, label = _archive(zf), str(archive)
    else:
        entries, label = _working_tree(root), "working tree"

    checked, bad = 0, []
    for path, read in entries:
        checked += 1
        reasons = findings(path, read)
        if reasons:
            bad.append((path, reasons))

    if bad:
        print(f"REFUSING: secrets found in {label}:", file=sys.stderr)
        for path, reasons in bad:
            print(f"  {path}  ({', '.join(reasons)})", file=sys.stderr)
        print(
            "\nRemove them, and ROTATE any credential that was already "
            "distributed —\ndeleting the file does not un-share a key that "
            "someone already has.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no secrets in {label} ({checked} paths checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
