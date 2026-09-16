"""Refuses to build a distributable archive that contains secrets.

WHY: `.gitignore` and `.dockerignore` both correctly exclude `.env`, and
both were correctly configured — yet a hand-built zip of this project was
distributed with a live GEMINI_API_KEY and HF_TOKEN inside it. Neither
ignore file governs `zip -r`. The gap was never in the configuration; it
was that nothing checked the artefact actually being handed over.

Run before sharing a build:

    python scripts/check_archive.py            # check the working tree
    python scripts/check_archive.py bao.zip    # check a built archive
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# Files that must never ship, and secret-bearing names to reject anywhere
# in a path. `.env.example` is explicitly allowed: it is the template.
FORBIDDEN_NAMES = {".env", "credentials.json", "service-account.json", "id_rsa"}
FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12")
ALLOWED = {".env.example", ".env.template"}


def offending(paths) -> list[str]:
    bad = []
    for raw in paths:
        name = Path(raw).name
        if name in ALLOWED:
            continue
        if name in FORBIDDEN_NAMES or name.startswith(".env."):
            bad.append(raw)
        elif name.endswith(FORBIDDEN_SUFFIXES):
            bad.append(raw)
    return bad


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        archive = Path(sys.argv[1])
        if not archive.exists():
            print(f"No such archive: {archive}", file=sys.stderr)
            return 2
        with zipfile.ZipFile(archive) as zf:
            paths = zf.namelist()
        label = str(archive)
    else:
        paths = [
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and ".git/" not in str(p)
        ]
        label = "working tree"

    bad = offending(paths)
    if bad:
        print(f"REFUSING: secret-bearing files found in {label}:", file=sys.stderr)
        for path in bad:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nRemove them, and ROTATE any credential that was already "
            "distributed —\ndeleting the file does not un-share a key that "
            "someone already has.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no secret-bearing files in {label} ({len(paths)} paths checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
