"""Try alternative MMS language codes for languages that have no voice.

    python scripts/probe_mms_codes.py

A confirmed 404 means no model exists at *that repository name*. It does not
prove no model exists for the language: MMS covers 1107 languages using its
own code set, which mostly follows ISO 639-3 but not always — some languages
appear under a macrolanguage code, a dialect code, or a script-suffixed
variant.

So before reporting a language as unsupported, this checks the plausible
alternatives. It only asks whether the repository resolves; it downloads
nothing.

Run it after `probe_tts.py` reports 404s. If everything here also 404s, the
"no offline voice" conclusion is evidenced rather than assumed — which is
the difference between a finding and a guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bao.core.config import Settings  # noqa: E402  (loads .env, so HF_TOKEN applies)

# Codes worth trying per language. ISO 639-3 first (what the app uses), then
# alternatives: macrolanguage codes, older 639-2/B forms, and the spellings
# MMS itself uses in its published language table.
CANDIDATES: dict[str, list[str]] = {
    "isiXhosa":   ["xho", "xh", "xos"],
    "Sesotho":    ["sot", "st", "sso", "nso"],
    "Setswana":   ["tsn", "tn", "tsw"],
    "Sepedi":     ["nso", "pdi", "sot", "nsa"],
    "Tshivenda":  ["ven", "ve", "vnd", "vem"],
    "siSwati":    ["ssw", "ss", "swz", "swi"],
    "isiNdebele": ["nbl", "nr", "nde", "ndl", "nel"],
    # Included because config.toml routes English through "afr", which 404s.
    # mms-tts-eng is known to exist, so this confirms the mapping is the bug.
    "English":    ["eng", "en"],
    "Afrikaans":  ["afr", "af"],
}


def repo_exists(code: str) -> tuple[bool, str]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    api = HfApi()
    name = f"facebook/mms-tts-{code}"
    try:
        api.model_info(name)
        return True, "exists"
    except RepositoryNotFoundError:
        return False, "404"
    except GatedRepoError:
        return False, "gated (needs access request)"
    except EntryNotFoundError:
        return False, "repo exists but is missing files"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def main() -> int:
    Settings()  # side effect: loads .env so HF_TOKEN is picked up

    print("\nSearching for MMS voices under alternative language codes")
    print("=" * 72)

    found: dict[str, str] = {}
    for language, codes in CANDIDATES.items():
        results = []
        hit = None
        for code in codes:
            ok, detail = repo_exists(code)
            results.append(f"{code}={'YES' if ok else detail}")
            if ok and hit is None:
                hit = code
        marker = f"-> USE {hit}" if hit else "-> none found"
        print(f"  {language:<12} {marker:<16} [{'  '.join(results)}]")
        if hit:
            found[language] = hit

    print("=" * 72)
    if not found:
        print("\n  No alternative codes resolved. The 'no offline voice' finding holds")
        print("  for these languages, and is now evidenced rather than assumed.\n")
        return 0

    print("\n  Found voices under different codes. Update [languages].mms_codes")
    print("  in config.toml:\n")
    for language, code in found.items():
        print(f'      "{language}" = "{code}"')
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
