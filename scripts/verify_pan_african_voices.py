"""Bao AI - Which of the 14 pan-African languages can actually be spoken?

Run this where Hugging Face is reachable. It does NOT assume: every code
below is a candidate to be checked, because this project has twice been
bitten by assuming a voice exists. The SA MMS table started at eleven
entries and finished at two once it was actually queried.

    python scripts/verify_pan_african_voices.py            # needs HF_TOKEN
    python scripts/verify_pan_african_voices.py --edge     # also list edge-tts

WHY THE CODES ARE NOT ALWAYS THE CLASSIFIER'S LABELS. The pan-African
detector emits MasakhaNEWS label codes. MMS repositories are named by ISO
639-3, and a macrolanguage code and its variety code do not both exist.
Probed with an authenticated request on 2026-09-23:

    Swahili  mms-tts-swa  404     mms-tts-swh  exists  (Coastal Swahili)
    Oromo    mms-tts-orm  exists  mms-tts-gaz  404

The two go OPPOSITE ways, which is why both candidates are tried below
rather than a rule being applied. An earlier version of this docstring
said Oromo was published as gaz. It is not, and anything built on that
claim would have 404ed silently. Swahili resolving to swh is a real
linguistic choice rather than a typo to paper over: Bao speaks Coastal
Swahili specifically.

FRENCH IS DIFFERENT. French has excellent neural voices on edge-tts and
does not need MMS. It is in the classifier because MasakhaNEWS includes
it, not because it is under-resourced. Do not present a French voice as
an under-resourced-language contribution.

WHAT A PASS HERE DOES NOT MEAN. A reachable repo means a voice exists,
not that it is good. MMS voices vary widely in quality, and a bad voice
in a language nobody on the team speaks is worse than no voice, because
nobody will notice it is bad. Anything that passes still needs a
first-language speaker to listen before it ships.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

# (display name, [candidate ISO 639-3 codes in preference order])
CANDIDATES = [
    ("Amharic",         ["amh"]),
    ("French",          ["fra"]),
    ("Hausa",           ["hau"]),
    ("Igbo",            ["ibo"]),
    ("Lingala",         ["lin"]),
    ("Luganda",         ["lug"]),
    ("Oromo",           ["gaz", "orm"]),   # see module docstring
    ("Nigerian Pidgin", ["pcm"]),
    ("Kirundi",         ["run"]),
    ("Shona",           ["sna"]),
    ("Somali",          ["som"]),
    ("Swahili",         ["swh", "swa"]),   # see module docstring
    ("Tigrinya",        ["tir"]),
    ("Yoruba",          ["yor"]),
]

# Microsoft neural locales worth checking by hand against the published
# edge-tts list. Listed as candidates only - this script does not verify
# them, because `edge-tts --list-voices` is the authoritative check and
# needs no token.
EDGE_CANDIDATE_LOCALES = {
    "Amharic": "am-ET", "French": "fr-FR", "Somali": "so-SO",
    "Swahili": "sw-KE / sw-TZ",
}


def repo_exists(code: str, token: str | None) -> bool | None:
    """True / False, or None when the check itself failed (network, auth)."""
    url = f"https://huggingface.co/api/models/facebook/mms-tts-{code}"
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status == 200
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        print(f"  ! {code}: HTTP {error.code}", file=sys.stderr)
        return None
    except Exception as error:  # noqa: BLE001 - network shapes vary
        print(f"  ! {code}: {type(error).__name__}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", action="store_true",
                        help="also print edge-tts locales to check by hand")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("No HF_TOKEN in the environment. Public repos may still "
              "resolve; gated ones will not.\n", file=sys.stderr)

    print("=" * 66)
    print("PAN-AFRICAN VOICE AVAILABILITY (facebook/mms-tts-<code>)")
    print("=" * 66)

    available, missing, unknown = {}, [], []
    for name, codes in CANDIDATES:
        for code in codes:
            result = repo_exists(code, token)
            if result is True:
                available[name] = code
                note = "  <- variety-specific" if code == "swh" else ""
                print(f"  {name:<17} {code}   available{note}")
                break
            if result is None:
                unknown.append(name)
                print(f"  {name:<17} {code}   CHECK FAILED")
                break
        else:
            missing.append(name)
            print(f"  {name:<17} {'/'.join(codes):<5} none found")

    print("=" * 66)
    print(f"{len(available)} available, {len(missing)} missing, "
          f"{len(unknown)} unverified, of {len(CANDIDATES)}")

    if available:
        # This used to say "paste into config.toml under
        # [speech.pan_african_mms_codes]" - a key nothing reads, so following
        # the instruction changed nothing. These codes are already the
        # defaults; overrides go in the table Settings.mms_codes merges.
        print("\nThese are the defaults in bao/core/config.py "
              "(DEFAULT_PAN_AFRICAN_MMS_CODES).")
        print("To override one, add it to mms_codes under [languages] in config.toml:")
        for name, code in available.items():
            print(f'  "{name}" = "{code}"')

    if args.edge:
        print("\nAlso check by hand - `edge-tts --list-voices | grep <prefix>`:")
        for name, locale in EDGE_CANDIDATE_LOCALES.items():
            print(f"  {name:<17} {locale}")

    print("\nReminder: a reachable repo is not a good voice. Have a "
          "first-language\nspeaker listen before any of these ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
