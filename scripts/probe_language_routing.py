"""Does a message in each supported language get answered in that language?

    python scripts/probe_language_routing.py            # offline routing only
    python scripts/probe_language_routing.py --online   # also calls Gemini

This asks the one question the rest of the test suite never asks end to
end. The unit tests check detection in isolation and voice coverage in
isolation; neither follows a sentence all the way through the decision
that actually matters to a user, which is *what language does the reply
come back in*.

That decision is not the detector's output. It is `reply_language`, and
the two differ on purpose: a detection below `min_detection_confidence`
is reported honestly and then NOT acted on, because answering a 42% guess
in the wrong language is worse than answering in English. So a language
can be detected perfectly and still never be replied to in — which is
invisible unless something walks the whole path.

Offline mode is the default here deliberately. It exercises security,
detection, the confidence floor, retrieval and voice selection without
spending an API call per language, and those are the stages where the
routing bugs live. `--online` adds the generation call for a real
end-to-end check when you want to confirm Gemini honours the instruction.

Exit codes:
    0  every language routed to itself
    1  at least one language was detected but not replied to in
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bao.bootstrap import build_orchestrator  # noqa: E402
from bao.core.config import Settings  # noqa: E402
from bao.services.language_detector import CompositeLanguageDetector  # noqa: E402

# One sentence per language, not one word. The classifier was trained on
# NCHLT news sentences, so a bare greeting is out of distribution for it —
# single words are what the keyword fallback is for, and testing routing
# with them measures the fallback rather than the model.
SA_SAMPLES = {
    "English": "Please tell me how I can register to vote in this country.",
    "Afrikaans": "Kan jy my asseblief help om my identiteitsdokument te kry.",
    "isiZulu": "Sawubona, ngicela ungisize ngolwazi mayelana nezempilo.",
    "isiXhosa": "Molo, ndicela uncedo malunga neencwadi zesikolo sabantwana.",
    "siSwati": "Sawubona, ngicela usisite ngelwati lolumayelana nemsebenti.",
    "isiNdebele": "Lotjhani, ngibawa ningisize ngelwazi malungana nezamaphilo.",
    "Sepedi": "Thobela, ke kgopela thušo mabapi le dithuto tša sekolo.",
    "Sesotho": "Lumela, ke kopa thuso mabapi le litokelo tsa basebetsi.",
    "Setswana": "Dumela, ke kopa thuso malebana le ditshwanelo tsa badiri.",
    "Xitsonga": "Avuxeni, ndzi kombela mpfuno hi ta swa rihanyo ni vutomi.",
    "Tshivenda": "Ndaa, ndi humbela thuso nga ha mafhungo a mutakalo.",
}

PAN_AFRICAN_SAMPLES = {
    "Amharic": "ሰላም፣ እባክዎን ስለ ጤና አገልግሎት መረጃ ይስጡኝ።",
    "French": "Bonjour, pouvez-vous m'aider à trouver des informations sur la santé.",
    "Hausa": "Sannu, ina neman taimako game da harkokin lafiya a yankinmu.",
    "Igbo": "Ndewo, biko nyere m aka banyere ozi gbasara ahụike na obodo anyị.",
    "Lingala": "Mbote, nasengi lisalisi mpo na makambo ya bokolongono ya nzoto.",
    "Luganda": "Oli otya, nsaba obuyambi ku bikwata ku by'obulamu mu kitundu kyaffe.",
    "Oromo": "Akkam jirtu, odeeffannoo fayyaa irratti gargaarsa barbaada.",
    "Nigerian Pidgin": "How you dey, abeg I wan know about health care for our area.",
    "Kirundi": "Amakuru, ndasaba ubufasha ku bijanye n'amagara y'abantu.",
    "Shona": "Mhoro, ndiri kukumbira rubatsiro nezve hutano hwevanhu munzvimbo yedu.",
    "Somali": "Salaan, waxaan doonayaa macluumaad ku saabsan caafimaadka bulshada.",
    "Swahili": "Habari yako, naomba msaada kuhusu huduma za afya katika eneo letu.",
    "Tigrinya": "ሰላም፣ ብዛዕባ ኣገልግሎት ጥዕና ሓበሬታ እደሊ ኣለኹ።",
    "Yoruba": "Bawo ni, jọwọ ran mi lọwọ nipa alaye nipa ilera ni agbegbe wa.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true",
                        help="also call Gemini, to confirm the reply really comes back in-language")
    args = parser.parse_args()

    settings = Settings()
    _, orchestrator = build_orchestrator(settings, with_document_retriever=False)

    detector = orchestrator.language_detector
    if isinstance(detector, CompositeLanguageDetector):
        # Probe what the app CAN do, not what config.toml currently ships.
        # The pan-African languages are off by default, and a probe that
        # inherited that would report fourteen failures for a feature that
        # was simply switched off.
        detector.enabled = True
        samples = {**SA_SAMPLES, **PAN_AFRICAN_SAMPLES}
        print("Pan-African detector: ON for this probe\n")
    else:
        samples = dict(SA_SAMPLES)
        print("Pan-African detector: unavailable — probing the 11 South African "
              "languages only\n")

    print(f"{'language':<17} {'detected':<17} {'conf':>5}  {'replies in':<17} verdict")
    print("-" * 74)

    failures: list[tuple[str, str, str]] = []
    for expected, text in samples.items():
        result = orchestrator.handle(text, force_offline=not args.online)
        replied = result.reply_language or result.detected_language
        ok = replied == expected
        if not ok:
            failures.append((expected, result.detected_language, replied))
        print(f"{expected:<17} {result.detected_language:<17} "
              f"{result.confidence:>4.0%}  {replied:<17} {'ok' if ok else 'MISROUTED'}")

    total = len(samples)
    print(f"\n{total - len(failures)}/{total} languages are replied to in their own language.")

    if failures:
        print("\nMisrouted — these are answered in a language the user did not write in:")
        for expected, detected, replied in failures:
            reason = ("detected correctly but below the confidence floor"
                      if detected == expected else f"detected as {detected}")
            print(f"  - {expected}: {reason}, replied in {replied}")
        print("\n  The floor is config.toml [model] min_detection_confidence. Lowering it "
              "\n  trades wrong-language replies for wrong-language BADGES; see the note on "
              "\n  Settings.min_detection_confidence before changing it.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
