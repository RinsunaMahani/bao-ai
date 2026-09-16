"""Bao AI - Does the pan-African detector survive the input users type?

WHY THIS EXISTS. The pan-African classifier is trained on MasakhaNEWS,
which is a corpus of news ARTICLES. Bao's users type short public-service
questions. Those are different distributions, and a test-set score
computed on held-out news says nothing about the second one. The corpus
is also small and uneven - roughly 3,000 articles per language, with
Lingala, Kirundi and Somali short of that - so the languages with the
least data are where a domain shift should bite first.

MEASURED, on the pilot set below: 67% on one-to-three-word input against
100% on sentence-length input. The failures are not random:

  Mbote / Melezi  (Lingala)         -> Swahili
  How far / Abeg help me  (Pidgin)  -> Hausa
  Merci beaucoup  (French)          -> Hausa

Lingala is one of the three under-represented languages in the corpus and
loses to the best-resourced Bantu language in the label set. Nigerian
Pidgin is English-lexified, so short Pidgin looks nothing like the news
prose the model learned it from. Both are the domain gap showing up
exactly where the data predicts.

SAMPLE SIZE. Nineteen items is a pilot, not a benchmark, and it is not a
basis for a headline accuracy claim. It is a basis for saying the gap is
real and worth measuring properly: ~100 short queries per language,
reported as accuracy by input length, with a confusion matrix. Treat the
numbers here as motivation for that work, and say so if asked.

    python eval_detector_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL = Path(__file__).parent / "models" / "language_detector_pan_african.joblib"

# Greetings and courtesy phrases: the shortest real input there is, and
# disproportionately what a user opens a conversation with.
SHORT = [
    ("Sannu", "hau"), ("Nagode", "hau"),
    ("Habari", "swa"), ("Asante sana", "swa"), ("Naomba msaada", "swa"),
    ("Bawo ni", "yor"), ("E se", "yor"),
    ("Kedu", "ibo"), ("Daalu", "ibo"),
    ("Mbote", "lin"), ("Melezi", "lin"),
    ("How far", "pcm"), ("Abeg help me", "pcm"),
    ("Bonjour", "fra"), ("Merci beaucoup", "fra"),
]

# Full questions - still conversational, but long enough to carry the
# distributional signal the model was trained on.
SENTENCE = [
    ("Naomba msaada wa kupata namba ya polisi tafadhali", "swa"),
    ("Abeg I wan know the number wey I go call for police emergency", "pcm"),
    ("Ina son in san lambar wayar yan sanda don taimako cikin gaggawa", "hau"),
    ("Je voudrais connaitre le numero de telephone de la police pour une urgence", "fra"),
]


def evaluate(pipeline, labels, cases, name):
    def decode(prediction):
        return labels[int(prediction)] if not isinstance(prediction, str) else prediction

    correct = 0
    misses = []
    for text, gold in cases:
        predicted = decode(pipeline.predict([text])[0])
        if predicted == gold:
            correct += 1
        else:
            misses.append((text, gold, predicted))

    print(f"\n{name}")
    for text, gold, predicted in misses:
        print(f"   MISS  {text[:44]:<44} gold={gold}  pred={predicted}")
    print(f"   {correct}/{len(cases)} correct ({correct / len(cases) * 100:.0f}%)")
    return correct / len(cases)


def main() -> int:
    if not MODEL.exists():
        print(f"Model not found: {MODEL}", file=sys.stderr)
        return 1
    try:
        import joblib
    except ImportError:
        print("joblib not installed.", file=sys.stderr)
        return 1

    bundle = joblib.load(MODEL)
    pipeline, labels = bundle["pipeline"], bundle["labels"]

    print("=" * 70)
    print(f"PAN-AFRICAN DETECTOR - INPUT LENGTH ROBUSTNESS ({len(labels)} languages)")
    print("=" * 70)

    short = evaluate(pipeline, labels, SHORT, "Short chat-style input (1-3 words):")
    sentence = evaluate(pipeline, labels, SENTENCE, "Sentence-length input:")

    print("\n" + "=" * 70)
    print(f"Short input {short * 100:.0f}%   vs   sentence-length {sentence * 100:.0f}%")
    if short < sentence:
        print(
            "\nAccuracy degrades on short input. The model was trained on news\n"
            "articles; users type short questions. A held-out news test score\n"
            "will NOT surface this, which is the argument for evaluating on\n"
            "user-style queries rather than only on the training corpus's own\n"
            "test split.\n\n"
            "Pilot only (19 items). Do not quote as a benchmark - quote it as\n"
            "the reason to build one."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
