"""Compare language detectors head-to-head on the same held-out data.

    python scripts/compare_detectors.py --test-set eval/detector_test.csv

Written to settle a real disagreement empirically rather than by argument.

The proposal: replace the TFLite LSTM (word-level tokenizer) with TF-IDF
character n-grams + a linear classifier, on the grounds of out-of-vocabulary
robustness and artifact size. The objection: it replaces a validated
component with an unvalidated one.

Both positions are reasonable and neither is settled by reasoning, because
the two models are good at different things:

  - Character n-grams see inside words, so they handle typos, slang and
    unseen vocabulary. A live session had a user type "axuxeni" for
    "avuxeni"; the word-level path could only emit <OOV> and fell through
    to English.
  - The LSTM reads word ORDER, which a bag of n-grams discards. That was
    the stated reason for choosing it over a bag-of-words model, and the
    case it was chosen for — separating Sepedi, Sesotho and Setswana —
    is the hardest one in the set.

Which effect wins on this corpus is a measurement, not an opinion. Run it.

The test set is a CSV with columns `text,language`. It MUST be the
held-out 10% split — not the training or validation portion, and not
anything either model has been tuned against. Both models see exactly the
same rows.

Reports macro F1 (not accuracy: the corpus is imbalanced about 4.3 to 1),
per-language F1, a confusion matrix, and a robustness section on inputs
that are short, misspelled, or out-of-vocabulary — which is the specific
claim being made for character n-grams.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bao.core.config import Settings  # noqa: E402
from bao.services.language_detector import (  # noqa: E402
    HeuristicLanguageDetector,
    SklearnLanguageDetector,
    TFLiteLanguageDetector,
    has_tflite_support,
)


def load_test_set(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            language = (row.get("language") or "").strip()
            if text and language:
                rows.append((text, language))
    return rows


def macro_f1(gold: list[str], predicted: list[str]) -> tuple[float, dict[str, float]]:
    """Macro-averaged F1, computed here so the script has no hard sklearn
    dependency. Macro, not accuracy: the corpus is imbalanced about 4.3 to
    1, so accuracy would let a model score well while failing the smallest
    languages — which are the ones this project exists to serve.
    """
    labels = sorted(set(gold))
    per_label: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted, strict=True) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return (sum(per_label.values()) / len(per_label) if per_label else 0.0), per_label


def evaluate(detector, rows: list[tuple[str, str]]) -> dict:
    gold, predicted = [], []
    started = time.perf_counter()
    for text, language in rows:
        gold.append(language)
        predicted.append(detector.detect(text).language)
    elapsed_ms = (time.perf_counter() - started) * 1000

    overall, per_label = macro_f1(gold, predicted)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for g, p in zip(gold, predicted, strict=True):
        confusion[g][p] += 1

    return {
        "n": len(rows),
        "macro_f1": overall,
        "per_label": per_label,
        "confusion": confusion,
        "accuracy": sum(1 for g, p in zip(gold, predicted, strict=True) if g == p) / len(rows),
        "ms_per_item": elapsed_ms / len(rows),
    }


def robustness_probe(detector, rows: list[tuple[str, str]], drop_rate: int = 7) -> dict:
    """The specific claim for character n-grams: they survive typos and
    unseen words. Tested by corrupting every Nth character, which is a
    crude but even-handed stand-in for the misspellings real users type.

    Both models get exactly the same corrupted inputs.
    """
    def corrupt(text: str) -> str:
        return "".join(c for i, c in enumerate(text) if i % drop_rate != 0)

    corrupted = [(corrupt(text), language) for text, language in rows]
    return evaluate(detector, corrupted)


def render(name: str, result: dict) -> str:
    lines = [
        f"\n{name}",
        "-" * 62,
        f"  macro F1      {result['macro_f1']:.4f}",
        f"  accuracy      {result['accuracy']:.4f}   ({result['n']} items)",
        f"  latency       {result['ms_per_item']:.3f} ms per item",
        "  per-language F1:",
    ]
    for label, score in sorted(result["per_label"].items(), key=lambda kv: kv[1]):
        lines.append(f"      {label:<12} {score:.4f}")
    return "\n".join(lines)


def render_confusion(result: dict, top: int = 6) -> str:
    """The most useful single output for the report: which languages get
    mistaken for which. Expect the Nguni cluster and the Sotho-Tswana
    cluster to dominate — if they don't, something is wrong.
    """
    pairs = []
    for gold, counts in result["confusion"].items():
        for predicted, n in counts.items():
            if predicted != gold:
                pairs.append((n, gold, predicted))
    pairs.sort(reverse=True)
    if not pairs:
        return "\n  no confusions\n"
    lines = ["\n  most frequent confusions (gold -> predicted):"]
    for n, gold, predicted in pairs[:top]:
        lines.append(f"      {gold:<12} -> {predicted:<12} {n}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-set", type=Path, required=True,
        help="CSV with columns text,language. MUST be the held-out split — "
             "not training or validation data, and not anything either model "
             "was tuned against.",
    )
    parser.add_argument("--sklearn-model", type=Path, default=None,
                        help="joblib bundle for the sklearn detector.")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()

    if not args.test_set.exists():
        print(f"No such test set: {args.test_set}", file=sys.stderr)
        return 2

    rows = load_test_set(args.test_set)
    if not rows:
        print("Test set is empty or malformed (need columns: text,language).", file=sys.stderr)
        return 2

    settings = Settings()
    detectors: list[tuple[str, object]] = [("Heuristic (keyword baseline)", HeuristicLanguageDetector())]

    if has_tflite_support(settings.classifier_model_path):
        detectors.append((
            "TFLite LSTM (word-level)",
            TFLiteLanguageDetector(settings.classifier_model_path,
                                   tokenizer_config_path=settings.tokenizer_config_path),
        ))
    else:
        print("\n  ! TFLite unavailable — install requirements-ml.txt, or this "
              "comparison is missing the incumbent.\n")

    sklearn_path = args.sklearn_model or (REPO / "models" / "language_detector_sklearn.joblib")
    sklearn_detector = SklearnLanguageDetector(str(sklearn_path))
    if sklearn_detector.is_available():
        detectors.append(("TF-IDF char n-gram + linear", sklearn_detector))
    else:
        print(f"\n  ! No sklearn bundle at {sklearn_path} — train it first.\n")

    if len(detectors) < 2:
        print("Need at least two detectors to compare.", file=sys.stderr)
        return 2

    print(f"\nDETECTOR COMPARISON — {len(rows)} held-out items")
    print("=" * 62)

    results = {}
    for name, detector in detectors:
        result = evaluate(detector, rows)
        results[name] = result
        print(render(name, result))
        print(render_confusion(result))

    if not args.skip_robustness:
        print("\nROBUSTNESS — same items with every 7th character removed")
        print("=" * 62)
        print("(the specific claim made for character n-grams)")
        for name, detector in detectors:
            probe = robustness_probe(detector, rows)
            clean = results[name]["macro_f1"]
            print(f"  {name:<32} {clean:.4f} -> {probe['macro_f1']:.4f} "
                  f"({probe['macro_f1'] - clean:+.4f})")

    print("\n" + "=" * 62)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True)
    best, best_result = ranked[0]
    print(f"  Highest macro F1: {best} ({best_result['macro_f1']:.4f})")
    if len(ranked) > 1:
        gap = best_result["macro_f1"] - ranked[1][1]["macro_f1"]
        print(f"  Margin over next: {gap:+.4f}")
        if gap < 0.01:
            print("  That margin is small. On a difference this size, prefer the")
            print("  component that is already validated and already documented.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
