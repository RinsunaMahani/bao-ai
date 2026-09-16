"""Score retrieval on a HELD-OUT set — one never used for tuning.

    python scripts/holdout_eval.py

`eval/rag_eval.csv` (70 queries) was used to choose `min_query_coverage`,
which makes it a development set. Precision 0.826 measured on it means
"measured on the set it was tuned on", not "expected on new questions".
A held-out set is what turns the second claim into a measurement.

**This file ships empty on purpose.** Whoever writes the holdout must not
be whoever tuned the threshold, or the exercise is circular — and the
threshold was tuned by an AI assistant, so it should not also author the
questions it is graded on.

To fill it, add rows to `eval/rag_holdout.csv`:

    query,expected_row,category,notes
    how do i get a new id book,-1,negative,not in the KB
    what number do i phone in an emergency,20,paraphrase,

  query        what a real user might type — NOT copied from the CSV
  expected_row 0-based row in data/african_data.csv, or -1 if the
               knowledge base genuinely cannot answer it
  category     exact | paraphrase | negative

Aim for roughly a third paraphrases of things the KB does cover, and half
negatives — plausible South African questions it does NOT cover. The
negatives matter most: they are what catches confidently-wrong answers.

Write them, then run this once and report the number. Looking at the
result and then adjusting the threshold turns it back into a development
set.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bao.core.config import Settings  # noqa: E402
from bao.knowledge.retriever import KnowledgeRetriever  # noqa: E402

HOLDOUT = REPO / "eval" / "rag_holdout.csv"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> int:
    rows = []
    if HOLDOUT.exists():
        with open(HOLDOUT, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if (r.get("query") or "").strip()]

    if not rows:
        print(__doc__)
        print(f"\n  {HOLDOUT} is empty — nothing to score yet.\n")
        return 2

    settings = Settings()
    retriever = KnowledgeRetriever(
        data_path=settings.knowledge_base_path,
        threshold=settings.similarity_threshold,
        min_coverage=settings.min_query_coverage,
    )

    tp = fp = tn = fn = 0
    mistakes = []
    for row in rows:
        query = row["query"].strip()
        expected = int(row["expected_row"])
        fact = retriever.lookup(query)
        retrieved_row = fact.row if fact else -1

        if expected >= 0 and retrieved_row == expected:
            tp += 1
        elif expected >= 0 and retrieved_row < 0:
            fn += 1
            mistakes.append(("missed", query, expected, retrieved_row))
        elif expected < 0 and retrieved_row < 0:
            tn += 1
        else:
            fp += 1
            mistakes.append(("wrong", query, expected, retrieved_row))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_lo, p_hi = wilson(tp, tp + fp)

    print(f"\nHELD-OUT RETRIEVAL — {len(rows)} queries")
    print("-" * 60)
    print(f"  TP {tp}   FP {fp}   TN {tn}   FN {fn}")
    print(f"  precision {precision:.3f}   95% CI [{p_lo:.3f}, {p_hi:.3f}]  (n={tp + fp})")
    print(f"  recall    {recall:.3f}")
    print(f"  F1        {f1:.3f}")
    print("-" * 60)
    print("  Dev set (eval/rag_eval.csv, used for tuning): precision 0.826, F1 0.731")
    print("  A drop here is the honest generalisation gap, and worth reporting as one.")

    if mistakes:
        print(f"\n  {len(mistakes)} mistake(s):")
        for kind, query, expected, got in mistakes[:10]:
            print(f"    [{kind}] {query!r} expected row {expected}, got {got}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
