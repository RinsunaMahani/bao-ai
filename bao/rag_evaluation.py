"""Bao AI - Retrieval Evaluation (precision, recall, F1, threshold sweep).

The existing `evaluate.py` self-consistency check answers "can the
knowledge base retrieve its own rows?" — a useful regression test, but it
can't detect the failure mode that actually matters in production: an
unrelated query clearing the similarity threshold and getting a
confidently wrong answer (the documented "airspeed velocity of an unladen
swallow" case, which scores 0.443 against a currency fact — nearly 3x the
0.15 default threshold).

This module answers the harder question, against a hand-labelled set
(`eval/rag_eval.csv`) with three query categories:

  - exact:      verbatim knowledge-base questions (should match, easy)
  - paraphrase: same intent, different words (should match, hard — this is
                where TF-IDF's lexical-overlap limitation bites)
  - negative:   topically unrelated (should NOT match — this is where the
                threshold earns its keep)

The threshold sweep exists because the obvious "fix" for a false positive
at 0.443 — raising the threshold above it — would also reject legitimate
paraphrases. The right threshold is an empirical trade-off between
precision and recall, not a number picked to defeat one embarrassing
example. This produces the evidence for choosing it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from bao.knowledge.retriever import KnowledgeRetriever


@dataclass
class RagEvalCase:
    query: str
    expected_row: int  # -1 means "should not match anything"
    category: str


@dataclass
class ThresholdResult:
    threshold: float
    true_positives: int   # should match, did match, correct row
    false_positives: int  # should not match, but did (or matched wrong row)
    true_negatives: int   # should not match, correctly rejected
    false_negatives: int  # should match, but was rejected

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom else 0.0


def load_rag_eval_cases(csv_path: Path) -> list[RagEvalCase]:
    cases: list[RagEvalCase] = []
    with open(csv_path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            query = (row.get("query") or "").strip()
            if not query:
                continue
            cases.append(RagEvalCase(
                query=query,
                expected_row=int(row.get("expected_row", -1)),
                category=(row.get("category") or "").strip(),
            ))
    return cases


def evaluate_threshold(
    retriever: KnowledgeRetriever,
    cases: list[RagEvalCase],
    threshold: float,
    apply_coverage_gate: bool = True,
) -> ThresholdResult:
    """Scores one threshold. Uses `best_match_index()` (threshold-free) so
    the whole sweep runs off one similarity computation per query rather
    than rebuilding the retriever for every candidate threshold.

    `apply_coverage_gate` mirrors what `query_fact()` actually does in
    production, so these numbers describe the shipped system rather than a
    simplified stand-in. Set it False to measure similarity alone — useful
    for showing what the gate is worth, and that is exactly how the
    before/after figures in REVIEW.md were produced.
    """
    tp = fp = tn = fn = 0

    for case in cases:
        match = retriever.best_match_index(case.query)
        matched_row, score = match if match else (-1, 0.0)
        retrieved = score >= threshold
        if retrieved and apply_coverage_gate:
            retrieved = retriever.query_coverage(case.query) >= retriever.min_coverage
        should_match = case.expected_row >= 0

        if should_match and retrieved and matched_row == case.expected_row:
            tp += 1
        elif should_match and not retrieved:
            fn += 1
        elif should_match and retrieved:
            # Retrieved something, but the WRONG row — counted as a false
            # positive, not a true positive. A confidently wrong answer is
            # worse than no answer, so it must not score as a success.
            fp += 1
        elif not should_match and retrieved:
            fp += 1
        else:
            tn += 1

    return ThresholdResult(threshold=threshold, true_positives=tp, false_positives=fp,
                           true_negatives=tn, false_negatives=fn)


def sweep_thresholds(retriever: KnowledgeRetriever, cases: list[RagEvalCase],
                     thresholds: list[float] | None = None) -> list[ThresholdResult]:
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 13)]  # 0.05 .. 0.60
    return [evaluate_threshold(retriever, cases, t) for t in thresholds]


def per_category_breakdown(retriever: KnowledgeRetriever, cases: list[RagEvalCase],
                           threshold: float,
                           apply_coverage_gate: bool = True) -> dict[str, tuple[int, int]]:
    """Returns {category: (correct, total)} — the aggregate numbers hide
    which category is failing, and exact/paraphrase/negative fail for
    completely different reasons.

    `apply_coverage_gate` MUST default to the same value as
    `evaluate_threshold`, and for a specific reason. Until this parameter
    existed, this function scored similarity alone while
    `evaluate_threshold` scored similarity AND the coverage gate — so the
    per-category table and the precision/recall/F1 headline printed side
    by side in the README were describing two DIFFERENT systems. The
    ungated view made negative rejection look far worse than it ships
    (20/40 rather than 37/40) and paraphrase recall look far better
    (11/15 rather than 4/15), which inverted the apparent weakness of the
    whole retrieval layer. Keep the two defaults in sync.
    """
    breakdown: dict[str, list[int]] = {}
    for case in cases:
        match = retriever.best_match_index(case.query)
        matched_row, score = match if match else (-1, 0.0)
        retrieved = score >= threshold
        if retrieved and apply_coverage_gate:
            retrieved = retriever.query_coverage(case.query) >= retriever.min_coverage
        should_match = case.expected_row >= 0

        correct = (
            (should_match and retrieved and matched_row == case.expected_row)
            or (not should_match and not retrieved)
        )
        bucket = breakdown.setdefault(case.category, [0, 0])
        bucket[1] += 1
        if correct:
            bucket[0] += 1
    return {k: (v[0], v[1]) for k, v in breakdown.items()}


def to_markdown(results: list[ThresholdResult], breakdown: dict[str, tuple[int, int]],
                current_threshold: float) -> str:
    lines = [
        "### Retrieval precision / recall by threshold",
        "",
        "| Threshold | Precision | Recall | F1 | False-positive rate | TP | FP | TN | FN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        marker = "  <- current default" if abs(r.threshold - current_threshold) < 1e-9 else ""
        lines.append(
            f"| {r.threshold:.2f} | {r.precision:.1%} | {r.recall:.1%} | {r.f1:.3f} | "
            f"{r.false_positive_rate:.1%} | {r.true_positives} | {r.false_positives} | "
            f"{r.true_negatives} | {r.false_negatives} |{marker}"
        )

    best = max(results, key=lambda r: r.f1)
    lines += [
        "",
        f"- Best F1: **{best.f1:.3f} at threshold {best.threshold:.2f}** "
        f"(precision {best.precision:.1%}, recall {best.recall:.1%})",
        f"- Current default threshold: {current_threshold:.2f}",
        "",
        f"### Per-category accuracy at the current threshold ({current_threshold:.2f})",
        "",
    ]
    for category, (correct, total) in sorted(breakdown.items()):
        pct = (correct / total * 100) if total else 0.0
        lines.append(f"- {category}: {correct}/{total} ({pct:.0f}%)")
    return "\n".join(lines)
