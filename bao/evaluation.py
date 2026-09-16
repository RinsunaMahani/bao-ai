"""Evaluation harness for Bao AI's retrieval and language detection.

Two evaluations are supported:

1. **Retrieval self-consistency** (``evaluate_retrieval_self_consistency``):
   for every row in the offline knowledge base, query the system with that
   row's own ``Question`` text and check whether that same row comes back
   as the top hit. This needs no external labelled data — generated
   automatically from whichever CSV is on disk — and is a standard
   sanity/regression check: if the system can't find a fact when asked
   almost verbatim, it won't find it when asked a real paraphrase either.
   Re-run this after any change to the knowledge base or vectorizer.

   NOTE: this module previously imported a `top_match` function from
   `knowledge.py` that was never actually defined there — the evaluation
   harness has never successfully imported, let alone run, until this
   migration. It's rebuilt here against `KnowledgeRetriever.best_match_index`,
   which is the real equivalent.

2. **Language detector accuracy** (``evaluate_language_detector``): given a
   list of ``(text, expected_label)`` pairs you supply, reports overall
   accuracy, a per-language breakdown, and mean confidence for correct vs.
   incorrect predictions. This genuinely needs real labelled examples — see
   ``eval/README.md``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from bao.knowledge.retriever import KnowledgeRetriever
from bao.services.language_detector import LanguageDetector


@dataclass
class RetrievalEvalReport:
    total: int
    correct: int
    mean_similarity_correct: float
    mean_similarity_incorrect: float
    misses: list[tuple[str, str, str]]  # (question, expected_answer, retrieved_answer)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_markdown(self) -> str:
        lines = [
            "### Retrieval self-consistency",
            "",
            f"- Knowledge base rows evaluated: {self.total}",
            f"- Top-1 self-retrieval accuracy: {self.accuracy:.1%} ({self.correct}/{self.total})",
            f"- Mean similarity on correct matches: {self.mean_similarity_correct:.3f}",
            "",
            "> This is a REGRESSION CHECK, not an accuracy claim. It asks the "
            "knowledge base its own questions verbatim, so ~100% is the "
            "expected result and a drop means something broke. It says "
            "nothing about paraphrases or about questions the KB cannot "
            "answer. For the number worth reporting, run:",
            ">",
            "> `python evaluate.py --rag-eval eval/rag_eval.csv`",
        ]
        if self.misses:
            lines.append(f"- Mean similarity on missed matches: {self.mean_similarity_incorrect:.3f}")
            lines.append("")
            lines.append("Misses:")
            for question, expected, retrieved in self.misses:
                lines.append(f"  - Q: {question!r} — expected {expected!r}, got {retrieved!r}")
        return "\n".join(lines)


def evaluate_retrieval_self_consistency(retriever: KnowledgeRetriever) -> RetrievalEvalReport:
    df = retriever.dataframe.reset_index(drop=True)
    correct = 0
    correct_scores: list[float] = []
    incorrect_scores: list[float] = []
    misses: list[tuple[str, str, str]] = []

    for row_index, row in df.iterrows():
        result = retriever.best_match_index(str(row["Question"]))
        if result is None:
            # Counted in `total`, so silently skipping it lowered the
            # reported accuracy while the row appeared in neither `correct`
            # nor `misses` — a figure that cannot be reconciled with its own
            # explanation. Record it as the miss it is.
            incorrect_scores.append(0.0)
            misses.append((str(row["Question"]), str(row["Answer"]), "(no match returned)"))
            continue
        best_index, similarity = result
        if best_index == row_index:
            correct += 1
            correct_scores.append(similarity)
        else:
            incorrect_scores.append(similarity)
            misses.append((str(row["Question"]), str(row["Answer"]), str(df.iloc[best_index]["Answer"])))

    total = len(df)
    return RetrievalEvalReport(
        total=total,
        correct=correct,
        mean_similarity_correct=(sum(correct_scores) / len(correct_scores)) if correct_scores else 0.0,
        mean_similarity_incorrect=(sum(incorrect_scores) / len(incorrect_scores)) if incorrect_scores else 0.0,
        misses=misses,
    )


@dataclass
class DetectorEvalReport:
    total: int
    correct: int
    per_label: dict[str, tuple[int, int]]  # label -> (correct, total)
    mean_confidence_correct: float
    mean_confidence_incorrect: float
    misclassifications: list[tuple[str, str, str, float]]  # text, expected, predicted, confidence

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_markdown(self) -> str:
        lines = [
            "### Language detector accuracy",
            "",
            f"- Labelled examples evaluated: {self.total}",
            f"- Overall accuracy: {self.accuracy:.1%} ({self.correct}/{self.total})",
            f"- Mean confidence on correct predictions: {self.mean_confidence_correct:.1f}%",
            f"- Mean confidence on incorrect predictions: {self.mean_confidence_incorrect:.1f}%",
            "",
            "Per-language accuracy:",
        ]
        for label, (label_correct, label_total) in sorted(self.per_label.items()):
            pct = (label_correct / label_total * 100) if label_total else 0.0
            lines.append(f"  - {label}: {label_correct}/{label_total} ({pct:.0f}%)")
        if self.misclassifications:
            lines.append("")
            lines.append("Misclassifications:")
            for text, expected, predicted, confidence in self.misclassifications:
                lines.append(f"  - {text!r}: expected {expected}, predicted {predicted} ({confidence:.1f}%)")
        return "\n".join(lines)


def load_labelled_examples(csv_path: Path) -> list[tuple[str, str]]:
    """Load (text, label) pairs from a CSV with `text` and `label` columns."""
    examples: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                examples.append((text, label))
    return examples


def evaluate_language_detector(detector: LanguageDetector, examples: list[tuple[str, str]]) -> DetectorEvalReport:
    per_label: dict[str, list[int]] = {}
    correct = 0
    correct_conf: list[float] = []
    incorrect_conf: list[float] = []
    misclassifications: list[tuple[str, str, str, float]] = []

    for text, expected_label in examples:
        result = detector.detect(text)
        confidence_pct = result.confidence * 100
        bucket = per_label.setdefault(expected_label, [0, 0])
        bucket[1] += 1
        if result.language == expected_label:
            bucket[0] += 1
            correct += 1
            correct_conf.append(confidence_pct)
        else:
            incorrect_conf.append(confidence_pct)
            misclassifications.append((text, expected_label, result.language, confidence_pct))

    total = len(examples)
    return DetectorEvalReport(
        total=total,
        correct=correct,
        per_label={label: (counts[0], counts[1]) for label, counts in per_label.items()},
        mean_confidence_correct=(sum(correct_conf) / len(correct_conf)) if correct_conf else 0.0,
        mean_confidence_incorrect=(sum(incorrect_conf) / len(incorrect_conf)) if incorrect_conf else 0.0,
        misclassifications=misclassifications,
    )
