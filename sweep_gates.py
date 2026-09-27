"""Bao AI - Two-dimensional gate sweep: similarity threshold x coverage gate.

`compare_retrievers.py` asks which BACKEND is better. This script asks a
prior question about a single backend: has the current backend been tuned
as far as it can go?

WHY THIS EXISTS. The retrieval layer has two gates, not one. Similarity
threshold was swept and tuned; the coverage gate was fixed at 0.7 by hand
and never swept. Reporting a one-dimensional sweep over a
two-dimensional configuration space understates how well-tuned the system
already is, and — more importantly — leaves open the objection that the
weak paraphrase recall is just a badly chosen constant. Sweeping both
closes that objection with evidence either way.

MEASURED RESULT on the 70-query set with TF-IDF and 38 KB rows, re-run
2026-09-27: F1 sits between 0.687 and 0.821 across all 24 combinations.
The best is coverage 0.7 at threshold 0.15 (F1 0.821); the shipped
threshold 0.25 scores 0.800. The difference is ONE paraphrase query, on
the set the thresholds were chosen on, so the default was left alone - an
earlier version of this note called 0.25 the global best, which was true
of the knowledge base at the time and is not now. The two gates trade
against each other: every setting that recovers paraphrase recall gives
back negative rejection. That is the evidence that further tuning of
TF-IDF is exhausted, and therefore that closing the paraphrase gap
requires a different REPRESENTATION rather than a different constant.

WHY IT MATTERS FOR THE SEMANTIC UPGRADE. The coverage gate is a
TF-IDF-specific remedy: it exists because a TF-IDF vectorizer silently
drops out-of-vocabulary terms, so a query can be scored on the words that
survived (see KnowledgeRetriever.query_coverage). Sentence-transformer
backends subword-tokenize and do not drop unknown words, so the failure
the gate was built for does not arise there. Carrying 0.7 over unchanged
would import a workaround into a backend that does not need it, and would
keep paraphrase recall suppressed for no reason. Re-run this script per
backend: the gate is a hyperparameter of the backend, not a constant of
the system.

    python sweep_gates.py                  # TF-IDF (default)
    python sweep_gates.py --semantic       # requires requirements-semantic.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bao.core.config import Settings
from bao.knowledge.embeddings import TfidfEmbeddings, has_semantic_support
from bao.knowledge.retriever import KnowledgeRetriever
from bao.rag_evaluation import load_rag_eval_cases

EVAL_CSV = Path(__file__).parent / "eval" / "rag_eval.csv"

COVERAGES = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)
THRESHOLDS = (0.15, 0.25, 0.35, 0.45)


def sweep(retriever: KnowledgeRetriever, cases) -> list[dict]:
    """Scores every (coverage, threshold) pair.

    Similarity and coverage are computed ONCE per query and reused across
    the grid. Rebuilding the retriever per cell would multiply runtime by
    24 and change nothing about the result.
    """
    precomputed = {}
    for case in cases:
        match = retriever.best_match_index(case.query)
        row, score = match if match else (-1, 0.0)
        precomputed[case.query] = (row, score, retriever.query_coverage(case.query))

    rows = []
    for coverage in COVERAGES:
        for threshold in THRESHOLDS:
            tp = fp = tn = fn = 0
            buckets: dict[str, list[int]] = {}
            for case in cases:
                row, score, cov = precomputed[case.query]
                retrieved = score >= threshold and cov >= coverage
                should_match = case.expected_row >= 0

                if should_match and retrieved and row == case.expected_row:
                    tp += 1
                    correct = True
                elif should_match and not retrieved:
                    fn += 1
                    correct = False
                elif should_match:
                    fp += 1  # retrieved the WRONG row: confidently wrong
                    correct = False
                elif retrieved:
                    fp += 1
                    correct = False
                else:
                    tn += 1
                    correct = True

                bucket = buckets.setdefault(case.category, [0, 0])
                bucket[1] += 1
                bucket[0] += int(correct)

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({
                "coverage": coverage, "threshold": threshold,
                "precision": precision, "recall": recall, "f1": f1,
                "buckets": buckets,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true",
                        help="sweep the sentence-embedding backend instead of TF-IDF")
    args = parser.parse_args()

    if not EVAL_CSV.exists():
        print(f"Evaluation set not found: {EVAL_CSV}", file=sys.stderr)
        return 1

    settings = Settings()
    cases = load_rag_eval_cases(EVAL_CSV)

    if args.semantic:
        if not has_semantic_support():
            print("sentence-transformers not installed. "
                  "pip install -r requirements-semantic.txt", file=sys.stderr)
            return 1
        from bao.knowledge.embeddings import SentenceEmbeddings
        backend, model = "Semantic", SentenceEmbeddings()
    else:
        backend, model = "TF-IDF", TfidfEmbeddings()

    retriever = KnowledgeRetriever(
        data_path=settings.knowledge_base_path,
        threshold=settings.similarity_threshold,
        embedding_model=model,
    )
    if not retriever.is_initialized:
        print("Retriever failed to initialize.", file=sys.stderr)
        return 1

    rows = sweep(retriever, cases)
    categories = sorted({c for r in rows for c in r["buckets"]})

    print(f"\n{'=' * 78}")
    print(f"GATE SWEEP - {backend} - {len(cases)} queries, {len(retriever.dataframe)} KB rows")
    print("=" * 78)
    header = f"{'cover':<7}{'thresh':<8}{'P':<8}{'R':<8}{'F1':<8}"
    header += "".join(f"{c:<13}" for c in categories)
    print(header)
    print("-" * 78)

    best = max(rows, key=lambda r: r["f1"])
    for row in rows:
        line = (f"{row['coverage']:<7.1f}{row['threshold']:<8.2f}"
                f"{row['precision']:<8.3f}{row['recall']:<8.3f}{row['f1']:<8.3f}")
        for cat in categories:
            got, total = row["buckets"].get(cat, (0, 0))
            line += f"{f'{got}/{total}':<13}"
        marker = "  <- best F1" if row is best else ""
        print(line + marker)

    print("=" * 78)
    shipped = next(
        (r for r in rows
         if abs(r["coverage"] - retriever.min_coverage) < 1e-9
         and abs(r["threshold"] - settings.similarity_threshold) < 1e-9),
        None,
    )
    print(f"Best F1 {best['f1']:.3f} at coverage={best['coverage']}, "
          f"threshold={best['threshold']}")
    if shipped:
        print(f"Shipped  {shipped['f1']:.3f} at coverage={retriever.min_coverage}, "
              f"threshold={settings.similarity_threshold}")
        spread = max(r["f1"] for r in rows) - min(r["f1"] for r in rows)
        if abs(shipped["f1"] - best["f1"]) < 1e-9:
            print(
                f"\nThe shipped configuration is already the best in the grid, and F1\n"
                f"varies by only {spread:.3f} across all {len(rows)} combinations. The two gates\n"
                f"trade against each other rather than compounding, so tuning this\n"
                f"backend further cannot help — closing the paraphrase gap needs a\n"
                f"different representation, not a different constant."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
