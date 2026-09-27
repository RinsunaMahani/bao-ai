"""Bao AI - Retrieval backend comparison: TF-IDF vs. semantic embeddings.

Runs BOTH retrieval backends against the same labelled evaluation set
(`eval/rag_eval.csv`) and prints their numbers side by side, so the choice
between them is made on measured evidence rather than on which sounds more
modern.

    python compare_retrievers.py

Requires `pip install -r requirements-semantic.txt` (downloads ~470 MB of
model weights on first run).

THE RULE, set before the first run: adopt semantic embeddings only if
they improve paraphrase recall WITHOUT giving back negative rejection. If
they don't, keep TF-IDF and document that - a measured negative result is
a real finding, not a failed experiment.

MEASURED 2026-09-27, paraphrase-multilingual-MiniLM-L12-v2 on
sentence-transformers 5.6.1 (70 queries; exact / paraphrase / negative):

    TF-IDF, shipped (threshold 0.25, coverage 0.7)   15/15   7/15  37/40  F1 0.800
    Semantic at 0.25                                 15/15  14/15  17/40  F1 0.707
    Semantic at its best threshold, 0.55             15/15  12/15  34/40  F1 0.857

Semantic does not pass the rule at any threshold: its best F1 buys five
paraphrases by answering three more unrelated questions as verified. The
README's "Retrieval precision and recall" section has the full reasoning
and why TF-IDF stayed the default.

INSTALL WARNING: `sentence-transformers` pulls in torch, which can collide
with an existing TensorFlow install at the native-library level (this was
observed as a hard segfault with both present). If the TFLite language
detector stops working after installing it, that's the likely cause —
consider running this experiment in a separate virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bao.core.config import Settings
from bao.knowledge.embeddings import TfidfEmbeddings, has_semantic_support
from bao.knowledge.retriever import KnowledgeRetriever
from bao.rag_evaluation import (
    load_rag_eval_cases,
    per_category_breakdown,
    sweep_thresholds,
)

EVAL_CSV = Path(__file__).parent / "eval" / "rag_eval.csv"


def _summarize(name: str, retriever: KnowledgeRetriever, cases, threshold: float) -> dict:
    breakdown = per_category_breakdown(retriever, cases, threshold)
    sweep = sweep_thresholds(retriever, cases)
    best = max(sweep, key=lambda r: r.f1)
    at_threshold = next(r for r in sweep if abs(r.threshold - threshold) < 1e-9) if any(
        abs(r.threshold - threshold) < 1e-9 for r in sweep
    ) else best
    return {
        "name": name,
        "breakdown": breakdown,
        # Each backend's similarity scores live on their own scale, so a
        # threshold tuned for one says nothing about the other. Showing
        # both at 0.25 alone made the semantic backend look worse than it
        # is at its own best setting.
        "best_breakdown": per_category_breakdown(retriever, cases, best.threshold),
        "precision": at_threshold.precision,
        "recall": at_threshold.recall,
        "f1": at_threshold.f1,
        "best_f1": best.f1,
        "best_threshold": best.threshold,
    }


def _print_side_by_side(a: dict, b: dict | None, threshold: float) -> None:
    print("\n" + "=" * 72)
    print(f"RETRIEVAL BACKEND COMPARISON (threshold {threshold:.2f}, {EVAL_CSV.name})")
    print("=" * 72)

    header = f"{'Metric':<28} | {a['name']:<18}"
    if b:
        header += f" | {b['name']:<18}"
    print(header)
    print("-" * 72)

    categories = sorted(set(a["breakdown"]) | set(b["breakdown"] if b else {}))
    for cat in categories:
        ac, at_ = a["breakdown"].get(cat, (0, 0))
        line = f"{cat:<28} | {f'{ac}/{at_}':<18}"
        if b:
            bc, bt = b["breakdown"].get(cat, (0, 0))
            line += f" | {f'{bc}/{bt}':<18}"
        print(line)

    for label, key in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")):
        line = f"{label:<28} | {a[key]:<18.3f}"
        if b:
            line += f" | {b[key]:<18.3f}"
        print(line)

    a_best = f"{a['best_f1']:.3f} @ {a['best_threshold']:.2f}"
    line = f"{'Best F1 (swept)':<28} | {a_best:<18}"
    if b:
        b_best = f"{b['best_f1']:.3f} @ {b['best_threshold']:.2f}"
        line += f" | {b_best:<18}"
    print(line)

    print("-" * 72)
    print("At each backend's own best threshold:")
    for summary in (a, b) if b else (a,):
        cells = ", ".join(
            f"{cat} {c}/{t}" for cat, (c, t) in sorted(summary["best_breakdown"].items())
        )
        print(f"  {summary['name']:<9} @ {summary['best_threshold']:.2f}: {cells}")
    print("=" * 72)


def main() -> int:
    settings = Settings()
    if not EVAL_CSV.exists():
        print(f"Evaluation set not found: {EVAL_CSV}", file=sys.stderr)
        return 1

    cases = load_rag_eval_cases(EVAL_CSV)
    threshold = settings.similarity_threshold
    print(f"Loaded {len(cases)} labelled queries from {EVAL_CSV.name}.")

    print("\nBuilding TF-IDF baseline...")
    tfidf = KnowledgeRetriever(
        data_path=settings.knowledge_base_path, threshold=threshold,
        embedding_model=TfidfEmbeddings(),
    )
    tfidf_summary = _summarize("TF-IDF", tfidf, cases, threshold)

    semantic_summary = None
    if not has_semantic_support():
        print(
            "\nsentence-transformers is not installed — showing the TF-IDF baseline only.\n"
            "Install it to run the comparison:  pip install -r requirements-semantic.txt",
            file=sys.stderr,
        )
    else:
        from bao.knowledge.embeddings import SentenceEmbeddings

        print("Building semantic backend (downloads model weights on first run)...")
        try:
            semantic = KnowledgeRetriever(
                data_path=settings.knowledge_base_path, threshold=threshold,
                embedding_model=SentenceEmbeddings(),
            )
            if not semantic.is_initialized:
                raise RuntimeError("semantic retriever failed to initialize")
            semantic_summary = _summarize("Semantic", semantic, cases, threshold)
        except Exception as e:
            print(f"\nSemantic backend unavailable: {e}", file=sys.stderr)

    _print_side_by_side(tfidf_summary, semantic_summary, threshold)

    if semantic_summary:
        print(
            "\nAdopt the semantic backend only if it genuinely improves paraphrase\n"
            "recall AND negative rejection. If it doesn't, keep TF-IDF and record\n"
            "that in the README - a measured negative result is a real finding."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
