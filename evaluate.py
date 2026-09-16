"""Command-line evaluation runner for Bao AI.

Usage:
    python evaluate.py
        Runs retrieval self-consistency against data/african_data.csv.

    python evaluate.py --detector-eval eval/language_eval_template.csv
        Also runs language detector accuracy against a labelled CSV
        (columns: text,label).

    python evaluate.py --out eval/report.md
        Also writes the combined report to disk as markdown.

See eval/README.md for how to build a trustworthy detector evaluation set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bao.core.config import Settings
from bao.evaluation import (
    evaluate_language_detector,
    evaluate_retrieval_self_consistency,
    load_labelled_examples,
)
from bao.knowledge.retriever import KnowledgeRetriever, has_retrieval_support
from bao.services.language_detector import get_language_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--detector-eval", type=Path, default=None,
        help="Path to a labelled CSV (columns: text,label) for language detector accuracy evaluation.",
    )
    parser.add_argument(
        "--use-ml-detector", action="store_true",
        help="Evaluate the TFLite language detector instead of the default heuristic one.",
    )
    parser.add_argument(
        "--rag-eval", type=Path, default=Path("eval/rag_eval.csv"),
        help="Labelled retrieval CSV (columns: query,expected_row,category) for "
             "precision/recall/F1 and a threshold sweep. Defaults to "
             "eval/rag_eval.csv — this is the meaningful retrieval metric, so "
             "it runs by default rather than behind a flag. Pass 'none' to skip.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional path to also write the report as markdown.")
    args = parser.parse_args()

    if not has_retrieval_support():
        print("scikit-learn is not installed; cannot run the retrieval evaluation.", file=sys.stderr)
        return 1

    settings = Settings()
    sections: list[str] = ["# Bao AI — Evaluation Report", ""]

    retriever = KnowledgeRetriever(
        data_path=settings.knowledge_base_path,
        threshold=settings.similarity_threshold,
        min_coverage=settings.min_query_coverage,
    )
    retrieval_report = evaluate_retrieval_self_consistency(retriever)
    print(retrieval_report.to_markdown())
    print()
    sections += [retrieval_report.to_markdown(), ""]

    if args.rag_eval and str(args.rag_eval).lower() != "none":
        if not args.rag_eval.exists():
            print(f"RAG eval file not found: {args.rag_eval}", file=sys.stderr)
        else:
            from bao.rag_evaluation import (
                load_rag_eval_cases,
                per_category_breakdown,
                sweep_thresholds,
                to_markdown,
            )

            cases = load_rag_eval_cases(args.rag_eval)
            if not cases:
                print(f"No labelled rows found in {args.rag_eval}.", file=sys.stderr)
            else:
                sweep = sweep_thresholds(retriever, cases)
                breakdown = per_category_breakdown(retriever, cases, settings.similarity_threshold)
                rag_report = to_markdown(sweep, breakdown, settings.similarity_threshold)
                print(rag_report)
                print()
                sections += [rag_report, ""]

    if args.detector_eval:
        if not args.detector_eval.exists():
            print(f"Detector eval file not found: {args.detector_eval}", file=sys.stderr)
        else:
            examples = load_labelled_examples(args.detector_eval)
            if not examples:
                print(f"No labelled rows found in {args.detector_eval}.", file=sys.stderr)
            else:
                detector = get_language_detector(
                    prefer_ml=args.use_ml_detector, model_path=settings.classifier_model_path,
                    tokenizer_config_path=settings.tokenizer_config_path,
                )
                if args.use_ml_detector and detector.detect("test").backend != "tflite":
                    print(
                        "WARNING: --use-ml-detector was requested but the detector silently "
                        "fell back to the heuristic (tensorflow/model/tokenizer unavailable). "
                        "The report below evaluates the heuristic, not the ML model.",
                        file=sys.stderr,
                    )
                detector_report = evaluate_language_detector(detector, examples)
                print(detector_report.to_markdown())
                sections += [detector_report.to_markdown(), ""]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(sections), encoding="utf-8")
        print(f"\nWrote report to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
