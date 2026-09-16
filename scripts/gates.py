"""Hard gates. Run before adversarial review and before scoring.

    python scripts/gates.py

Exit codes:
    0  all gates PASS       — candidate may proceed to review
    1  a gate FAILED        — reject
    2  a gate is UNVERIFIED — blocked, not rejected; see below

The rule this exists to enforce: **a gate that skips is not a gate that
passes.**

Running the suite today prints `126 passed, 17 skipped` — green. Ten of
those skips are the tokenizer parity tests, which skip when TensorFlow
isn't installed. So on a machine without the ML extras, the single most
important question in the project — does the shipped tokenizer agree with
the one the model was trained on — goes unanswered while the gate layer
reports success. A candidate could pass every gate with the parity
question never having been asked.

That is the same failure this project has been bitten by twice already:
the heuristic detector silently standing in for the ML classifier, and the
speech backend silently returning no audio. In both cases nothing failed;
something quietly did less than it appeared to. Gates that report
"unverified" as distinct from "passed" are how that stops happening.

Exit code 2 means: not rejected, but not cleared either. Install the
missing dependency and run again on a machine that can answer the
question.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"

# Well below the current 19 true positives: this catches collapse, not
# drift. Drift is the statistical layer's job, and it has error bars.
RETRIEVAL_FLOOR = 14


@dataclass
class GateResult:
    name: str
    status: str
    detail: str


def gate_tests() -> GateResult:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "no output"
    if proc.returncode != 0:
        return GateResult("Tests", FAIL, tail)
    return GateResult("Tests", PASS, tail)


def gate_lint() -> GateResult:
    proc = subprocess.run(["ruff", "check", "."], cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "ruff failed"
        return GateResult("Lint", FAIL, tail)
    return GateResult("Lint", PASS, "ruff check . clean")


def gate_tokenizer_parity() -> GateResult:
    """The gate that must not be allowed to skip quietly.

    Reported accuracy was measured with Keras's own tokenizer and the full
    Keras model. The app ships a reimplementation feeding a quantised
    TFLite model. If they disagree anywhere, the deployed accuracy is not
    the measured accuracy — and nothing crashes to say so.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "tests/test_tokenizer_parity.py", "-k", "matches_keras"],
        cwd=REPO, capture_output=True, text=True,
    )
    output = proc.stdout
    if proc.returncode != 0:
        tail = output.strip().splitlines()[-1] if output.strip() else "parity failed"
        return GateResult("Tokenizer parity", FAIL, tail)

    ran = output.count(".") and " passed" in output
    if not ran or "skipped" in output and " passed" not in output.split("skipped")[0]:
        return GateResult(
            "Tokenizer parity", UNVERIFIED,
            "skipped — needs full TensorFlow (Keras), which requires Python <= 3.13. "
            "The classifier itself only needs ai-edge-litert and runs without this; "
            "this gate is the extra check that the shipped tokenizer matches the "
            "one the model was trained with.",
        )
    return GateResult("Tokenizer parity", PASS, "matches Keras on every probe input")


def gate_runtime() -> GateResult:
    """Every orchestrator branch must complete without raising, offline.

    Import-time success is not enough — the branches that matter (security
    short-circuit, knowledge-base hit, offline fallback) are the ones a
    refactor silently breaks.
    """
    try:
        from bao.bootstrap import build_orchestrator

        _, orchestrator = build_orchestrator()

        checks = [
            ("security short-circuit", "", "blocked"),
            ("knowledge-base hit", "avuxeni", "knowledge_base"),
            ("offline fallback", "qqzz unanswerable nonsense xyz", "offline_fallback"),
        ]
        for label, query, expected_source in checks:
            result = orchestrator.handle(query, force_offline=True)
            if result.source != expected_source:
                return GateResult(
                    "Runtime integrity", FAIL,
                    f"{label}: expected source={expected_source!r}, got {result.source!r}",
                )
            if not result.text:
                return GateResult("Runtime integrity", FAIL, f"{label}: empty response text")

        # A second turn must not be affected by the first (the class of bug
        # that put per-turn state on the orchestrator).
        first = orchestrator.handle("help", force_offline=True, language_override="Sepedi")
        orchestrator.handle("avuxeni", force_offline=True)
        if not first.language_was_overridden:
            return GateResult("Runtime integrity", FAIL, "override state did not survive an intervening turn")

        return GateResult("Runtime integrity", PASS, f"{len(checks) + 1} branches exercised offline")
    except Exception as e:
        return GateResult("Runtime integrity", FAIL, f"{type(e).__name__}: {e}")


def gate_retrieval_integrity() -> GateResult:
    """Did retrieval collapse? Not: did retrieval get better.

    Split out from the runtime gate because they answer different
    questions and fail for different reasons. Runtime asks whether the
    code paths execute; this asks whether they still do their job.

    That distinction was earned, not assumed. The runtime gate originally
    carried this check and passed a configuration that rejected most real
    questions — because its branch probes use exact trigger words
    ("avuxeni") that survive almost any threshold. Executing a path proves
    very little about it.

    Deliberately a FLOOR, not a metric. Gates answer "did something
    collapse"; the statistical layer answers "is this better", and it has
    confidence intervals because at n=23 most apparent movement is noise.
    A gate with a metric in it is a hill to climb, which is exactly the
    failure the scorecard exists to prevent.
    """
    try:
        from bao.bootstrap import build_orchestrator
        from bao.evaluation import evaluate_retrieval_self_consistency
        from bao.rag_evaluation import evaluate_threshold, load_rag_eval_cases

        _, orchestrator = build_orchestrator()
        retriever = orchestrator.knowledge_retriever
        if retriever is None or not retriever.is_initialized:
            return GateResult("Retrieval integrity", FAIL, "retriever failed to initialize")

        # (a) The knowledge base can still find its own questions. Any value
        #     below 100% means the index or the corpus broke.
        consistency = evaluate_retrieval_self_consistency(retriever)
        if consistency.correct < consistency.total:
            missed = consistency.total - consistency.correct
            return GateResult(
                "Retrieval integrity", FAIL,
                f"self-consistency broke: {missed} of {consistency.total} rows "
                "no longer retrieve their own question",
            )

        # (b) Real labelled questions still get answered.
        cases = load_rag_eval_cases(str(REPO / "eval" / "rag_eval.csv"))
        outcome = evaluate_threshold(retriever, cases, retriever.threshold)
        if outcome.true_positives < RETRIEVAL_FLOOR:
            return GateResult(
                "Retrieval integrity", FAIL,
                f"collapsed to {outcome.true_positives} true positives "
                f"(floor {RETRIEVAL_FLOOR}, baseline 19)",
            )

        return GateResult(
            "Retrieval integrity", PASS,
            f"self-consistency {consistency.correct}/{consistency.total}, "
            f"{outcome.true_positives} TP >= floor {RETRIEVAL_FLOOR}",
        )
    except Exception as e:
        return GateResult("Retrieval integrity", FAIL, f"{type(e).__name__}: {e}")


def gate_api_contracts() -> GateResult:
    """The surface the UIs and scripts call by name.

    These are not enforced by the type checker at runtime, so a rename
    that misses one call site fails only when that path is exercised —
    which, for the console app or a script, may be during the demo.
    """
    try:
        import inspect

        from bao.ai.orchestrator import Orchestrator, PipelineResult
        from bao.bootstrap import build_orchestrator
        from bao.knowledge.retriever import KnowledgeRetriever
        from bao.services import speech

        required = {
            "PipelineResult fields": (
                {f for f in PipelineResult.__dataclass_fields__},
                {"text", "detected_language", "confidence", "detection_backend",
                 "source", "latency_ms", "audio", "audio_mime", "speech_language",
                 "speech_error", "stage_timings", "language_was_overridden"},
            ),
            "Orchestrator methods": (
                {m for m in dir(Orchestrator) if not m.startswith("__")},
                {"handle", "speak"},
            ),
            "KnowledgeRetriever methods": (
                {m for m in dir(KnowledgeRetriever) if not m.startswith("__")},
                {"lookup", "query_fact", "best_match_index", "query_coverage",
                 "is_initialized", "dataframe"},
            ),
            "speech module surface": (
                {m for m in dir(speech) if not m.startswith("_")},
                {"synthesize_speech", "transcribe_audio_bytes", "select_backend",
                 "has_tts_backend", "has_stt_backend", "tts_availability",
                 "reset_tts_cache", "SpeechAudio"},
            ),
        }
        for label, (actual, expected) in required.items():
            missing = expected - actual
            if missing:
                return GateResult("API contracts", FAIL, f"{label}: missing {sorted(missing)}")

        # handle() must keep the keyword arguments the UIs pass.
        params = set(inspect.signature(Orchestrator.handle).parameters)
        for keyword in ("want_speech", "on_chunk", "language_override", "force_offline"):
            if keyword not in params:
                return GateResult("API contracts", FAIL, f"Orchestrator.handle lost {keyword!r}")

        settings, orchestrator = build_orchestrator()
        if orchestrator.knowledge_retriever.min_coverage != settings.min_query_coverage:
            return GateResult("API contracts", FAIL, "config no longer reaches the retriever")

        return GateResult("API contracts", PASS, f"{len(required)} contracts intact")
    except Exception as e:
        return GateResult("API contracts", FAIL, f"{type(e).__name__}: {e}")


# Ordered cheapest-to-most-specific is tempting, but they run in the order
# a failure is most useful to read: a broken contract explains a runtime
# failure, and a runtime failure explains a retrieval failure. Reading top
# to bottom, the first FAIL is usually the cause rather than a symptom.
GATES = [
    gate_api_contracts,        # 1
    gate_runtime,              # 2
    gate_retrieval_integrity,  # 3
    gate_tokenizer_parity,     # 4
    gate_tests,                # 5
    gate_lint,                 # 5
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-unverified", action="store_true",
        help="Exit 0 even if a gate could not be checked. Use only when you have "
             "decided to accept that risk deliberately — it is off by default so "
             "the decision is never made by accident.",
    )
    args = parser.parse_args()

    print("\nHARD GATES")
    print("-" * 74)

    results = []
    for gate in GATES:
        result = gate()
        results.append(result)
        marker = {PASS: "  ok  ", FAIL: " FAIL ", UNVERIFIED: "  ??  "}[result.status]
        print(f"[{marker}] {result.name:<20} {result.detail}")

    print("-" * 74)

    failed = [r for r in results if r.status == FAIL]
    unverified = [r for r in results if r.status == UNVERIFIED]

    if failed:
        print(f"  REJECT — {len(failed)} gate(s) failed: {', '.join(r.name for r in failed)}\n")
        return 1
    if unverified and not args.allow_unverified:
        print(f"  BLOCKED — {len(unverified)} gate(s) could not be checked:")
        for r in unverified:
            print(f"      {r.name}: {r.detail}")
        print("\n  Not a rejection. Run again where the question can be answered,")
        print("  or pass --allow-unverified to accept the risk deliberately.\n")
        return 2

    if unverified:
        print(f"  PASS (with {len(unverified)} unverified, accepted via --allow-unverified)\n")
    else:
        print("  PASS — proceed to adversarial review.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
