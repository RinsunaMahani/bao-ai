"""Reproducible scorecard for accept/reject decisions on proposed changes.

    python scripts/scorecard.py --save baseline.json
    # ... apply a proposed change ...
    python scripts/scorecard.py --compare baseline.json

Why this exists, and why it reports error bars rather than a single number.

An accept-if-the-score-improved loop is only as good as the score's ability
to tell signal from noise. This project's labelled retrieval set has 70
queries, which means precision is measured over 23 items and recall over
29. At those sizes:

    precision 19/23 = 0.826   95% CI [0.629, 0.930]   width 0.302
    recall    19/29 = 0.655   95% CI [0.473, 0.801]   width 0.327

One query flipping moves precision 4.3 points. A change would have to flip
roughly 7-9 of the 70 before the movement is distinguishable from chance.
So a scorecard that printed "F1 0.731 -> 0.744, accept" would be reporting
noise as progress, and a loop run on that signal would spend its iterations
fitting the eval set rather than improving the system.

Hence `--compare` refuses to call a difference an improvement unless the
confidence intervals actually separate. It is designed to say "no
detectable change" often, because that is usually the truth.

Two things this cannot do, stated so they aren't assumed:

  1. It cannot detect overfitting to eval/rag_eval.csv. The retrieval
     coverage threshold was already tuned on that file, so it is a
     development set, not a held-out one. Repeated accept/reject cycles
     against it will keep improving the number and stop improving Bao.
     A genuine holdout has to be written by someone who is not proposing
     the changes, and it has to be looked at once.
  2. It cannot score what has no ground truth here — whether the greetings
     are linguistically correct, whether the tokenizer matches the one the
     model was trained with (see tests/test_tokenizer_parity.py), whether
     a voice sounds right. Those are the project's largest open risks and
     no amount of this loop will touch them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bao.bootstrap import build_orchestrator  # noqa: E402
from bao.evaluation import evaluate_retrieval_self_consistency  # noqa: E402
from bao.rag_evaluation import evaluate_threshold, load_rag_eval_cases  # noqa: E402

LATENCY_QUERIES = ["avuxeni", "what is the capital of south africa", "sawubona", "molo"]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct for small samples, unlike the normal
    approximation, which produces intervals extending past 0 or 1 at these
    counts.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def measure_retrieval(orchestrator) -> dict:
    retriever = orchestrator.knowledge_retriever
    cases = load_rag_eval_cases(str(REPO / "eval" / "rag_eval.csv"))
    result = evaluate_threshold(retriever, cases, retriever.threshold)

    tp, fp, fn = result.true_positives, result.false_positives, result.false_negatives
    precision_n, recall_n = tp + fp, tp + fn
    precision = tp / precision_n if precision_n else 0.0
    recall = tp / recall_n if recall_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "cases": len(cases),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": result.true_negatives,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "precision_ci": [round(v, 4) for v in wilson(tp, precision_n)],
        "precision_n": precision_n,
        "recall": round(recall, 4),
        "recall_ci": [round(v, 4) for v in wilson(tp, recall_n)],
        "recall_n": recall_n,
        "f1": round(f1, 4),
    }


def measure_self_consistency(orchestrator) -> dict:
    report = evaluate_retrieval_self_consistency(orchestrator.knowledge_retriever)
    return {
        "rows": report.total,
        "correct": report.correct,
        # A regression check, not an accuracy claim — it asks the knowledge
        # base its own questions verbatim. Any value below 1.0 means
        # something broke.
        "ratio": round(report.correct / report.total, 4) if report.total else 0.0,
    }


def measure_latency(orchestrator, repeats: int = 40) -> dict:
    for query in LATENCY_QUERIES:          # warm caches first
        orchestrator.handle(query, force_offline=True)

    samples: list[float] = []
    for _ in range(repeats):
        for query in LATENCY_QUERIES:
            start = time.perf_counter()
            orchestrator.handle(query, force_offline=True)
            samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    return {
        "samples": len(samples),
        "mean_ms": round(sum(samples) / len(samples), 3),
        "p50_ms": round(samples[len(samples) // 2], 3),
        "p95_ms": round(samples[int(len(samples) * 0.95)], 3),
    }


def measure_tests() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = failed = skipped = 0
    for token, label in (("passed", "p"), ("failed", "f"), ("skipped", "s")):
        for part in tail.replace(",", "").split():
            if part == token:
                index = tail.replace(",", "").split().index(part)
                value = int(tail.replace(",", "").split()[index - 1])
                if label == "p":
                    passed = value
                elif label == "f":
                    failed = value
                else:
                    skipped = value
    return {"passed": passed, "failed": failed, "skipped": skipped, "exit_code": proc.returncode}


def measure_lint() -> dict:
    proc = subprocess.run(["ruff", "check", "."], cwd=REPO, capture_output=True, text=True)
    return {"clean": proc.returncode == 0, "output_tail": proc.stdout.strip().splitlines()[-1:] or [""]}


def build_scorecard() -> dict:
    _, orchestrator = build_orchestrator()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrieval": measure_retrieval(orchestrator),
        "self_consistency": measure_self_consistency(orchestrator),
        "latency_offline": measure_latency(orchestrator),
        "tests": measure_tests(),
        "lint": measure_lint(),
    }


def render(card: dict) -> str:
    r, s, latency, t = card["retrieval"], card["self_consistency"], card["latency_offline"], card["tests"]
    lines = [
        "",
        f"SCORECARD  {card['generated_at']}",
        "-" * 66,
        f"  Retrieval        TP {r['true_positives']}  FP {r['false_positives']}  "
        f"TN {r['true_negatives']}  FN {r['false_negatives']}   ({r['cases']} cases)",
        f"    precision      {r['precision']:.3f}   95% CI [{r['precision_ci'][0]:.3f}, "
        f"{r['precision_ci'][1]:.3f}]  (n={r['precision_n']})",
        f"    recall         {r['recall']:.3f}   95% CI [{r['recall_ci'][0]:.3f}, "
        f"{r['recall_ci'][1]:.3f}]  (n={r['recall_n']})",
        f"    F1             {r['f1']:.3f}",
        f"  Self-consistency {s['correct']}/{s['rows']}  (regression check — expect 1.00)",
        f"  Offline latency  mean {latency['mean_ms']:.2f} ms   p95 {latency['p95_ms']:.2f} ms",
        f"  Tests            {t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped",
        f"  Lint             {'clean' if card['lint']['clean'] else 'FAILING'}",
        "",
    ]
    return "\n".join(lines)


def compare(before: dict, after: dict) -> str:
    lines = ["", "COMPARISON", "-" * 66]
    verdicts: list[str] = []

    b, a = before["retrieval"], after["retrieval"]
    for metric in ("precision", "recall"):
        b_lo, b_hi = b[f"{metric}_ci"]
        a_lo, a_hi = a[f"{metric}_ci"]
        delta = a[metric] - b[metric]
        separated = a_lo > b_hi or a_hi < b_lo
        if separated:
            verdict = "IMPROVED" if delta > 0 else "REGRESSED"
        else:
            verdict = "no detectable change"
        verdicts.append(verdict)
        lines.append(
            f"  {metric:<10} {b[metric]:.3f} -> {a[metric]:.3f}  ({delta:+.3f})   {verdict}"
        )
        if not separated and abs(delta) > 0.001:
            lines.append(
                f"             (intervals overlap: [{b_lo:.3f},{b_hi:.3f}] vs "
                f"[{a_lo:.3f},{a_hi:.3f}] — this difference is within noise)"
            )

    # Hard gates: these are not judgement calls.
    if after["tests"]["failed"] or after["tests"]["exit_code"] != 0:
        verdicts.append("BLOCKED")
        lines.append(f"  tests      FAILING ({after['tests']['failed']}) — reject regardless of score")
    if not after["lint"]["clean"]:
        verdicts.append("BLOCKED")
        lines.append("  lint       FAILING — reject regardless of score")
    if after["self_consistency"]["ratio"] < before["self_consistency"]["ratio"]:
        verdicts.append("BLOCKED")
        lines.append("  self-consistency dropped — retrieval broke; reject")

    b_ms, a_ms = before["latency_offline"]["mean_ms"], after["latency_offline"]["mean_ms"]
    change = (a_ms - b_ms) / b_ms * 100 if b_ms else 0.0
    lines.append(f"  latency    {b_ms:.2f} -> {a_ms:.2f} ms  ({change:+.1f}%)")

    lines.append("-" * 66)
    if "BLOCKED" in verdicts:
        lines.append("  VERDICT: REJECT — a hard gate failed.")
    elif "REGRESSED" in verdicts:
        lines.append("  VERDICT: REJECT — a metric moved down beyond noise.")
    elif "IMPROVED" in verdicts:
        lines.append("  VERDICT: ACCEPT — a metric moved up beyond noise.")
    else:
        lines.append("  VERDICT: NO EVIDENCE either way.")
        lines.append("  Decide on grounds this scorecard does not measure — readability,")
        lines.append("  a bug fixed, a risk removed. Do not accept it as a score win.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Write the scorecard to a JSON file.")
    parser.add_argument("--compare", type=Path, help="Compare against a saved scorecard.")
    args = parser.parse_args()

    card = build_scorecard()
    print(render(card))

    if args.compare:
        if not args.compare.exists():
            print(f"No such scorecard: {args.compare}", file=sys.stderr)
            return 2
        print(compare(json.loads(args.compare.read_text()), card))

    if args.save:
        args.save.write_text(json.dumps(card, indent=2))
        print(f"Saved to {args.save}\n")

    return 0 if card["tests"]["exit_code"] == 0 and card["lint"]["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
