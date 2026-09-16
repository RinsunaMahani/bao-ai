"""
Bao AI - Benchmarking Suite
Measures offline vs. online pipeline latency, language-detection accuracy
and detection score, and memory footprint.

This is a rewrite, not just a rename: the original benchmark called
`self.llm_service.generate_response(query=..., force_offline=...)` and read
`self.llm_service.is_online` — neither existed on the actual LLM service
class (it only had `.generate(prompt, context, lang)`), so the benchmark
has never successfully run to completion. It now drives the real
`Orchestrator.handle()` — the same call path the UI uses — so a benchmark
result here is guaranteed to reflect real app behavior, not a mocked-up
approximation of it.
"""

import json
import os
import time
import tracemalloc
from statistics import mean, median

import psutil

from bao.ai.client import GeminiClient
from bao.ai.orchestrator import Orchestrator
from bao.core.config import Settings
from bao.core.security import SecurityGuardrails
from bao.knowledge.retriever import KnowledgeRetriever, has_retrieval_support
from bao.services.language_detector import get_language_detector

TEST_QUERIES = [
    {"query": "How do I clear the local index cache?", "expected_lang": "English"},
    {"query": "Sawubona, ngicela usizo ngale software", "expected_lang": "isiZulu"},
    {"query": "Dumela, nka thusha bjang ka system e?", "expected_lang": "Sepedi"},
    {"query": "Dankie vir die hulp met die lêers", "expected_lang": "Afrikaans"},
    {"query": "Explain how offline retrieval routing works", "expected_lang": "English"},
]


class BaoBenchmarker:
    def __init__(self, iterations: int = 5):
        self.iterations = iterations
        self.process = psutil.Process(os.getpid())

        print("Initializing Bao AI pipeline...")
        self.settings = Settings()
        self.security = SecurityGuardrails(max_length=self.settings.max_query_length)
        # Matches the real app's default (bao/ui/streamlit_app.py,
        # bao/ui/console_app.py): the validated TFLite detector, heuristic
        # as automatic fallback if the model/tokenizer files are missing.
        # See REVIEW.md Sec 5 for why this is now the default.
        self.language_detector = get_language_detector(
            prefer_ml=True,
            model_path=self.settings.classifier_model_path,
            tokenizer_config_path=self.settings.tokenizer_config_path,
        )
        self.knowledge_retriever = (
            KnowledgeRetriever(
                data_path=self.settings.knowledge_base_path,
                threshold=self.settings.similarity_threshold,
            )
            if has_retrieval_support() else None
        )
        self.gemini_client = GeminiClient(self.settings)
        self.orchestrator = Orchestrator(
            self.security, self.language_detector, self.knowledge_retriever, self.gemini_client,
        )

    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def compare_detectors(self) -> dict:
        """Head-to-head accuracy comparison, heuristic vs. TFLite, on the
        same query set — the concrete evidence behind switching the
        app's default (see REVIEW.md Sec 5). Returns the comparison dict
        so run_benchmark can fold it into the exported JSON.
        """
        from bao.services.language_detector import HeuristicLanguageDetector, get_language_detector

        heuristic = HeuristicLanguageDetector()
        tflite = get_language_detector(
            prefer_ml=True,
            model_path=self.settings.classifier_model_path,
            tokenizer_config_path=self.settings.tokenizer_config_path,
        )

        # Label the column by the backend that ACTUALLY ran. Without this,
        # an environment without a working tensorflow silently falls back to
        # the heuristic and the benchmark prints two identical columns, one
        # of them falsely labelled "tflite" — which is exactly what happened
        # to an external reviewer, who reported TFLite failing the Sepedi
        # "Dumela" case that it actually classifies correctly at 99.3%
        # confidence. Same masking bug already fixed in the tests; the
        # benchmark had no equivalent guard until now.
        actual_backend = tflite.detect("test").backend
        ml_label = actual_backend
        if actual_backend != "tflite":
            print(
                f"\n  WARNING: the ML detector fell back to '{actual_backend}' — "
                "tensorflow/model/tokenizer unavailable.\n"
                "  The second column below is NOT the TFLite model. Install "
                "requirements-ml.txt for a real comparison.",
            )

        print("\nHEURISTIC vs. ML DETECTOR — head-to-head on the same queries")
        print("-" * 65)
        comparison = {"heuristic": {"correct": 0, "total": 0}, ml_label: {"correct": 0, "total": 0}}
        for item in TEST_QUERIES:
            h = heuristic.detect(item["query"])
            t = tflite.detect(item["query"])
            comparison["heuristic"]["total"] += 1
            comparison[ml_label]["total"] += 1
            if h.language == item["expected_lang"]:
                comparison["heuristic"]["correct"] += 1
            if t.language == item["expected_lang"]:
                comparison[ml_label]["correct"] += 1
            h_mark = "OK" if h.language == item["expected_lang"] else "  "
            t_mark = "OK" if t.language == item["expected_lang"] else "  "
            print(
                f"  expected={item['expected_lang']:10} | "
                f"heuristic[{h_mark}]={h.language:10} | "
                f"{ml_label}[{t_mark}]={t.language:10} | {item['query'][:35]!r}"
            )

        for name in ("heuristic", ml_label):
            c, n = comparison[name]["correct"], comparison[name]["total"]
            print(f"  {name:10}: {c}/{n} ({c / n * 100:.0f}%)")
        return comparison

    def run_benchmark(self):
        detector_comparison = self.compare_detectors()

        print(f"\nStarting Bao AI Benchmark ({self.iterations} iterations per query)...")
        print("=" * 65)

        tracemalloc.start()
        mem_baseline = self.get_memory_mb()

        results = {
            "pipeline": {"latencies": [], "detection_scores": []},
            "language_detection": {"latencies": [], "detection_scores": [], "correct": 0, "total": 0},
        }

        print("\n[1/2] Benchmarking language detection...")
        for item in TEST_QUERIES:
            for _ in range(self.iterations):
                start = time.perf_counter()
                result = self.language_detector.detect(item["query"])
                lat = (time.perf_counter() - start) * 1000
                results["language_detection"]["latencies"].append(lat)
                results["language_detection"]["detection_scores"].append(result.confidence)
                results["language_detection"]["total"] += 1
                if result.language == item["expected_lang"]:
                    results["language_detection"]["correct"] += 1

        online = self.gemini_client.is_available()
        print(f"[2/2] Benchmarking full pipeline ({'online' if online else 'offline'} mode)...")
        for item in TEST_QUERIES:
            for _ in range(self.iterations):
                start = time.perf_counter()
                result = self.orchestrator.handle(item["query"])
                lat = (time.perf_counter() - start) * 1000
                results["pipeline"]["latencies"].append(lat)
                results["pipeline"]["detection_scores"].append(result.confidence)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_final = self.get_memory_mb()

        self._print_report(results, mem_baseline, mem_final, peak_mem / (1024 * 1024), online, detector_comparison)

    @staticmethod
    def _p95(data: list) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(0.95 * len(sorted_data))
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def _print_report(self, results, mem_base, mem_final, peak_traced_mb,
                      online: bool, detector_comparison: dict | None = None):
        print("\n" + "=" * 65)
        print("                    BAO AI BENCHMARK REPORT                   ")
        print("=" * 65)

        print("\nLATENCY PERFORMANCE (ms)")
        print("-" * 65)
        print(f"{'Stage':<22} | {'Mean':<8} | {'Median (P50)':<12} | {'P95':<8}")
        print("-" * 65)
        for stage, data in results.items():
            lats = data["latencies"]
            if lats:
                print(f"{stage.title():<22} | {mean(lats):<8.2f} | {median(lats):<12.2f} | {self._p95(lats):<8.2f}")
            else:
                print(f"{stage.title():<22} | N/A      | N/A          | N/A")

        print("\nLANGUAGE DETECTION")
        print("-" * 65)
        lang_scores = results["language_detection"]["detection_scores"]
        correct, total = results["language_detection"]["correct"], results["language_detection"]["total"]
        if total:
            unique = len(TEST_QUERIES)
            print(f"Unique queries classified correctly : {correct // self.iterations}/{unique}")
            print(f"  ({total} total inference runs = {unique} queries x {self.iterations} iterations,")
            print("   repeated for latency measurement — NOT independent language examples)")
        if lang_scores:
            backend = self.language_detector.detect(TEST_QUERIES[0]["query"]).backend
            score_label = "heuristic keyword-match score" if backend == "heuristic" else f"{backend} model score"
            print(f"Average {score_label} : {mean(lang_scores) * 100:.1f}%")
            print(f"Minimum {score_label} : {min(lang_scores) * 100:.1f}%")
            if backend == "heuristic":
                print("(Note: a heuristic keyword-match score is not a calibrated probability.)")

        print("\nMEMORY FOOTPRINT (RAM)")
        print("-" * 65)
        print(f"Initial Baseline RAM Usage : {mem_base:.2f} MB")
        print(f"Final RAM Usage            : {mem_final:.2f} MB")
        print(f"Net Memory Change          : {mem_final - mem_base:+.2f} MB")
        print(f"Peak Python Allocated RAM  : {peak_traced_mb:.2f} MB")
        print(f"\nGemini reachable this run  : {online}")
        print("=" * 65 + "\n")

        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gemini_online": online,
            "language_detection_accuracy": {
                # Deliberately NOT reported as "25/25": that would imply 25
                # independent labelled examples. It's 5 unique queries run
                # 5 times each for latency measurement. The unique count is
                # the one that means anything about accuracy.
                "unique_cases": len(TEST_QUERIES),
                "unique_correct": results["language_detection"]["correct"] // self.iterations,
                "total_inference_runs": results["language_detection"]["total"],
                "iterations_per_query": self.iterations,
                "note": "unique_correct/unique_cases is the accuracy figure; total_inference_runs is for latency only",
            },
            "detector_comparison": detector_comparison,
            "memory": {"baseline_mb": mem_base, "final_mb": mem_final, "peak_traced_mb": peak_traced_mb},
            "latencies_ms": {
                stage: {"mean": mean(d["latencies"]) if d["latencies"] else 0, "p95": self._p95(d["latencies"])}
                for stage, d in results.items()
            },
        }
        with open("benchmark_results.json", "w") as f:
            json.dump(report_data, f, indent=2)
        print("Detailed results exported to 'benchmark_results.json'")


if __name__ == "__main__":
    BaoBenchmarker(iterations=5).run_benchmark()
