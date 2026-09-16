"""Reports which of the eleven voices actually work on THIS machine.

Run this on the demo laptop before presenting:

    python scripts/probe_tts.py

Why it exists: "only Xitsonga speaks" has at least four different causes,
and they need different fixes —

  1. the speech extras aren't installed at all;
  2. the voice is mid-download (each MMS language is a separate ~145 MB
     fetch on first use, so the first request in a new language routinely
     looks broken while the second is instant);
  3. no `facebook/mms-tts-<code>` repository exists for that language;
  4. no network, which only affects the three online edge-tts voices.

Guessing between these from a description wastes time. This measures.

By default it only LOADS each model, which is the step that fails. Pass
--synthesize to also generate a short clip, which is slower but proves the
voice produces audio rather than merely loading.
"""
from __future__ import annotations

import argparse
import sys
import time

from bao.core.config import LABELS, Settings
from bao.services.speech import (
    _EDGE_VOICES,
    has_edge_backend,
    has_mms_backend,
    load_tts_model,
    select_backend,
    synthesize_speech,
)

# One short, safe phrase — the point is whether the voice loads and speaks,
# not what it says.
SAMPLE_TEXT = "Bao."


def probe(language: str, settings: Settings, synthesize: bool) -> tuple[str, str, str]:
    """Returns (backend, status, detail) for one language."""
    backend = select_backend(language, settings.tts_backend)
    if backend is None:
        return "-", "NO BACKEND", "neither edge-tts nor MMS can serve this language"

    if backend == "edge":
        voice = _EDGE_VOICES[language]
        if not synthesize:
            return "edge", "READY", f"{voice} (online; use --synthesize to confirm)"
        errors: list[str] = []
        started = time.perf_counter()
        audio = synthesize_speech(
            SAMPLE_TEXT, language, backend="edge", on_error=errors.append
        )
        elapsed = (time.perf_counter() - started) * 1000
        if audio:
            return "edge", "OK", f"{voice}, {len(audio.data) / 1024:.0f} KB in {elapsed:.0f} ms"
        return "edge", "FAILED", errors[-1] if errors else "no audio returned"

    code = settings.mms_codes.get(language, "eng")
    started = time.perf_counter()
    try:
        load_tts_model(code)
    except Exception as e:
        detail = str(e)
        lowered = detail.lower()
        # 401 first, and distinct from 404. transformers renders both with the
        # same "not a valid model identifier" sentence, which is how a refused
        # request got misreported as a missing model.
        if "401" in detail or "unauthorized" in lowered or "gated" in lowered:
            return "mms", "AUTH NEEDED", f"facebook/mms-tts-{code} refused (401) — set HF_TOKEN"
        if "429" in detail or "rate limit" in lowered:
            return "mms", "RATE LIMITED", f"facebook/mms-tts-{code} — set HF_TOKEN or wait"
        if "404" in detail or "repositorynotfound" in lowered:
            return "mms", "NO SUCH MODEL", f"facebook/mms-tts-{code} returned 404"
        if any(k in lowered for k in ("connection", "timeout", "resolve", "network", "ssl")):
            return "mms", "NO NETWORK", f"could not download facebook/mms-tts-{code}"
        return "mms", "FAILED", f"mms-tts-{code}: {detail[:80]}"
    load_ms = (time.perf_counter() - started) * 1000

    if not synthesize:
        return "mms", "LOADED", f"mms-tts-{code} in {load_ms:.0f} ms"

    errors = []
    started = time.perf_counter()
    audio = synthesize_speech(SAMPLE_TEXT, language, backend="mms", on_error=errors.append)
    speak_ms = (time.perf_counter() - started) * 1000
    if audio:
        return "mms", "OK", f"mms-tts-{code}, spoke in {speak_ms:.0f} ms"
    return "mms", "FAILED", errors[-1] if errors else "loaded but produced no audio"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthesize", action="store_true",
        help="Also generate a short clip per language (slower, but proves it speaks).",
    )
    args = parser.parse_args()

    settings = Settings()
    from bao.services.language_detector import tflite_runtime_name

    runtime = tflite_runtime_name()
    print(f"\nML classifier      : {runtime or 'NOT AVAILABLE — pip install ai-edge-litert'}")
    print(f"edge-tts installed : {has_edge_backend()}")
    print(f"MMS installed      : {has_mms_backend()}")
    print(f"backend preference : {settings.tts_backend}\n")

    if not has_edge_backend() and not has_mms_backend():
        print("Neither backend is installed. Run:")
        print("    pip install -r requirements-speech.txt\n")
        return 1

    print(f"{'LANGUAGE':<12} {'BACKEND':<8} {'STATUS':<14} DETAIL")
    print("-" * 92)

    problems = []
    for language in LABELS:
        backend, status, detail = probe(language, settings, args.synthesize)
        print(f"{language:<12} {backend:<8} {status:<14} {detail}")
        if status not in {"OK", "LOADED", "READY"}:
            problems.append((language, status))

    print()
    if not problems:
        print("All eleven voices are available on this machine.\n")
        return 0

    print(f"{len(problems)} of {len(LABELS)} languages have no working voice:\n")
    for language, status in problems:
        print(f"  - {language}: {status}")

    statuses = {s for _, s in problems}
    print("\nWhat to do:")
    if "AUTH NEEDED" in statuses or "RATE LIMITED" in statuses:
        print("  AUTH NEEDED   -> Hugging Face refused the request. Get a free token at")
        print("                   huggingface.co/settings/tokens, then:")
        print("                     Windows : setx HF_TOKEN \"hf_xxx\"   (reopen the terminal)")
        print("                     macOS   : export HF_TOKEN=hf_xxx")
        print("                   and re-run. This is NOT the same as the model being absent.")
    if "NO NETWORK" in statuses:
        print("  NO NETWORK    -> connect and re-run; each MMS voice is a ~145 MB first-use download.")
    if "NO SUCH MODEL" in statuses:
        print("  NO SUCH MODEL -> a genuine 404. MMS has no voice for that language;")
        print("                   say so in the presentation rather than demoing it.")
    if "NO BACKEND" in statuses:
        print("  NO BACKEND    -> pip install -r requirements-speech.txt")
    if "FAILED" in statuses:
        print("  FAILED        -> read the detail column.")
    if not has_edge_backend():
        print()
        print("  edge-tts is NOT installed. It needs no Hugging Face account and gives")
        print("  real South African voices for English, Afrikaans and isiZulu:")
        print("      pip install edge-tts")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
