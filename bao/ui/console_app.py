"""
Bao AI — Console Entrypoint.

Run with: python bao_console.py
"""
from __future__ import annotations

import io

try:
    import speech_recognition as sr
    _HAS_SPEECH_RECOGNITION = True
except ImportError:
    # Microphone input is an optional extra (requirements-speech.txt).
    # Importing it unguarded made `pytest` and `evaluate.py` fail on a
    # core-only install, because both import this module transitively.
    _HAS_SPEECH_RECOGNITION = False

try:
    import sounddevice as sd
    import soundfile as sf
    _HAS_AUDIO_PLAYBACK = True
except (ImportError, OSError):
    # OSError, not just ImportError: `sounddevice` imports fine but raises
    # OSError("PortAudio library not found") at import time on any machine
    # without the native PortAudio library — which is every headless server,
    # slim Docker image, and CI runner. Catching only ImportError made the
    # console app crash on startup there instead of degrading to text mode.
    _HAS_AUDIO_PLAYBACK = False

from bao.bootstrap import build_orchestrator
from bao.core.config import Settings
from bao.services.speech import transcribe_audio_bytes


def _play_audio_bytes(audio_bytes: bytes | None, mime: str = "audio/wav") -> None:
    if not audio_bytes or not _HAS_AUDIO_PLAYBACK:
        return
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        if mime == "audio/mpeg":
            # edge-tts returns MP3. soundfile only decodes it with
            # libsndfile >= 1.1; older builds raise here. The web UI plays
            # MP3 natively, so this only affects console playback.
            print(
                "[Console playback needs a newer libsndfile for MP3 "
                "(pip install -U soundfile), or set backend = \"mms\" in config.toml]"
            )
            return
        print(f"[Audio playback error: {e}]")


def _listen(settings: Settings, target_language: str = "English") -> str | None:
    if not _HAS_SPEECH_RECOGNITION:
        print(
            "\n[Microphone input unavailable — install the speech extras:"
            "  pip install -r requirements-speech.txt]"
        )
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"\n[Microphone Active ({target_language}) - Speak clearly]")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
            wav_data = audio.get_wav_data()
            return transcribe_audio_bytes(wav_data, target_language, settings.stt_codes)
        except Exception as error:
            print(f"[Microphone error: {error}]")
            return None


def run_console_app() -> None:
    settings, orchestrator = build_orchestrator(with_document_retriever=False)
    print("\nSystems online. Bao AI ready.")

    last_language = "English"

    while True:
        print("\n" + "=" * 40)
        print("BAO AI CONSOLE MODE")
        print("1. Speak to Bao (Microphone)")
        print("2. Type to Bao (Keyboard)")
        print("3. Exit")
        print("=" * 40)

        choice = input("Select an option (1, 2, or 3): ").strip()
        if choice == "1":
            text = _listen(settings, target_language=last_language)
            if text:
                print(f"Heard: '{text}'")
                result = orchestrator.handle(text, want_speech=True)
                last_language = result.detected_language
                print(f"[{result.detected_language} · {result.confidence * 100:.1f}%] {result.text}")
                _play_audio_bytes(result.audio, result.audio_mime)
        elif choice == "2":
            text = input("Type your sentence here: ").strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit", "stop"}:
                print("Goodbye.")
                break
            result = orchestrator.handle(text, want_speech=True)
            last_language = result.detected_language
            print(f"[{result.detected_language} · {result.confidence * 100:.1f}%] {result.text}")
            _play_audio_bytes(result.audio, result.audio_mime)
        elif choice == "3":
            print("Shutting down. Goodbye.")
            break
        else:
            print("Please choose 1, 2, or 3.")


if __name__ == "__main__":
    run_console_app()