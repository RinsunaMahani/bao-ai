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
from bao.services.speech import SPEAK_AUTO, speech_input_language, transcribe_audio_bytes


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


def _reply(orchestrator, text: str) -> str:
    """Runs one turn, prints and plays it, and returns the detected
    language. Shared by the typed and spoken paths so they cannot report
    a turn differently.
    """
    result = orchestrator.handle(text, want_speech=True)
    shown = result.reply_language or result.detected_language
    print(f"[{shown} · {result.confidence * 100:.1f}%] {result.text}")
    # The web app shows these; the console printed neither, so a missing
    # or substituted voice was silent here in both senses.
    if result.voice_note:
        print(f"[Voice: {result.voice_note}]")
    elif result.speech_error and not result.audio:
        print(f"[No audio: {result.speech_error}]")
    _play_audio_bytes(result.audio, result.audio_mime)
    return result.detected_language


def _choose_speaking_language(settings: Settings, current: str) -> str:
    """Asks which language the microphone should listen for. Speech
    recognition must be told before it hears anything — see
    speech_input_language — so this cannot be detected automatically.
    """
    options = [SPEAK_AUTO, *settings.stt_codes]
    print("\nWhich language will you speak?")
    for number, name in enumerate(options, 1):
        print(f"  {number:>2}. {name}" + ("   <- current" if name == current else ""))
    raw = input("Number (Enter to keep the current choice): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return current


def run_console_app() -> None:
    settings, orchestrator = build_orchestrator(with_document_retriever=False)
    print("\nSystems online. Bao AI ready.")

    last_language = "English"
    speaking_choice = SPEAK_AUTO

    while True:
        listening_for = speech_input_language(speaking_choice, last_language, settings.stt_codes)
        print("\n" + "=" * 40)
        print("BAO AI CONSOLE MODE")
        print(f"1. Speak to Bao (Microphone, listening for {listening_for})")
        print("2. Type to Bao (Keyboard)")
        print("3. Choose the language I'll speak")
        print("4. Exit")
        print("=" * 40)

        choice = input("Select an option (1-4): ").strip()
        if choice == "1":
            text = _listen(settings, target_language=listening_for)
            if text:
                print(f"Heard: '{text}'")
                last_language = _reply(orchestrator, text)
        elif choice == "2":
            text = input("Type your sentence here: ").strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit", "stop"}:
                print("Goodbye.")
                break
            last_language = _reply(orchestrator, text)
        elif choice == "3":
            speaking_choice = _choose_speaking_language(settings, speaking_choice)
        elif choice == "4":
            print("Shutting down. Goodbye.")
            break
        else:
            print("Please choose 1, 2, 3 or 4.")


if __name__ == "__main__":
    run_console_app()