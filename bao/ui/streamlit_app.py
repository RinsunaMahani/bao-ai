"""
Bao AI — Streamlit Web Application Entrypoint.

Run with: streamlit run bao/ui/streamlit_app.py
"""

import inspect

import psutil
import streamlit as st

from bao.ai.orchestrator import Orchestrator
from bao.bootstrap import build_orchestrator, for_session
from bao.core.config import ASSISTANT_LOGO_PATH, LABELS, PAN_AFRICAN_LABELS, Settings
from bao.knowledge.loader import extract_text_from_bytes, has_pdf_support
from bao.services.language_detector import CompositeLanguageDetector
from bao.services.speech import (
    has_edge_backend,
    has_mms_backend,
    has_stt_backend,
    has_tts_backend,
    transcribe_audio_bytes,
    tts_availability,
    voice_coverage,
)
from bao.ui.components import thinking_indicator_html

USER_AVATAR = ":material/person:"

# st.chat_input gained a built-in microphone (accept_audio) in a later
# Streamlit than this project's floor of 1.40. Probed rather than
# version-compared, because the feature is what matters, not the number —
# and an older install falls back to the separate recorder below instead
# of crashing on an unexpected keyword.
_CHAT_INPUT_HAS_MIC = "accept_audio" in inspect.signature(st.chat_input).parameters

st.set_page_config(
    page_title="Bao AI — Multilingual Assistant",
    page_icon=ASSISTANT_LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _shared_pipeline():
    """Built ONCE for the whole server, and shared by every visitor.

    Correct for what it holds: the TFLite classifier, the fitted knowledge
    base and the API client are identical for everybody and cost seconds
    to construct, so rebuilding them per session would be waste.

    It must not be used directly — see init_system.
    """
    return build_orchestrator()


def init_system():
    """The pipeline for THIS visitor.

    `@st.cache_resource` caches across all users, sessions and reruns, so
    handing its object straight to the page shared one conversation memory
    and one uploaded-document store between everyone connected. A second
    visitor's question arrived carrying the first visitor's conversation,
    and a file one person uploaded was retrievable by the next.

    That was invisible from the screen, which is what made it dangerous:
    `display_messages` is per-session and correct, so each visitor saw
    only their own chat bubbles while the model received everybody's.

    So the heavy read-only parts stay shared and the mutable ones are
    per-session, held in session_state rather than rebuilt on each rerun.
    """
    settings, shared = _shared_pipeline()
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = for_session(shared)
    orchestrator = st.session_state.orchestrator
    return settings, orchestrator, orchestrator.document_retriever


def render_sidebar(settings: Settings, orchestrator: Orchestrator) -> None:
    with st.sidebar:
        st.image(ASSISTANT_LOGO_PATH, width=120)
        st.title("Bao AI")
        st.caption("Multilingual South African AI Assistant")
        st.caption("Text-first, with optional speech output.")
        st.caption("\"Bao\" — short for baobab: deep roots, offline-first.")
        st.markdown("---")

        st.subheader("System Status")
        if orchestrator.gemini_client.is_available():
            st.success("Online (Gemini + RAG)")
        else:
            st.warning("Offline (Local RAG Only)")

        st.subheader("Live Telemetry")
        mem = psutil.virtual_memory()
        st.metric("RAM Used", f"{mem.used / (1024**3):.2f} GB", f"{mem.percent}%")
        timings = st.session_state.get("last_timings") or {}
        if "last_latency" in st.session_state:
            # "Time to answer", not "latency". latency_ms is measured when
            # handle() returns, which is when the TEXT is on screen; speech
            # is synthesized afterwards and its time lands in stage_timings.
            # Labelling that total "latency" while listing a speech stage
            # underneath it meant the stages visibly summed to more than the
            # figure above them, with nothing explaining why.
            st.metric("Time to answer", f"{st.session_state['last_latency']:.0f} ms")
            speech_ms = timings.get("speech")
            if speech_ms:
                st.metric(
                    "Voice (after the text)", f"{speech_ms:.0f} ms",
                    help="Synthesis runs after the answer is already on screen, "
                         "so it does not delay reading it.",
                )
        if timings:
            # Which stage was slow, not just that the turn was. Retrieval
            # is sub-millisecond; if a turn felt slow, this says whether
            # to blame the network or the voice.
            st.caption(
                " · ".join(f"{name} {ms:.0f}ms" for name, ms in timings.items())
            )

        st.markdown("---")
        st.subheader("Language Coverage")

        # Runtime toggle rather than a config edit. The composite detector
        # is always constructed when its bundle exists, so flipping this
        # costs nothing — no pipeline rebuild, no model reload.
        #
        # Safe to mutate because `for_session` gave this visitor their own
        # copy of the composite. It previously mutated the object shared by
        # every visitor, so one person switching these languages on
        # switched them on for everybody.
        detector = orchestrator.language_detector
        if isinstance(detector, CompositeLanguageDetector):
            enabled = st.checkbox(
                "Include 14 more African languages",
                value=detector.enabled,
                key="pan_african_enabled",
                help=(
                    "Amharic, French, Hausa, Igbo, Lingala, Luganda, Oromo, "
                    "Nigerian Pidgin, Kirundi, Shona, Somali, Swahili, "
                    "Tigrinya, Yoruba. Consulted only when the South African "
                    "classifier is unsure, so the 11 are unaffected. Works on "
                    "phrases rather than single words."
                ),
            )
            detector.enabled = enabled
            pan_african_live = enabled
            if enabled:
                st.caption("25 languages detected · 11 with curated answers")
            else:
                st.caption("11 South African languages")
        else:
            pan_african_live = False
            st.caption("11 South African languages")
            st.caption(
                "The 14-language model is not loaded — see "
                "`scripts/build_pan_african_bundle.py`."
            )

        st.markdown("### Supported Languages")
        st.caption(
            "isiZulu, Sepedi, Setswana, isiXhosa, Afrikaans, English, "
            "isiNdebele, siSwati, Tshivenda, Xitsonga, Sesotho"
        )

        st.markdown("---")
        st.subheader("Speech Output")
        st.checkbox(
            "Speak replies",
            value=True,
            key="speech_enabled",
            help=(
                "Synthesis is the slowest stage in the pipeline. Text always "
                "appears first; turning this off skips the voice entirely."
            ),
            disabled=not has_tts_backend(),
        )
        # Reports what this process has actually loaded, not what is
        # theoretically supported. The old caption said "Available" whenever
        # any backend was importable, which reads as "all eleven languages
        # have a voice" even when most of them had failed to load.
        availability = tts_availability()
        loaded = [c for c, r in availability.items() if r == "loaded"]
        failures = {c: r for c, r in availability.items() if r != "loaded"}

        if has_edge_backend():
            st.success("Real SA voices for English, Afrikaans, isiZulu")
            if not has_mms_backend():
                st.caption(
                    "The other eight need the offline MMS voices — "
                    "install `requirements-speech.txt`."
                )
        elif has_tts_backend():
            st.success("Offline MMS voices")
            st.caption(
                "No en-ZA voice offline — English is spoken with Afrikaans "
                "phonetics. Install `edge-tts` for real South African English."
            )
        else:
            st.info("Not installed — install `requirements-speech.txt` for spoken replies.")

        # The twelve pan-African voices are only reachable when their
        # languages can be detected, so they are reported only when that
        # detector is on — otherwise the panel would list voices for
        # languages the app will never identify.
        covered_languages = LABELS + PAN_AFRICAN_LABELS if pan_african_live else LABELS
        coverage = voice_coverage(settings.mms_codes, languages=covered_languages)
        # startswith, not equality: a locally trained checkpoint reports as
        # "native (locally trained)", which is the BEST tier, not an
        # approximation. Exact matching excluded it from the count and gave
        # it the amber "substituted voice" icon.
        native = [lang for lang, tier in coverage.items() if tier.startswith("native")]
        st.caption(f"Native voices: {len(native)} of {len(coverage)}")
        with st.expander("Voice coverage by language"):
            for language, tier in coverage.items():
                if tier.startswith("native"):
                    icon = "🟢"
                elif tier == "text only":
                    icon = "⚪"
                else:
                    icon = "🟡"
                st.caption(f"{icon} **{language}** — {tier}")
            silent = [lang for lang, tier in coverage.items() if tier == "text only"]
            if silent:
                st.caption(
                    f"{len(silent)} of {len(coverage)} have no open text-to-speech "
                    "model. Fallbacks are configurable in config.toml and off by "
                    "default."
                )

        if loaded:
            st.caption(f"Voices loaded this session: {len(loaded)}")
        if failures:
            with st.expander(f"{len(failures)} voice(s) unavailable"):
                for code, reason in failures.items():
                    st.caption(f"**{code}** — {reason}")
                st.caption("Run `python scripts/probe_tts.py` to check all eleven.")

        st.markdown("---")
        uploaded_files = st.file_uploader(
            "Upload documents to index into RAG memory",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Index Uploaded Documents"):
            with st.spinner("Extracting text & updating index..."):
                total_chunks, file_count = 0, 0
                for file in uploaded_files:
                    text = extract_text_from_bytes(file.getvalue(), file.name)
                    if text.strip():
                        total_chunks += orchestrator.document_retriever.add_document(file.name, text)
                        file_count += 1
                if file_count:
                    st.success(f"Indexed {file_count} doc(s) into {total_chunks} chunks.")
                elif not has_pdf_support():
                    st.warning("No readable text found. (Install `pypdf` to enable PDF uploads.)")
                else:
                    st.warning("No readable text found in uploaded files.")


def _format_detection_badge(result) -> str:
    if result.detection_backend == "heuristic":
        detection_part = f"Language: {result.detected_language} (keyword match)"
    elif result.detection_backend == "tflite":
        detection_part = f"Language: {result.detected_language} (ML model, {result.confidence * 100:.0f}%)"
    elif result.detection_backend == "keras":
        detection_part = (
            f"Language: {result.detected_language} "
            f"(ML model via Keras fallback, {result.confidence * 100:.0f}%)"
        )
    else:
        detection_part = f"Language: {result.detected_language}"
    # When detection was too weak to act on, say so. Otherwise the badge
    # claims a language the reply was not actually written in.
    if result.reply_language and result.reply_language != result.detected_language:
        detection_part += f" — too unsure to use, replying in {result.reply_language}"
    badge = f"{detection_part} · source: {result.source}"
    # Cross-lingual turns ("explain X in Xitsonga") answer in a different
    # language from the question. Showing it makes a wrong voice obvious.
    if result.speech_language and result.speech_language != result.detected_language:
        badge += f" · voice: {result.speech_language}"
    return badge


def render_message(msg: dict, assistant_avatar: str) -> None:
    avatar = assistant_avatar if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        if "lang_badge" in msg:
            st.caption(msg["lang_badge"])
        st.markdown(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format=msg.get("audio_mime", "audio/wav"))


def _transcribe(settings: Settings, audio_file) -> str | None:
    """Turns a recorded clip into text, or returns None with a visible
    warning. Shared by the in-chat microphone and the legacy recorder so
    both paths behave identically.
    """
    try:
        text = transcribe_audio_bytes(
            audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file.read(),
            st.session_state.get("last_language", "English"),
            settings.stt_codes,
        )
        return text or None
    except Exception as e:
        st.warning(f"Could not transcribe audio: {e}")
        return None


def render_legacy_voice_input(settings: Settings, orchestrator: Orchestrator) -> None:
    """Fallback recorder for Streamlit builds whose chat_input has no
    microphone. Only rendered when the probe at the top of this file says
    the built-in one is unavailable — on a current Streamlit the mic lives
    in the chat box, where people expect it.
    """
    if not has_stt_backend() or _CHAT_INPUT_HAS_MIC:
        return
    with st.expander("Voice Input (record from microphone)"):
        recorded = st.audio_input("Speak to Bao", key="voice_recorder")
        if recorded is not None:
            transcription = _transcribe(settings, recorded)
            if transcription:
                st.write(f"Transcribed: **{transcription}**")
                if st.button("Send voice transcription"):
                    _handle_turn(settings, orchestrator, transcription, display_prefix="[Voice] ")


def _handle_turn(
    settings: Settings,
    orchestrator: Orchestrator,
    user_input: str,
    display_prefix: str = "",
    language_override: str | None = None,
) -> None:
    st.session_state.display_messages.append({"role": "user", "content": display_prefix + user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(display_prefix + user_input)

    with st.chat_message("assistant", avatar=settings.assistant_avatar):
        placeholder = st.empty()
        placeholder.markdown(thinking_indicator_html(), unsafe_allow_html=True)

        # Render Gemini's reply as it arrives rather than after it lands.
        # Total generation time is unchanged; what changes is that the
        # first words appear in well under a second instead of after the
        # whole (often multi-paragraph) answer is complete.
        streamed: list[str] = []

        def on_chunk(piece: str) -> None:
            streamed.append(piece)
            placeholder.markdown("".join(streamed))

        # want_speech=False on purpose: the text is shown the moment it's
        # ready, and the voice is synthesized afterwards (below). Leaving
        # it True made every turn feel as slow as its slowest stage.
        result = orchestrator.handle(
            user_input,
            want_speech=False,
            on_chunk=on_chunk,
            language_override=language_override,
        )

        placeholder.empty()

        # Drawn into a placeholder and rewritten after speech, because half
        # of what the badge reports is not known until then: `speak()` is
        # what sets `speech_language`, so a badge rendered once — before it
        # ran — silently dropped the "voice: ..." suffix that exists to make
        # a cross-lingual voice visible. That suffix has been dead since
        # synthesis moved out of the blocking path.
        badge_slot = st.empty()
        badge = _format_detection_badge(result)
        badge_slot.caption(badge)
        st.markdown(result.text)

        # Text is on screen and the turn is usable from here on. Speech is
        # strictly additive, so it runs last and behind a toggle.
        if st.session_state.get("speech_enabled", True):
            with st.spinner("Generating voice..."):
                orchestrator.speak(result)
            badge = _format_detection_badge(result)
            badge_slot.caption(badge)
            if result.audio:
                st.audio(result.audio, format=result.audio_mime)
                if result.voice_note:
                    # A substitution the listener is told about is a
                    # fallback; one they are not told about is a
                    # misrepresentation.
                    st.caption(result.voice_note)
            elif result.speech_error:
                st.caption(result.speech_error)

        st.session_state["last_latency"] = result.latency_ms
        st.session_state["last_timings"] = result.stage_timings
        st.session_state["last_language"] = result.detected_language
        st.session_state.display_messages.append({
            "role": "assistant",
            "content": result.text,
            "lang_badge": badge,
            "audio": result.audio,
            "audio_mime": result.audio_mime,
        })


def render_chat_input(settings: Settings) -> tuple[str, str] | None:
    """Draws the chat box and returns (text, display_prefix), or None.

    Speech is an input MODE, not a separate feature panel, so the
    microphone belongs inside the message box next to the send button —
    where every other chat product puts it — rather than in an expander
    above the transcript that has to be opened first.
    """
    can_record = has_stt_backend() and _CHAT_INPUT_HAS_MIC

    if not can_record:
        typed = st.chat_input("Type your message here...")
        return (typed, "") if typed else None

    submitted = st.chat_input(
        "Type a message, or use the mic to speak...",
        accept_audio=True,
    )
    if not submitted:
        return None

    # accept_audio makes chat_input return a value object rather than a
    # plain string, so both fields have to be read.
    text = (submitted.text or "").strip()
    if text:
        return text, ""

    if submitted.audio is not None:
        with st.spinner("Transcribing..."):
            transcription = _transcribe(settings, submitted.audio)
        if transcription:
            return transcription, "[Voice] "
    return None


def main() -> None:
    settings, orchestrator, _ = init_system()
    render_sidebar(settings, orchestrator)

    st.header("Bao AI Conversational Interface")
    st.caption("Ask questions in any of South Africa's 11 spoken official languages. "
               "SASL, the 12th, is not supported.")

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    render_legacy_voice_input(settings, orchestrator)

    for msg in st.session_state.display_messages:
        render_message(msg, assistant_avatar=settings.assistant_avatar)

    submitted = render_chat_input(settings)
    if submitted:
        user_input, prefix = submitted
        _handle_turn(settings, orchestrator, user_input, display_prefix=prefix)


if __name__ == "__main__":
    main()