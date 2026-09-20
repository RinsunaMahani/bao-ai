"""
Bao AI - Composition root.

The one place the object graph is assembled.

`ai/orchestrator.py` says the two interfaces "can never again drift into
implementing the pipeline differently." That was true of the pipeline and
false of everything around it: `ui/console_app.build_orchestrator()` and
`ui/streamlit_app.init_system()` each hand-built the same six components
with the same arguments, in two files. Change a detector argument or a
threshold in one and the console and the web app silently disagree — the
exact class of drift the orchestrator exists to prevent, just moved one
level out.

So construction lives here, and the interfaces call it. Streamlit still
wraps the call in `@st.cache_resource` and adds its own presentation-only
pieces (the sign recognizer and phrase book, which the console has no use
for) — but the pipeline itself is assembled once, in one place.
"""
from __future__ import annotations

import copy

from bao.ai.client import GeminiClient
from bao.ai.memory import ConversationMemory
from bao.ai.orchestrator import Orchestrator
from bao.core.config import Settings
from bao.core.logging import get_logger
from bao.core.security import SecurityGuardrails
from bao.knowledge.retriever import (
    DocumentRetriever,
    KnowledgeRetriever,
    has_retrieval_support,
)
from bao.services.language_detector import (
    CompositeLanguageDetector,
    SklearnLanguageDetector,
    get_language_detector,
)
from bao.services.speech import enable_coqui_sa, load_local_voices, set_coqui_speaker

logger = get_logger(__name__)


def build_orchestrator(
    settings: Settings | None = None,
    *,
    with_document_retriever: bool = True,
) -> tuple[Settings, Orchestrator]:
    """Assembles the full pipeline and returns it with the settings used.

    Settings are returned alongside the orchestrator because every caller
    needs both, and building `Settings()` twice would re-read and re-parse
    config.toml for no reason.

    `with_document_retriever` is the one genuine difference between the
    two interfaces: the web app lets users upload files, the console has
    no upload mechanism. It is a parameter rather than two functions so
    the rest of the graph stays identical by construction.
    """
    settings = settings or Settings()

    # Registered here, before anything can ask about voice coverage.
    # config.toml documented [speech.local_voices] in detail and nothing
    # read it: there was no Settings property and no caller, so a
    # checkpoint could be trained, committed and configured while the
    # speech layer never learned it existed. Registration is global to the
    # speech module rather than passed down the graph because voice
    # coverage is a property of the machine, not of one orchestrator.
    load_local_voices(settings.local_voices)

    # Opt-in, and for a licence reason rather than a technical one: the
    # model is cc-by-nc-4.0 and this repository is MIT. See
    # Settings.coqui_sa_enabled. Registered here so the voice tiers know
    # about it before anything asks what can be spoken.
    enable_coqui_sa(settings.coqui_sa_enabled)
    set_coqui_speaker(settings.coqui_sa_speaker)

    knowledge_retriever = (
        KnowledgeRetriever(
            data_path=settings.knowledge_base_path,
            threshold=settings.similarity_threshold,
            min_coverage=settings.min_query_coverage,
        )
        if has_retrieval_support()
        else None
    )
    if knowledge_retriever is None:
        logger.warning("scikit-learn unavailable — running without knowledge retrieval.")

    detector = get_language_detector(
        prefer_ml=True,
        model_path=settings.classifier_model_path,
        tokenizer_config_path=settings.tokenizer_config_path,
    )

    # The composite is CONSTRUCTED whenever a bundle exists, but starts in
    # whatever state config.toml specifies. Building it unconditionally is
    # what lets the UI offer a runtime toggle: flipping `.enabled` costs
    # nothing, whereas rebuilding the pipeline would mean invalidating
    # Streamlit's cache and reloading the TFLite model mid-session.
    #
    # Disabled, the composite returns the primary's result untouched, so
    # the default path is unchanged rather than merely intended to be.
    pan_african = SklearnLanguageDetector(settings.pan_african_model_path)
    if pan_african.is_available():
        detector = CompositeLanguageDetector(
            primary=detector,
            secondary=pan_african,
            enabled=settings.pan_african_enabled,
        )
        state = "enabled" if settings.pan_african_enabled else "available but off"
        logger.info(f"Pan-African detector {state}.")
    elif settings.pan_african_enabled:
        logger.warning(
            "enable_pan_african is true but no usable bundle was found at "
            f"{settings.pan_african_model_path} — continuing with the primary detector."
        )

    orchestrator = Orchestrator(
        security=SecurityGuardrails(max_length=settings.max_query_length),
        language_detector=detector,
        knowledge_retriever=knowledge_retriever,
        gemini_client=GeminiClient(settings),
        memory=ConversationMemory(),
        document_retriever=DocumentRetriever() if with_document_retriever else None,
        tts_codes=settings.mms_codes,
        tts_speaking_rate=settings.tts_speaking_rate,
        tts_noise_scale=settings.tts_noise_scale,
        tts_backend=settings.tts_backend,
        voice_fallback_related=settings.voice_fallback_related,
        voice_fallback_english=settings.voice_fallback_english,
        min_detection_confidence=settings.min_detection_confidence,
        max_speech_characters=settings.max_speech_characters,
    )
    return settings, orchestrator


def for_session(shared: Orchestrator) -> Orchestrator:
    """A per-session orchestrator that reuses the expensive, read-only
    parts of `shared` and gets its own conversation and uploads.

    Streamlit's `@st.cache_resource` caches across ALL users, sessions and
    reruns — that is its documented purpose, and for the heavy read-only
    pieces it is exactly right: the TFLite classifier, the fitted knowledge
    base and the API client are identical for everyone and cost seconds to
    build.

    Two of the orchestrator's parts are not read-only, and sharing those
    leaked one person's data into another's request:

      - ConversationMemory is prepended to every prompt, so a second
        visitor's question arrived carrying the first visitor's
        conversation as context.
      - DocumentRetriever holds uploaded files, so a document one person
        uploaded was retrievable by the next. Demonstrated with a private
        results file: three differently-worded questions from a second
        session all returned it.

    Neither was visible from the screen, which is what made it dangerous:
    `display_messages` lives in session_state and is correctly per-session,
    so each visitor saw only their own chat bubbles while the model
    received everybody's.

    A shallow copy rather than a re-listed constructor call. Every tuning
    value — voice codes, speaking rate, thinking level, the speech cap —
    is carried across automatically, so a parameter added to Orchestrator
    later cannot silently stop reaching session orchestrators. Only the
    fields that must not be shared are replaced.
    """
    session = copy.copy(shared)
    session.memory = ConversationMemory()
    session.document_retriever = DocumentRetriever()

    # The composite detector carries a mutable `enabled` flag that the
    # sidebar toggles. Copied too, so one visitor switching the
    # pan-African languages on does not switch them on for everyone. The
    # copy is shallow, so both still share the loaded models underneath
    # and this costs nothing.
    if isinstance(session.language_detector, CompositeLanguageDetector):
        session.language_detector = copy.copy(session.language_detector)

    return session
