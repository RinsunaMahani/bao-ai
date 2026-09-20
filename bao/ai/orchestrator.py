"""Bao AI - Request Orchestrator.

Implements the pipeline:

    User -> Security -> Language Detection -> Memory -> Knowledge Retrieval
         -> Gemini -> Translation (if needed) -> Speech (optional) -> Response

Every stage is a call into a module that has exactly one job (core.security,
services.language_detector, ai.memory, knowledge.retriever, ai.client,
services.translation, services.speech). This class's only responsibility is
sequencing them and deciding which branch to take — it should never contain
business logic that belongs in one of those modules.

Both bao/ui/streamlit_app.py and bao/ui/console_app.py call this same
class, so the two UIs can never again drift into implementing the pipeline
differently from each other (which is exactly what happened between the
old app.py and nompilo_web.py).

TWO DELIBERATELY SEPARATE ANSWER SOURCES, not one blended system:
  - The curated knowledge base (`knowledge_retriever`) returns VERIFIED
    facts, word for word or translated, never rephrased or summarized by
    an LLM. `source="knowledge_base"` in the result means "this exact
    answer is in a human-curated CSV," a stronger guarantee than anything
    generated.
  - Uploaded documents (`document_retriever`) feed a Gemini prompt as
    unverified context — Gemini synthesizes an answer FROM them, it
    doesn't return them verbatim. `source="gemini"` (with doc context)
    means "generated, grounded in what you uploaded, not independently
    checked."
This distinction is a real, explainable design decision, not an
accident of control flow: a knowledge-base match always wins over
document context and never gets sent through an LLM.

A previously discarded case now handled: if there's no knowledge-base
match AND Gemini is unreachable AND a document match exists, the old
behavior silently dropped the document match and returned the generic
"no answer" message — the retrieval work happened for nothing. That
document-search feature was effectively online-only despite living in an
otherwise offline-first app. It now surfaces the raw matched excerpt
instead (source="document_context"), clearly labeled as an unprocessed
excerpt rather than a generated answer, since offline mode has no way to
summarize or verify it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from bao.ai.client import GeminiClient
from bao.ai.memory import ConversationMemory
from bao.ai.prompts import (
    GENERATION_ERROR_MESSAGE,
    OFFLINE_NO_KEY_MESSAGE,
    OFFLINE_NO_MATCH_MESSAGE,
    generation_busy_message,
    offline_document_excerpt,
    open_ended_prompt,
    system_instruction,
)
from bao.core.exceptions import (
    GenerationError,
    GenerationUnavailableError,
    SecurityViolationError,
)
from bao.core.logging import get_logger
from bao.core.security import SecurityGuardrails
from bao.knowledge.retriever import DocumentRetriever, KnowledgeRetriever
from bao.services.language_detector import (
    DetectionResult,
    LanguageDetector,
    named_target_language,
)
from bao.services.offline import should_use_offline
from bao.services.speech import resolve_voice_language, synthesize_speech
from bao.services.translation import translate_fact

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    text: str
    detected_language: str
    confidence: float
    # "heuristic" | "tflite" — which detector produced `confidence`, so
    # callers don't present it as calibrated when it isn't.
    detection_backend: str
    # "knowledge_base" | "gemini" | "document_context" | "offline_fallback"
    # | "provider_busy" | "blocked"
    source: str
    latency_ms: float
    audio: bytes | None = None
    audio_mime: str = "audio/wav"
    # Per-stage milliseconds. `latency_ms` alone says a turn was slow; it
    # never said WHICH stage was slow, so tuning was guesswork. Retrieval
    # is sub-millisecond here — the time is always in generation or
    # speech, and this makes that visible instead of assumed.
    stage_timings: dict[str, float] = field(default_factory=dict)
    # The language the voice actually used. Usually equal to
    # detected_language, but NOT when the reply is in a different language
    # from the question (see _speech_language). Surfaced so a wrong voice
    # is visible in the UI instead of being a silent mismatch.
    speech_language: str | None = None
    # Why there is no audio, when there is no audio. Surfaced in the UI so
    # a missing voice is diagnosable instead of just absent.
    speech_error: str | None = None
    # Whether the caller pinned the language rather than it being detected.
    # Lives on the RESULT, not the orchestrator: speak() is a separate call,
    # so orchestrator-level state describes whichever turn ran most
    # recently, not the turn being spoken.
    language_was_overridden: bool = False
    # Whether the QUESTION named the language ("...in Xitsonga"). Treated
    # the same as an override when choosing the voice: the user said it, so
    # a detector reading of the reply must not quietly overrule them.
    language_was_requested: bool = False
    # Set when the reply was spoken by a language other than its own, or not
    # spoken at all. Shown to the user — a substitution they are told about
    # is a fallback; one they are not told about is a misrepresentation.
    voice_note: str | None = None
    # The language actually used to generate the reply. Usually equal to
    # detected_language — but NOT when detection was too unconfident to act
    # on, in which case the detection is still reported honestly and English
    # is used instead. Separate fields because "what the model said" and
    # "what we did about it" are different claims.
    reply_language: str | None = None


class Orchestrator:
    def __init__(
        self,
        security: SecurityGuardrails,
        language_detector: LanguageDetector,
        knowledge_retriever: KnowledgeRetriever | None,
        gemini_client: GeminiClient,
        memory: ConversationMemory | None = None,
        document_retriever: DocumentRetriever | None = None,
        tts_codes: dict[str, str] | None = None,
        tts_speaking_rate: float | None = None,
        tts_noise_scale: float | None = None,
        tts_backend: str = "auto",
        voice_fallback_related: bool = False,
        voice_fallback_english: bool = False,
        min_detection_confidence: float = 0.0,
        max_speech_characters: int = 0,
    ):
        self.security = security
        self.language_detector = language_detector
        self.knowledge_retriever = knowledge_retriever
        self.gemini_client = gemini_client
        self.memory = memory or ConversationMemory()
        self.document_retriever = document_retriever
        # Previously not plumbed through at all: config.toml exposed
        # [languages].mms_codes and Settings.mms_codes read it, but nothing
        # ever passed it to synthesize_speech, so the call always fell back
        # to DEFAULT_MMS_CODES. Editing the voice mapping in config.toml
        # silently did nothing — a dead knob that looked live.
        self.tts_codes = tts_codes
        self.tts_speaking_rate = tts_speaking_rate
        self.tts_noise_scale = tts_noise_scale
        self.tts_backend = tts_backend
        self.voice_fallback_related = voice_fallback_related
        self.voice_fallback_english = voice_fallback_english
        self.min_detection_confidence = min_detection_confidence
        self.max_speech_characters = max_speech_characters

    def handle(
        self,
        user_input: str,
        force_offline: bool = False,
        want_speech: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        language_override: str | None = None,
    ) -> PipelineResult:
        """Runs one full turn.

        `on_chunk`, when given, is called with pieces of the reply as
        Gemini produces them, so a UI can render progressively. It changes
        nothing about the returned result — callers that don't care (the
        console, the tests) pass nothing and get identical behaviour.

        `want_speech=True` synthesizes audio inline, which BLOCKS the
        return until the voice is ready. That is correct for the console
        (it plays audio and nothing else is waiting) but wrong for a chat
        UI, where it means the user stares at a spinner while text that
        was ready seconds ago waits on a voice. UIs should leave this
        False and call `speak()` after rendering — see its docstring.

        `language_override` skips detection and forces the reply language,
        for input that carries no language of its own to detect or where
        the user has stated a preference explicitly.

        It was introduced for sign input (a recognised sign is a gloss, not
        a sentence in any spoken language) and outlived that feature. Kept
        because it is the mechanism a "reply in ..." selector would use,
        and because it is what makes the pan-African languages reachable
        when the detector is unsure. Currently exercised by the tests and
        by scripts/gates.py, not by either interface — remove it if that is
        still true when the project is next tidied.
        """
        start = time.perf_counter()
        timings: dict[str, float] = {}

        # 1. Security
        try:
            self.security.validate_or_raise(user_input)
        except SecurityViolationError as e:
            return PipelineResult(
                text=f"Your message was blocked: {e}",
                detected_language="Unknown",
                confidence=0.0,
                detection_backend="none",
                source="blocked",
                latency_ms=self._elapsed_ms(start),
            )

        # 2. Language detection
        stage = time.perf_counter()
        if language_override:
            detection = DetectionResult(
                language=language_override, confidence=1.0, backend="override"
            )
        else:
            detection = self.language_detector.detect(user_input)
        timings["detection"] = self._elapsed_ms(stage)

        # A weak detection is reported but not acted on. Detection chooses
        # the language Gemini answers in, so acting on a 42% guess produces
        # an answer in the wrong language — a much worse outcome than a
        # wrong badge. English is the fallback because it is this app's
        # lingua franca, and because an explicit "in isiXhosa" in the
        # question is still honoured by the prompt itself.
        reply_language = detection.language
        if (
            not language_override
            # A detection the composite detector explicitly vouched for is
            # not re-judged here. Two models trained on different corpora
            # produce confidences on different scales, so one number cannot
            # gate both: the pan-African model scores 16-23% on input that
            # is not its language and 37-95% on input that is, so 43% from
            # it is a clear signal — while 43% from the LSTM is a coin flip.
            #
            # CompositeLanguageDetector has already applied its own
            # calibrated floor (secondary_min) before returning a sklearn
            # answer. Applying this floor on top created a dead band between
            # the two thresholds where a correct detection was accepted and
            # then silently discarded: "sannu" was identified as Hausa at
            # 43% and answered in English.
            and detection.backend != "sklearn"
            and detection.confidence < self.min_detection_confidence
        ):
            logger.info(
                f"Detection too weak to act on ({detection.language} "
                f"{detection.confidence:.0%}); replying in English."
            )
            reply_language = "English"

        # An explicitly named language outranks everything above it.
        #
        # "explain calculus in Xitsonga" is WRITTEN in English, so detection
        # correctly says English and the system instruction then reads
        # "Primary response language: English" — and the model obeys that
        # header over the request inside the sentence. Measured on
        # gemini-3.5-flash-lite: "explain gravity in Afrikaans" came back in
        # English on both attempts. It is not a detection failure; detection
        # was right. The pipeline was answering a different question from
        # the one asked.
        #
        # Skipped when the caller pinned the language, since an explicit
        # override is a stronger statement than a phrase in the text.
        language_was_requested = False
        if not language_override:
            requested = named_target_language(user_input)
            if requested and requested != reply_language:
                logger.info(
                    f"Question asks for {requested}; answering in it rather than "
                    f"{reply_language}."
                )
                reply_language = requested
            language_was_requested = requested is not None

        # 3. Knowledge retrieval (verified facts first, then session documents)
        #
        # Uses reply_language, not detection.language. Round 31 added the
        # confidence floor to generation but left retrieval on the raw
        # detection, so a 35% guess still chose which row was returned: the
        # badge read "too unsure to use, replying in English" while the
        # answer came back from the siSwati row. One decision, one input.
        stage = time.perf_counter()
        # The detected language is passed to retrieval so that shared
        # greetings ("Dumela" — Sepedi, Sesotho and Setswana) resolve to the
        # row for the language the user is actually speaking, instead of
        # whichever row TF-IDF happened to score highest.
        fact = (
            self.knowledge_retriever.lookup(user_input, prefer_language=reply_language)
            if self.knowledge_retriever else None
        )
        doc_context = self.document_retriever.search(user_input) if self.document_retriever else ""
        timings["retrieval"] = self._elapsed_ms(stage)

        offline = should_use_offline(self.gemini_client, force_offline=force_offline)

        # 4/5/6. Generation branch + translation
        stage = time.perf_counter()
        if fact:
            text, source = self._respond_with_fact(fact, reply_language, offline)
        elif offline:
            if doc_context:
                text, source = offline_document_excerpt(doc_context), "document_context"
            else:
                text, source = OFFLINE_NO_MATCH_MESSAGE, "offline_fallback"
        else:
            text, source = self._respond_with_gemini(
                user_input, reply_language, doc_context, on_chunk=on_chunk
            )
        timings["generation"] = self._elapsed_ms(stage)

        # Memory update happens after generation, so the assistant's own
        # reply is available as context for the *next* turn.
        self.memory.add("user", user_input, language=detection.language)
        self.memory.add("assistant", text, language=detection.language)

        result = PipelineResult(
            text=text,
            detected_language=detection.language,
            confidence=detection.confidence,
            detection_backend=detection.backend,
            source=source,
            latency_ms=self._elapsed_ms(start),
            stage_timings=timings,
            reply_language=reply_language,
            language_was_overridden=language_override is not None,
            language_was_requested=language_was_requested,
        )

        # 7. Speech (optional, and deliberately last)
        if want_speech:
            self.speak(result)
            result.latency_ms = self._elapsed_ms(start)

        return result

    def speak(self, result: PipelineResult) -> PipelineResult:
        """Attaches audio to an already-complete result, in place.

        Separate from `handle()` so a UI can show the text the instant
        it's ready and synthesize the voice afterwards. Synthesis is the
        slowest stage in the whole pipeline by a wide margin — MMS runs a
        VITS forward pass per sentence on CPU — so leaving it inside the
        blocking path made every turn feel as slow as its audio, even
        though the answer was sitting there finished.
        """
        stage = time.perf_counter()
        speech_language = self._speech_language(
            result.text,
            fallback=result.reply_language or result.detected_language,
            was_overridden=result.language_was_overridden or result.language_was_requested,
        )
        # Which language will actually be spoken, and why — Tiers 1 to 4.
        spoken_language, note = resolve_voice_language(
            speech_language,
            mms_codes=self.tts_codes,
            allow_related=self.voice_fallback_related,
            allow_english=self.voice_fallback_english,
        )
        result.voice_note = note
        if spoken_language is None:
            result.speech_language = speech_language
            result.audio = None
            result.speech_error = note
            result.stage_timings["speech"] = self._elapsed_ms(stage)
            return result

        errors: list[str] = []
        trims: list[str] = []
        speech = synthesize_speech(
            result.text,
            spoken_language,
            mms_codes=self.tts_codes,
            speaking_rate=self.tts_speaking_rate,
            noise_scale=self.tts_noise_scale,
            backend=self.tts_backend,
            on_error=errors.append,
            max_characters=self.max_speech_characters,
            on_trim=trims.append,
        )
        # A shortened reading the listener is told about is a summary; one
        # they are not told about is the app appearing to lose the end of
        # its own answer. Appended rather than assigned, so it cannot erase
        # a substituted-voice note that matters just as much.
        if trims and speech is not None:
            result.voice_note = f"{note} {trims[-1]}".strip() if note else trims[-1]
        result.speech_language = spoken_language
        result.speech_error = errors[-1] if errors and speech is None else None
        result.audio = speech.data if speech else None
        result.audio_mime = speech.mime if speech else "audio/wav"
        result.stage_timings["speech"] = self._elapsed_ms(stage)
        return result

    # Below this, the heuristic detector scores an unmatched sentence at
    # 0.20 and a marker-word match at 0.75, so this threshold sits between
    # them on purpose.
    SPEECH_LANGUAGE_MIN_CONFIDENCE = 0.5

    # How much of a reply to read when deciding which voice speaks it.
    #
    # The whole reply is the wrong sample when the reply QUOTES another
    # language, which this assistant does constantly — lyrics, a passage
    # being translated, a term given in both languages. Observed live:
    # "ni kombela u hlaya national anthem ya shona" was answered in
    # Xitsonga with the Shona anthem quoted inside it. Over the full text
    # the detector returned Shona at 85%, so a Xitsonga reply was read
    # aloud by a Shona voice. Over the opening it returns Xitsonga at 100%.
    #
    # The opening is where an assistant speaks in its own voice — the
    # greeting and the framing sentence — before it quotes anything. That
    # is the language being SPOKEN, as opposed to the languages appearing
    # in the reply, and it is the first that the voice should follow.
    SPEECH_LANGUAGE_SAMPLE_CHARS = 300

    def _speech_language(self, text: str, fallback: str, was_overridden: bool = False) -> str:
        """Picks the voice language from the REPLY, not the question.

        These are usually the same and it was reasonable to conflate them
        — until a cross-lingual turn showed up: "explain cloud computing
        in tsonga" is detected as English (it is English), Gemini answers
        in Xitsonga, and the reply was then spoken by an English/Afrikaans
        voice reading Xitsonga words. This also fixes the reverse case
        offline, where an isiZulu question retrieves an English knowledge
        base fact that no longer gets an isiZulu voice.

        Falls back to the question's language when the detector isn't
        confident, so a weak guess on the reply can't override a solid one
        on the input.

        Only the OPENING of the reply is sampled, because this assistant
        quotes other languages constantly — lyrics, a passage being
        translated, a term given in both. See
        SPEECH_LANGUAGE_SAMPLE_CHARS for the case that showed it.

        Known limitation, worth stating rather than hiding: this is only
        as good as the detector. On genuinely code-switched replies (the
        knowledge base has bilingual greetings like "Sawubona! Unjani? How
        can I help you?") the heuristic detector's slight English baseline
        tips it to English, so a mixed reply to an isiZulu question gets
        an English voice. There is no single correct voice for a sentence
        that is half in each language. The ML classifier handles
        single-language replies far better, which is the case that
        actually matters here.
        """
        if not text.strip():
            return fallback

        if was_overridden:
            # The user explicitly asked for this language; a detector guess
            # on the reply must not quietly overrule them.
            return fallback

        # Sampled from the opening rather than the whole reply — see
        # SPEECH_LANGUAGE_SAMPLE_CHARS. Cut on a space so a word is not
        # split, which would hand the detector a fragment.
        sample = text[: self.SPEECH_LANGUAGE_SAMPLE_CHARS]
        if len(text) > self.SPEECH_LANGUAGE_SAMPLE_CHARS:
            cut = sample.rfind(" ")
            if cut > self.SPEECH_LANGUAGE_SAMPLE_CHARS // 2:
                sample = sample[:cut]

        result = self.language_detector.detect(sample)
        if result.confidence < self.SPEECH_LANGUAGE_MIN_CONFIDENCE:
            return fallback
        if result.language != fallback:
            logger.info(
                f"Reply language ({result.language}) differs from query language "
                f"({fallback}); routing speech to {result.language}."
            )
        return result.language

    def _respond_with_fact(self, fact, language: str, offline: bool) -> tuple[str, str]:
        """Returns a curated answer, translating it only when it genuinely
        needs translating.

        Three reasons the language check matters, not just the API call it
        saves. Greetings in the knowledge base are written in-language, so
        asking a translator to render a Xitsonga greeting "into Xitsonga"
        (a) spends a network round trip on the very first turn of a demo,
        (b) risks the model rewriting a correct, human-checked string into
        something worse, and (c) makes the offline path behave differently
        from the online one for no reason.
        """
        if offline:
            return fact.answer, "knowledge_base"

        # A curated answer already written in a South African language is
        # served verbatim, even when the detector thinks the user asked in
        # a DIFFERENT one. Two reasons, both observed live:
        #
        #  - It was written and human-checked in that language. Sending it
        #    through a translator throws that verification away. In
        #    practice the model also decorated it: a Setswana greeting came
        #    back as `"Lumela! Nka o me/thusa jwang?"` with quote marks, a
        #    slash artifact, and a parenthetical note about Lesotho
        #    orthography — all of which then get read aloud by the voice.
        #  - The detector is least reliable exactly where this fires.
        #    "dumela" is shared by Sepedi, Sesotho and Setswana, so the
        #    heuristic backend picks one essentially arbitrarily and the
        #    "translation" is between two languages, one of which was a
        #    coin flip.
        #
        # English-tagged rows (the factual ones) still translate normally —
        # those genuinely need it.
        if fact.language and fact.language.strip().lower() != "english":
            logger.info(f"Fact is curated in {fact.language}; serving verbatim, not translating.")
            return fact.answer, "knowledge_base"

        return translate_fact(fact.answer, language, self.gemini_client), "knowledge_base"

    def _respond_with_gemini(
        self,
        user_input: str,
        language: str,
        doc_context: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        if not self.gemini_client.is_available():
            return OFFLINE_NO_KEY_MESSAGE, "offline_fallback"
        try:
            prompt = open_ended_prompt(user_input, context=doc_context)
            memory_context = self.memory.as_context()
            if memory_context:
                prompt = f"{memory_context}\n\n{prompt}"
            instruction = system_instruction(language)

            if on_chunk is None:
                return self.gemini_client.generate(prompt, system_instruction=instruction), "gemini"

            pieces: list[str] = []
            for piece in self.gemini_client.generate_stream(prompt, system_instruction=instruction):
                pieces.append(piece)
                on_chunk(piece)
            return "".join(pieces), "gemini"
        except GenerationUnavailableError as e:
            # The provider was busy or rate-limiting, and retrying inside
            # the client did not clear it. Told apart from a real failure
            # because the fix is different: this one is "send it again",
            # and saying "I ran into a problem" instead invites the user to
            # rewrite a question that was never the problem.
            logger.warning(f"Generation unavailable after retries: {e}")
            return generation_busy_message(e.retry_after), "provider_busy"
        except GenerationError as e:
            logger.error(f"Generation failed: {e}")
            return GENERATION_ERROR_MESSAGE, "offline_fallback"

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)