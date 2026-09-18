"""Bao AI - Speech Service (Text-to-Speech / Speech-to-Text).

TWO TTS BACKENDS, chosen per language, because no single one covers the
brief:

  - **edge-tts** (`_EDGE_VOICES`) — Microsoft's neural voices, free and
    keyless, but ONLINE. It has genuine South African locales for exactly
    three of the eleven: en-ZA, af-ZA, zu-ZA. This is the only backend
    here that produces real South African *English*.
  - **MMS-TTS** (`facebook/mms-tts-*`) — Meta's VITS models, fully
    OFFLINE after a one-time download, and the only option that covers
    the remaining eight languages at all.

Why this split matters, and the trap it avoids: MMS has no en-ZA model.
The tempting workaround is to route English text through the Afrikaans
model (`"English" = "afr"`), which does change the accent — but it changes
it by applying *Afrikaans* letter-to-sound rules to English words. "very"
comes out closer to "ferry", "will" closer to "vill". That is an Afrikaans
speaker mispronouncing English, not South African English. edge-tts
`en-ZA-LukeNeural` is a real South African English voice, so when it's
available it wins for English.

The `"English" = "afr"` mapping is still in config.toml as the OFFLINE
fallback, because when there's no network the choice is only between a
neutral international-English voice and an Afrikaans-flavoured one, and
this project would rather sound local. That's a taste call, and it's
config, so it can be changed in one line.

NOT VERIFIED IN CI: neither backend can be exercised in the test
environment (no Hugging Face download, no Microsoft endpoint). The tests
cover backend *selection* and text chunking — the parts that are pure
logic. Actual audio quality has to be checked by ear on a real machine.
"""
from __future__ import annotations

import asyncio
import io
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from bao.core.config import DEFAULT_MMS_CODES, DEFAULT_STT_CODES
from bao.core.exceptions import SpeechError
from bao.core.logging import get_logger

try:
    import speech_recognition as sr
    _HAS_STT_BACKEND = True
except ImportError:
    sr = None  # type: ignore[assignment]
    _HAS_STT_BACKEND = False

try:
    import torch
    from transformers import AutoTokenizer, VitsModel
    _HAS_MMS_BACKEND = True
except Exception:  # pragma: no cover - torch/CUDA mismatches raise non-ImportError
    torch = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    VitsModel = None  # type: ignore[assignment]
    _HAS_MMS_BACKEND = False

try:
    import edge_tts
    _HAS_EDGE_BACKEND = True
except ImportError:
    edge_tts = None  # type: ignore[assignment]
    _HAS_EDGE_BACKEND = False

try:
    # The maintained fork. The original `TTS` package caps at Python <3.12;
    # `coqui-tts` supports 3.10-3.14 and installs the same import name.
    from TTS.utils.synthesizer import Synthesizer as _CoquiSynthesizer
    _HAS_COQUI_BACKEND = True
except Exception:  # pragma: no cover - heavy optional dep, many failure modes
    _CoquiSynthesizer = None  # type: ignore[assignment]
    _HAS_COQUI_BACKEND = False

logger = get_logger(__name__)

WAV_MIME = "audio/wav"
MP3_MIME = "audio/mpeg"

# The only three of South Africa's eleven official languages with a real
# Microsoft neural locale. Voice IDs verified against the published
# edge-tts voice list (2026-08-25) rather than assumed — a wrong ID fails
# at request time with an unhelpful error.
_EDGE_VOICES = {
    # South African locales — the only three of the eleven Microsoft has.
    "English": "en-ZA-LukeNeural",
    "Afrikaans": "af-ZA-WillemNeural",
    "isiZulu": "zu-ZA-ThembaNeural",
    # Pan-African locales. Of the fourteen languages the optional detector
    # adds, these four are the only ones with a Microsoft neural voice —
    # confirmed by reading the published voice list on 2026-09-18 (322
    # voices, 11 relevant locales), not by assuming a locale exists because
    # the language is widely spoken. The other ten fall through to MMS.
    #
    # Swahili resolves to the Kenyan locale rather than Tanzanian; both
    # exist and neither is more correct for an app that does not ask which.
    "Amharic": "am-ET-MekdesNeural",
    "French": "fr-FR-HenriNeural",
    "Somali": "so-SO-UbaxNeural",
    "Swahili": "sw-KE-ZuriNeural",
}

# --- Coqui multilingual South African VITS ------------------------------
#
# The only model found that covers all eleven South African languages. It
# closes the exact gap edge-tts and MMS leave: seven languages with no
# voice anywhere.
#
# THREE THINGS TO KNOW BEFORE ENABLING IT, none of which are defects but
# all of which are the user's to weigh:
#
#  1. LICENCE. cc-by-nc-4.0 — non-commercial. This repository is MIT, so
#     the model is downloaded at runtime and never vendored; enabling it
#     makes the resulting *deployment* non-commercial. That is fine for an
#     academic project and not fine for a product, which is why this is
#     opt-in rather than a default.
#  2. GATED. `gated: auto` on Hugging Face, so access is granted
#     immediately on accepting the terms at the model page — but it must
#     be accepted once, with HF_TOKEN set, or every fetch returns 403.
#  3. UNVERIFIED. Its model card is an unfilled template: no training
#     details, no evaluation, no author. Nothing here can tell you how it
#     sounds. Listen to it before demoing it.
#
# Codes are the repo's own language ids, taken from its declared language
# tags. If the model is retrained with different ids, synthesis fails with
# a clear message rather than silently picking the wrong voice — see
# _synthesize_coqui.
COQUI_SA_REPO = "guymandude/South-African-TTS-11-Vits"
COQUI_SA_LANGUAGES = {
    "Afrikaans": "afr", "English": "eng", "isiNdebele": "nbl", "Sepedi": "nso",
    "Sesotho": "sot", "siSwati": "ssw", "Setswana": "tsn", "Xitsonga": "tso",
    "Tshivenda": "ven", "isiXhosa": "xho", "isiZulu": "zul",
}

# Off unless config.toml turns it on, for reasons 1-3 above. Set by
# bootstrap from Settings.coqui_sa_enabled, the same way local voices are
# registered, so nothing here reads config directly.
_COQUI_ENABLED = False


def enable_coqui_sa(enabled: bool) -> bool:
    """Turns the multilingual South African VITS model on or off.

    Returns what it actually ended up as: asking for it without
    `coqui-tts` installed leaves it off rather than promising a voice that
    cannot load.
    """
    global _COQUI_ENABLED
    _COQUI_ENABLED = bool(enabled) and _HAS_COQUI_BACKEND
    if enabled and not _HAS_COQUI_BACKEND:
        logger.warning(
            "enable_coqui_sa is set but coqui-tts is not installed — the seven "
            "languages it would cover stay text-only. pip install coqui-tts"
        )
    elif _COQUI_ENABLED:
        logger.info(f"Coqui South African VITS enabled ({COQUI_SA_REPO}).")
    return _COQUI_ENABLED


def has_coqui_backend() -> bool:
    """True when the Coqui model is installed AND switched on."""
    return _HAS_COQUI_BACKEND and _COQUI_ENABLED


def _coqui_speaks(language: str) -> bool:
    return has_coqui_backend() and language in COQUI_SA_LANGUAGES


# VITS degrades on long inputs (attention alignment drifts and memory
# grows with sequence length), so MMS input is chunked. Chunks longer than
# this with no sentence terminator get split on whitespace as a backstop —
# a 400-word bullet list with no full stop is a real thing an LLM returns.
_MAX_CHUNK_CHARS = 220


@dataclass(frozen=True)
class SpeechAudio:
    """Audio plus its MIME type. The type is carried rather than assumed
    because the two backends return different containers: MMS produces raw
    WAV, edge-tts produces MP3. Callers that hardcoded "audio/wav" played
    silence for the edge backend.
    """

    data: bytes
    mime: str


def has_mms_backend() -> bool:
    return _HAS_MMS_BACKEND


def has_edge_backend() -> bool:
    return _HAS_EDGE_BACKEND


def has_tts_backend() -> bool:
    """True if ANY speech synthesis is possible."""
    return _HAS_MMS_BACKEND or _HAS_EDGE_BACKEND


def has_stt_backend() -> bool:
    return _HAS_STT_BACKEND


# --- Voice coverage policy ------------------------------------------
#
# Four of the eleven have a real voice. What happens for the other seven is
# a design decision, not an accident, so it is written down here.
#
# TIER 1  native voice            English, isiZulu, Afrikaans (edge-tts);
#                                 Xitsonga (MMS, offline)
# TIER 2  related-language voice  only where the substitution is defensible
#                                 (see below). OFF by default.
# TIER 3  speak in English        with the substitution disclosed to the user.
#                                 OFF by default.
# TIER 4  text only               the default, and never silent about it.
#
# Tier 2 is where care is needed. Substituting a voice means applying one
# language's letter-to-sound rules to another's words, which produces
# mispronunciation rather than an accent — this project already rejected
# routing English through the Afrikaans model for exactly that reason.
#
# So only substitutions within the SAME closely-related family are offered,
# and a proposed table was cut from seven rows to three after checking:
#
#   isiXhosa, siSwati, isiNdebele -> isiZulu   KEPT. All Nguni; they share
#       orthography including the click letters c/q/x, so the isiZulu model
#       has seen these characters.
#
#   Sesotho, Setswana, Sepedi -> each other    DROPPED. All three return 404
#       on Hugging Face, so there is no voice anywhere in the Sotho-Tswana
#       family to substitute.
#
#   Tshivenda -> Xitsonga                      DROPPED. Venda is its own
#       branch of Bantu and Tsonga is Tswa-Ronga; they are not close. Venda
#       orthography also uses dental diacritics (ṱ ḓ ṋ ḽ) absent from Tsonga,
#       and a character-level VITS model mangles characters it never saw.
#
# Even the three kept substitutions are approximations. They are opt-in, and
# the UI says which voice was actually used.
# Locally trained VITS checkpoints, by language. These are voices this
# project trained itself on NCHLT rather than pulled from Hugging Face,
# and before this existed there was no way to load one: coverage only
# consulted edge-tts and the pretrained MMS catalogue, so a language with
# a trained voice sitting on disk still reported "text only".
#
# Paths come from config.toml ([speech.local_voices]) so a checkpoint can
# be dropped in without a code change. A path that does not exist is
# ignored rather than raising — the app must still start on a machine
# that has not synced the model files.
LOCAL_VOICE_MODELS: dict[str, str] = {}


def load_local_voices(paths: dict[str, str] | None) -> dict[str, str]:
    """Registers locally trained checkpoints, keeping only ones present on
    disk. Returns what was actually registered.
    """
    LOCAL_VOICE_MODELS.clear()
    for language, path in (paths or {}).items():
        if path and Path(path).exists():
            LOCAL_VOICE_MODELS[language] = path
        elif path:
            logger.warning(
                f"Local voice for {language} configured at {path} but not found — "
                f"{language} falls back to its next tier."
            )
    if LOCAL_VOICE_MODELS:
        logger.info(f"Local voices registered: {sorted(LOCAL_VOICE_MODELS)}")
    return dict(LOCAL_VOICE_MODELS)


# Substitutions within a closely-related family, offered only where a
# voice in that family actually exists. Nguni routes are static because
# isiZulu always has an edge-tts voice; Sotho-Tswana routes are NOT,
# because they depend on a locally trained checkpoint that may or may not
# be installed — see related_language_voices().
_NGUNI_ROUTES = {
    "isiXhosa": "isiZulu",
    "siSwati": "isiZulu",
    "isiNdebele": "isiZulu",
}

# Sotho-Tswana. Previously dropped outright: all three 404 on Hugging
# Face, so there was no voice in the family to substitute from. A locally
# trained Sepedi checkpoint removes that blocker, and these two routes
# become available — but ONLY while such a checkpoint is registered.
#
# The substitution is defensible within the family: the three are closely
# related and share the Latin orthography a character-level model reads.
# It is still an approximation, and orthographic conventions differ
# (Sepedi's š has no Sesotho counterpart), so it stays opt-in and
# disclosed like every other Tier 2 route.
_SOTHO_TSWANA_ROUTES = {
    "Sesotho": "Sepedi",
    "Setswana": "Sepedi",
}


def related_language_voices() -> dict[str, str]:
    """Tier 2 routes available right now.

    Computed rather than constant because a route is only honest if the
    target language can actually be spoken on this machine. Offering
    "Sesotho, spoken with the Sepedi voice" when no Sepedi voice is
    installed promises audio that never arrives.
    """
    routes = dict(_NGUNI_ROUTES)
    if "Sepedi" in LOCAL_VOICE_MODELS:
        routes.update(_SOTHO_TSWANA_ROUTES)
    return routes


# Retained as a module-level name for callers that import it directly.
# Prefer related_language_voices(), which reflects installed checkpoints.
RELATED_LANGUAGE_VOICES = _NGUNI_ROUTES


def voice_coverage(
    mms_codes: dict[str, str] | None = None,
    languages: list[str] | None = None,
) -> dict[str, str]:
    """Which tier each language currently falls into, for display.

    Reports capability, not attempts: it does not download anything, so it
    is safe to call on every render.

    Derived by ASKING resolve_voice_language, not by re-deriving the tiers
    from the same tables a second time. The two used to be independent
    walks over overlapping data, and they drifted: this function consulted
    LOCAL_VOICE_MODELS and related_language_voices() while the resolver
    consulted neither. Registering a locally trained Sepedi checkpoint
    therefore made the sidebar report "Sepedi — native (locally trained)"
    and "Sesotho — related (Sepedi)" while speak() returned no audio for
    either, which is the worst version of this bug: the interface made a
    claim about coverage that the audio path could not honour.

    A tier reported here is now one synthesis will honour, by
    construction rather than by agreement.
    """
    from bao.core.config import LABELS

    # Defaults to the eleven South African languages. The caller passes a
    # wider list when the pan-African detector is on, because those twelve
    # extra voices are only reachable when their languages can be detected
    # in the first place — reporting coverage for a language the app will
    # never identify would be noise.
    coverage = {}
    for language in languages or LABELS:
        # Fallbacks off — anything returned is a voice for the language itself.
        spoken, _ = resolve_voice_language(language, mms_codes)
        if spoken is not None:
            if language in LOCAL_VOICE_MODELS:
                coverage[language] = "native (locally trained)"
            elif select_backend(language, "auto") == "coqui":
                coverage[language] = "native (SA VITS)"
            else:
                coverage[language] = "native"
            continue
        # Fallbacks on — what someone who opts in would actually hear.
        spoken, _ = resolve_voice_language(language, mms_codes, allow_related=True)
        coverage[language] = f"related ({spoken})" if spoken else "text only"
    return coverage


# Codes with a real facebook/mms-tts-<code> repo, verified against Hugging
# Face with an authenticated request rather than inferred from the language
# being widely spoken. Listed explicitly so coverage can be reported
# without a network call.
#
# South African (probed 2026-08-30): only tso and eng resolve. xho, sot,
# tsn, nso, ven, ssw, nbl and afr all 404 — MMS covers ~1100 languages but
# not those.
#
# Pan-African (probed 2026-09-18): twelve of the fourteen resolve, which is
# a far better hit rate and the reason those languages go from silent to
# spoken without a new dependency. ibo and lin 404 under every code tried,
# so Igbo and Lingala stay text-only and say so.
_KNOWN_MMS_VOICES = {
    "tso", "eng",
    "amh", "fra", "hau", "lug", "orm", "pcm", "run", "sna", "som", "swh", "tir", "yor",
}


def select_backend(target_language: str, preference: str = "auto") -> str | None:
    """Decides which backend synthesizes this language. Returns "edge",
    "mms", or None if neither can.

    Split out as a pure function so the routing rules are testable without
    a network call or a model download — the audio itself isn't testable
    here, but "does English go to edge when edge is installed" is.

    preference: "auto" prefers edge where a real SA voice exists and falls
    back to MMS; "edge" and "mms" force one backend and return None rather
    than silently using the other.
    """
    if preference == "coqui":
        return "coqui" if _coqui_speaks(target_language) else None

    if preference == "mms":
        return "mms" if _HAS_MMS_BACKEND else None

    if preference == "edge":
        return "edge" if (_HAS_EDGE_BACKEND and target_language in _EDGE_VOICES) else None

    # "auto" — Tier 1 only. Tiers 2 and 3 are applied by the caller via
    # resolve_voice_language(), because they change WHICH language is
    # spoken, not which backend speaks it, and that has to be visible.
    #
    # A checkpoint trained on this language specifically outranks a generic
    # voice, so it is checked before edge — otherwise training an isiZulu
    # model and registering it would change nothing, because edge already
    # claims isiZulu. Local checkpoints are VITS, so they run on the same
    # engine as MMS and report as "mms"; the backend is the same, only the
    # weights differ.
    if _HAS_MMS_BACKEND and target_language in LOCAL_VOICE_MODELS:
        return "mms"
    if _HAS_EDGE_BACKEND and target_language in _EDGE_VOICES:
        return "edge"
    # Before the generic MMS fall-through but after edge: the Coqui model
    # is a real voice for the language, where falling through to MMS means
    # whatever code the table happens to hold. It does NOT outrank edge —
    # Microsoft's neural voices are better than a community NCHLT model for
    # the three languages both cover, and this model's job is the seven
    # neither covers.
    if _coqui_speaks(target_language) and not _has_native_mms(target_language, None):
        return "coqui"
    if _HAS_MMS_BACKEND:
        return "mms"
    return None


def _has_native_mms(target_language: str, mms_codes: dict[str, str] | None) -> bool:
    """True when MMS has a verified voice for this language specifically,
    as opposed to the "eng" default `synthesize_speech` falls back to.
    """
    codes = mms_codes or DEFAULT_MMS_CODES
    return _HAS_MMS_BACKEND and codes.get(target_language) in _KNOWN_MMS_VOICES


def resolve_voice_language(
    target_language: str,
    mms_codes: dict[str, str] | None = None,
    allow_related: bool = False,
    allow_english: bool = False,
) -> tuple[str | None, str | None]:
    """Decides which language will actually be SPOKEN, and why.

    Returns (language_to_speak, note). `note` is None for a native voice and
    a short user-facing sentence otherwise — the substitution is never
    silent, because a listener who is told "spoken with the isiZulu voice"
    can judge it, and one who isn't isn't just hears bad isiXhosa.

    Returns (None, reason) when nothing suitable exists, which is Tier 4:
    text only, stated plainly.
    """
    codes = mms_codes or DEFAULT_MMS_CODES

    # Tier 1 — a real voice for this language. A locally trained checkpoint
    # comes first: it was trained on this language rather than merely
    # covering it. Gated on the MMS backend because a local checkpoint is a
    # VITS model and needs the same torch/transformers stack to run —
    # promising a voice that cannot be loaded is the failure this whole
    # tier system exists to avoid.
    if _HAS_MMS_BACKEND and target_language in LOCAL_VOICE_MODELS:
        return target_language, None
    if _HAS_EDGE_BACKEND and target_language in _EDGE_VOICES:
        return target_language, None
    if _HAS_MMS_BACKEND and codes.get(target_language) in _KNOWN_MMS_VOICES:
        return target_language, None
    # Still Tier 1: a voice in the language the user is actually reading,
    # which is the whole point of the tier. This is what turns the related-
    # language substitution below from the first fallback into a last
    # resort — borrowing isiZulu for isiXhosa is only ever an approximation,
    # and an isiXhosa voice beats it whenever one exists.
    if _coqui_speaks(target_language):
        return target_language, None

    # Tier 2 — a closely-related language, opt-in. Read from the FUNCTION,
    # not the static table: the Sotho-Tswana routes exist only while a
    # Sepedi checkpoint is registered, and the static table never carried
    # them. Using it here is what made those routes unreachable no matter
    # what was installed.
    routes = related_language_voices()
    if allow_related and target_language in routes:
        related = routes[target_language]
        speakable, _ = resolve_voice_language(related, codes)
        if speakable:
            return speakable, (
                f"No {target_language} voice exists yet — spoken with the "
                f"{related} voice, which is closely related but not the same."
            )

    # Tier 3 — English, opt-in.
    if allow_english and _HAS_EDGE_BACKEND and "English" in _EDGE_VOICES:
        return "English", (
            f"No {target_language} voice exists yet — the reply is spoken in English."
        )

    # Tier 4.
    return None, f"No voice is available for {target_language}; showing text only."


def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    if waveform.dtype.kind == "f":
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform = (waveform * 32767).astype(np.int16)
    elif waveform.dtype.kind in {"u", "i"}:
        waveform = waveform.astype(np.int16)
    return waveform


def clean_text_for_speech(text: str) -> str:
    """Strips markdown that would otherwise be read aloud as punctuation
    noise ("star star Cloud star star").

    Deliberately does NOT strip the apostrophe: it is a letter-level part
    of the orthography in several of these languages (Xitsonga
    "ematshan'weni", isiZulu "wam'"), and removing it changes the word
    rather than tidying it.
    """
    # Ordered longest-first so "**" is removed before "*" leaves a stray.
    for token in ("**", "__", "```", "*", "_", "#", "[", "]", "(", ")", "`",
                  '"', "|", ">", "~~"):
        text = text.replace(token, "")

    # Bare list bullets read as "dash" or "bullet" in some voices.
    text = re.sub(r"^\s*[-\u2022\u2013]\s+", "", text, flags=re.MULTILINE)
    # A colon introduces a list far more often than it ends a clause here;
    # turning it into a sentence break stops the synthesizer running the
    # heading into the first item.
    text = re.sub(r":\s*", ". ", text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Splits text into synthesis-sized chunks, keeping terminal
    punctuation.

    Punctuation is kept because both backends use it for prosody — a
    stripped chunk is read as a flat statement, so a whole paragraph comes
    out monotone. (The earlier version split on `[.\\n!?]+` and discarded
    the match, which is where that flatness came from.)
    """
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return []

    # Keep the terminator with the sentence it ends.
    raw_chunks = re.split(r"(?<=[.!?])\s+|\n+", cleaned)

    sentences: list[str] = []
    for chunk in raw_chunks:
        collapsed = " ".join(chunk.split()).strip()
        if len(collapsed) <= 1:
            continue
        sentences.extend(_split_long_chunk(collapsed))
    return sentences


def _split_long_chunk(chunk: str) -> list[str]:
    """Backstop for text with no sentence terminators at all. Splits on
    word boundaries so a word is never cut in half.
    """
    if len(chunk) <= _MAX_CHUNK_CHARS:
        return [chunk]

    pieces: list[str] = []
    current = ""
    for word in chunk.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > _MAX_CHUNK_CHARS and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


# --- MMS (offline) -----------------------------------------------------


# Successful loads AND failures are both remembered. `functools.cache` only
# caches successful returns — a raising call is never stored, so a language
# whose model can't be fetched re-attempted the download on EVERY message.
# That turned one broken voice into a permanent per-turn network stall plus
# a repeating traceback in the log, which is why "voice doesn't work" and
# "messages are slow" were the same bug.
_TTS_MODELS: dict[str, tuple] = {}
_TTS_FAILURES: dict[str, str] = {}


def load_tts_model(mms_code: str, local_path: str | None = None):
    """Loads (and memoizes) a VITS voice. Raises on failure, but only the
    first attempt actually tries — subsequent calls re-raise the recorded
    reason immediately.

    `local_path` points at a checkpoint this project trained itself. A
    local directory and a Hugging Face repo id are interchangeable to
    `from_pretrained`, so a locally trained voice needs no separate
    synthesis path — only a different source. That is the whole reason
    the feature is a few lines rather than a second backend.

    The cache is keyed on the SOURCE, not the language code, so a local
    checkpoint and the pretrained repo for the same code can coexist and
    a failure against one is never remembered against the other.
    """
    key = local_path or mms_code
    if key in _TTS_MODELS:
        return _TTS_MODELS[key]
    if key in _TTS_FAILURES:
        raise SpeechError(_TTS_FAILURES[key])

    if not _HAS_MMS_BACKEND:
        raise ImportError("torch and transformers are required for VITS speech synthesis.")

    source = local_path or f"facebook/mms-tts-{mms_code}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = VitsModel.from_pretrained(source)
        model.eval()
    except Exception as e:
        # Remembered so it is attempted once per process, not once per turn.
        _TTS_FAILURES[key] = f"{source} could not be loaded: {e}"
        raise

    _TTS_MODELS[key] = (tokenizer, model)
    logger.info(f"Loaded VITS voice from '{source}'.")
    return tokenizer, model


def reset_tts_cache() -> None:
    """Clears both caches, including remembered failures — so a language
    that failed because the network was down can be retried after it comes
    back, without restarting the app.
    """
    _TTS_MODELS.clear()
    _TTS_FAILURES.clear()


def tts_availability() -> dict[str, str]:
    """What this process currently knows: "loaded", or the failure reason.
    Drives the honest sidebar caption instead of claiming every language
    has a voice.
    """
    status = {code: "loaded" for code in _TTS_MODELS}
    status.update(_TTS_FAILURES)
    return status


def _synthesize_mms(
    sentences: list[str],
    mms_code: str,
    speaking_rate: float | None,
    noise_scale: float | None,
    local_path: str | None = None,
) -> SpeechAudio | None:
    tokenizer, model = load_tts_model(mms_code, local_path=local_path)

    # VitsModel.forward reads these off the instance, so setting them here
    # is the supported way to control pacing. speaking_rate < 1.0 is
    # SLOWER (transformers uses length_scale = 1 / speaking_rate).
    if speaking_rate is not None:
        model.speaking_rate = speaking_rate
    if noise_scale is not None:
        model.noise_scale = noise_scale

    waveforms: list[np.ndarray] = []
    # Built once rather than per sentence — it never varies within a call.
    silence_gap = np.zeros(int(model.config.sampling_rate * 0.25), dtype=np.int16)

    for sentence in sentences:
        inputs = tokenizer(sentence.lower(), return_tensors="pt")
        if inputs.input_ids.shape[-1] == 0:
            continue
        # inference_mode is strictly faster than no_grad: it also skips
        # version-counter bookkeeping on the tensors. Safe here because
        # nothing downstream ever backpropagates through this output.
        with torch.inference_mode():
            output = model(**inputs).waveform
        waveforms.append(_normalize_waveform(output.squeeze().cpu().numpy()))
        waveforms.append(silence_gap)

    if not waveforms:
        return None

    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=model.config.sampling_rate, data=np.concatenate(waveforms))
    buffer.seek(0)
    return SpeechAudio(data=buffer.read(), mime=WAV_MIME)


# --- Coqui multilingual SA VITS (offline after first download) ---------

# Same memoisation contract as the MMS loader: successes AND failures are
# both remembered, so a model that cannot be fetched is attempted once per
# process rather than once per message.
_COQUI_SYNTH = None
_COQUI_FAILURE: str | None = None


def load_coqui_synthesizer():
    """Downloads and memoises the multilingual South African VITS model.

    Raises SpeechError on failure, and re-raises the recorded reason
    immediately on later calls.
    """
    global _COQUI_SYNTH, _COQUI_FAILURE
    if _COQUI_SYNTH is not None:
        return _COQUI_SYNTH
    if _COQUI_FAILURE is not None:
        raise SpeechError(_COQUI_FAILURE)
    if not _HAS_COQUI_BACKEND:
        raise ImportError("coqui-tts is required for the South African VITS model.")

    try:
        from huggingface_hub import hf_hub_download

        # Fetched file by file rather than with snapshot_download so a
        # missing one names itself. The checkpoint is ~150 MB; the rest are
        # small.
        paths = {
            name: hf_hub_download(COQUI_SA_REPO, name)
            for name in ("config.json", "vits_11_ZA_model.pth",
                         "model_speakers.pth", "language_ids.json")
        }
        _COQUI_SYNTH = _CoquiSynthesizer(
            tts_checkpoint=paths["vits_11_ZA_model.pth"],
            tts_config_path=paths["config.json"],
            tts_speakers_file=paths["model_speakers.pth"],
            tts_languages_file=paths["language_ids.json"],
            use_cuda=False,
        )
    except Exception as e:
        _COQUI_FAILURE = _explain_coqui_failure(e)
        raise SpeechError(_COQUI_FAILURE) from e

    logger.info(f"Loaded Coqui South African VITS from {COQUI_SA_REPO}.")
    return _COQUI_SYNTH


def _explain_coqui_failure(error: Exception) -> str:
    """The two likely causes have completely different fixes, and the raw
    exception names neither.
    """
    detail = str(error).strip() or type(error).__name__
    lowered = detail.lower()
    if "403" in detail or "gated" in lowered or "awaiting a review" in lowered:
        return (
            f"Access to {COQUI_SA_REPO} has not been granted yet. It is a gated "
            "repo with automatic approval: open "
            f"https://huggingface.co/{COQUI_SA_REPO}, accept the terms once, and "
            "make sure HF_TOKEN is set in .env. This is not a download failure."
        )
    if "401" in detail or "unauthorized" in lowered:
        return (
            f"Hugging Face refused the request for {COQUI_SA_REPO} (401). Set a "
            "HF_TOKEN in .env — the model is gated and cannot be fetched anonymously."
        )
    if any(k in lowered for k in ("connection", "timeout", "resolve", "network", "ssl")):
        return (
            f"Could not download {COQUI_SA_REPO} (~150 MB on first use). "
            "Check the network, then retry."
        )
    return f"The South African VITS model could not be loaded: {detail[:160]}"


def reset_coqui_cache() -> None:
    """Clears the model and any remembered failure, so a fetch that failed
    because the gate had not been accepted can be retried once it has,
    without restarting the app.
    """
    global _COQUI_SYNTH, _COQUI_FAILURE
    _COQUI_SYNTH = None
    _COQUI_FAILURE = None


def _synthesize_coqui(text: str, target_language: str) -> SpeechAudio | None:
    synthesizer = load_coqui_synthesizer()
    language_id = COQUI_SA_LANGUAGES[target_language]

    # The model is multi-speaker; a language id alone does not identify a
    # voice. Speakers are read off the loaded model rather than hard-coded,
    # because the names are the checkpoint's own and a wrong one raises
    # deep inside Coqui with an unhelpful message.
    speaker = None
    manager = getattr(synthesizer, "tts_speakers_file", None) and getattr(
        synthesizer.tts_model, "speaker_manager", None)
    if manager is not None and getattr(manager, "speaker_names", None):
        speaker = manager.speaker_names[0]

    known = getattr(getattr(synthesizer.tts_model, "language_manager", None),
                    "language_names", None)
    if known and language_id not in known:
        raise SpeechError(
            f"{COQUI_SA_REPO} has no language id {language_id!r} for "
            f"{target_language} (it offers {sorted(known)}). The checkpoint's "
            "language ids differ from the ones this app expects — update "
            "COQUI_SA_LANGUAGES."
        )

    waveform = synthesizer.tts(text, speaker_name=speaker, language_name=language_id)
    array = _normalize_waveform(np.asarray(waveform, dtype=np.float32))
    if array.size == 0:
        return None

    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=synthesizer.output_sample_rate, data=array)
    buffer.seek(0)
    return SpeechAudio(data=buffer.read(), mime=WAV_MIME)


# --- edge-tts (online, real SA voices) ---------------------------------


def _rate_percent(speaking_rate: float | None) -> str:
    """Converts the MMS-style multiplier in config.toml into the
    percentage string edge-tts expects, so one config knob drives both
    backends instead of two that can drift apart.
    """
    if speaking_rate is None:
        return "+0%"
    delta = round((speaking_rate - 1.0) * 100)
    return f"{delta:+d}%"


def _run_coroutine_blocking(make_coroutine):
    """Runs an async call from sync code, on its own loop in its own
    thread.

    Streamlit reruns scripts inside worker threads and other callers may
    already be inside a loop, so a bare `asyncio.run()` here raises
    "cannot be called from a running event loop" in some contexts and
    works in others. Owning the loop removes that whole class of
    environment-dependent failure.
    """
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(make_coroutine())
        except Exception as e:  # re-raised on the calling thread below
            box["error"] = e

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in box:
        raise box["error"]
    return box.get("value")


def _synthesize_edge(text: str, voice: str, speaking_rate: float | None) -> SpeechAudio | None:
    rate = _rate_percent(speaking_rate)

    async def stream() -> bytes:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        return bytes(audio)

    data = _run_coroutine_blocking(stream)
    if not data:
        return None
    logger.info(f"Synthesized speech with edge-tts voice '{voice}' (rate {rate}).")
    return SpeechAudio(data=data, mime=MP3_MIME)


# --- Public entry point ------------------------------------------------


def synthesize_speech(
    text: str,
    target_language: str = "English",
    mms_codes: dict[str, str] | None = None,
    speaking_rate: float | None = None,
    noise_scale: float | None = None,
    backend: str = "auto",
    on_error: Callable[[str], None] | None = None,
) -> SpeechAudio | None:
    """Synthesizes `text` in `target_language`. Returns None rather than
    raising: speech is an enhancement, and a failed voice must never take
    down a turn that has a perfectly good text answer.

    `on_error` receives a short human-readable reason when synthesis
    fails. Without it a failure was invisible — no audio player appeared
    and nothing said why, so "some languages have no voice" was
    indistinguishable from "the model is still downloading", "that voice
    code doesn't exist", and "there's no network". Each MMS language is a
    separate ~145 MB download on first use, so the first request in a new
    language routinely looks like a broken feature when it is really a
    download in progress or a failed one.
    """
    if not text.strip():
        return None

    chosen = select_backend(target_language, backend)
    if chosen is None:
        reason = (
            f"No speech backend available for {target_language}. "
            "Install requirements-speech.txt for offline MMS voices."
        )
        logger.info(reason)
        if on_error:
            on_error(reason)
        return None

    try:
        if chosen == "coqui":
            spoken = clean_text_for_speech(text)
            if not spoken:
                return None
            return _synthesize_coqui(spoken, target_language)

        if chosen == "edge":
            # Clean first. This was missing: only the MMS branch cleaned its
            # input (via split_into_sentences), so edge-tts received raw
            # markdown and read it aloud — a reply containing
            # "**Differential Calculus**" was spoken with the asterisks.
            #
            # No sentence splitting here: edge-tts handles long text server
            # side, and it is the punctuation-aware chunking that MMS needs
            # for its own reasons, not a cleaning step.
            spoken = clean_text_for_speech(text)
            if not spoken:
                return None
            return _synthesize_edge(spoken, _EDGE_VOICES[target_language], speaking_rate)

        codes = mms_codes or DEFAULT_MMS_CODES
        mms_code = codes.get(target_language, "eng")
        # A checkpoint registered for this language replaces the pretrained
        # repo as the source. Nothing else about the path changes: same
        # engine, same chunking, same pacing controls.
        local_path = LOCAL_VOICE_MODELS.get(target_language)
        sentences = split_into_sentences(text)
        if not sentences:
            return None
        return _synthesize_mms(
            sentences, mms_code, speaking_rate, noise_scale, local_path=local_path
        )
    except Exception as e:
        # Deliberately a one-line warning, not logger.exception: the
        # common cause is "edge-tts is online and there's no network,"
        # which is an expected condition on this project's target
        # deployments. A full traceback per turn would bury the actual
        # conversation in the console during a live demo.
        logger.warning(f"Speech synthesis failed via '{chosen}' for {target_language}: {e}")

        # An edge failure is usually the network. Retry offline — but on a
        # model that actually speaks this language where one exists, since
        # falling straight to MMS means the "eng" default reading, say,
        # isiXhosa words.
        if chosen == "edge" and _coqui_speaks(target_language):
            logger.info("Retrying synthesis on the offline Coqui model.")
            return synthesize_speech(
                text, target_language, mms_codes, speaking_rate, noise_scale,
                backend="coqui", on_error=on_error,
            )
        if chosen == "edge" and _HAS_MMS_BACKEND:
            logger.info("Retrying synthesis on the offline MMS backend.")
            return synthesize_speech(
                text, target_language, mms_codes, speaking_rate, noise_scale,
                backend="mms", on_error=on_error,
            )

        if on_error:
            on_error(_explain_failure(chosen, target_language, mms_codes, e))
        return None


def _explain_failure(chosen: str, target_language: str, mms_codes: dict | None, error: Exception) -> str:
    """Turns an exception into something worth showing a user.

    The raw message is usually a stack-trace fragment or an HTTP status,
    which tells a presenter nothing. The three real causes are worth
    naming explicitly, because they have different fixes.
    """
    detail = str(error).strip() or type(error).__name__

    if chosen == "edge":
        return f"Online voice for {target_language} failed (network?): {detail[:120]}"

    if chosen == "coqui":
        # Already a written explanation by the time it reaches here; the
        # MMS advice below is about a different model entirely.
        return detail[:400]

    code = (mms_codes or DEFAULT_MMS_CODES).get(target_language, "eng")
    lowered = detail.lower()

    # A locally trained checkpoint never touches the hub, so none of the
    # 401/404/rate-limit advice below applies to it. Telling someone to set
    # a HF_TOKEN when the real problem is a half-synced directory on their
    # own disk sends them to fix the wrong thing.
    local_path = LOCAL_VOICE_MODELS.get(target_language)
    if local_path:
        return (
            f"The locally trained {target_language} voice at {local_path} could not "
            f"be loaded: {detail[:120]}. Check the checkpoint directory is complete, "
            f"or unregister it from [speech.local_voices] in config.toml to fall back."
        )

    # transformers reports EVERY hub failure with the same sentence — "is not
    # a local folder and is not a valid model identifier" — regardless of
    # whether the repo is absent, gated, or the request was simply refused.
    # An earlier version of this function read that sentence as proof of
    # absence and told users the model "is not a published model", which was
    # wrong: a live probe showed nine repos returning HTTP 401 (refused)
    # while a tenth returned 200 and loaded fine. 401 is not 404.
    #
    # So the status code is checked first, and only a genuine 404 is reported
    # as absence. Anything else is reported as what it is: a refused request,
    # which authentication usually fixes.
    if "401" in detail or "unauthorized" in lowered or "gated" in lowered:
        return (
            f"Hugging Face refused the request for facebook/mms-tts-{code} (401). "
            "That is an authentication problem, not a missing model — set a "
            "HF_TOKEN environment variable (free token from "
            "huggingface.co/settings/tokens) and try again."
        )
    if "429" in detail or "rate limit" in lowered:
        return (
            f"Rate-limited while fetching facebook/mms-tts-{code}. Set a HF_TOKEN "
            "for higher limits, or wait and retry."
        )
    if "404" in detail or "repositorynotfound" in lowered:
        return (
            f"No offline voice for {target_language}: facebook/mms-tts-{code} "
            "returned 404. MMS covers ~1100 languages but not every one."
        )
    if "not a valid model identifier" in lowered or "is not a local folder" in lowered:
        # Ambiguous by construction — say so rather than guessing.
        return (
            f"facebook/mms-tts-{code} could not be fetched. transformers reports "
            "absent, gated and refused requests identically, so this may be "
            "authentication rather than a missing model — try setting HF_TOKEN."
        )
    if any(k in lowered for k in ("connection", "timeout", "resolve", "network", "ssl")):
        return (
            f"Could not download the {target_language} voice (facebook/mms-tts-{code}, "
            "~145 MB on first use). Check the network, then retry."
        )
    return f"Voice for {target_language} (mms-tts-{code}) failed: {detail[:120]}"


def transcribe_audio_bytes(
    audio_bytes: bytes, target_language: str = "English", stt_codes: dict | None = None
) -> str:
    if not _HAS_STT_BACKEND:
        raise SpeechError("Speech-to-text is unavailable (SpeechRecognition not installed).")

    codes = stt_codes or DEFAULT_STT_CODES
    language_code = codes.get(target_language, "en-ZA")

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language=language_code)
    except sr.UnknownValueError as e:
        raise SpeechError("Could not understand the audio.") from e
    except Exception as e:
        logger.exception("Speech transcription failed")
        raise SpeechError(str(e)) from e
