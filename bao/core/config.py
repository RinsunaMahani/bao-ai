"""
Bao AI - Configuration Loader & Constants

Single source of truth for settings. Every other module imports configuration
values from here instead of declaring independent defaults.
"""

import logging
import os

# Standard library from Python 3.11, which is this project's floor. It
# used to fall back to tomli, which nothing installed, so on 3.10
# config.toml was silently ignored; the floor was raised instead of the
# fallback being patched - see pyproject.toml.
import tomllib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Module-Level Constants ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSISTANT_NAME = "Bao"
ASSISTANT_LOGO_PATH = os.path.join(BASE_DIR, "docs", "assets", "logo.jpg")
# Matches config.toml's choice, so the app behaves the same if config.toml
# fails to load. It was gemini-3.6-flash — the model whose free tier allows
# 20 requests a DAY — so any failure to read the config quietly moved the
# app onto its most restrictive quota. See config.toml for how 3.5-flash was
# chosen.
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash"

LABELS = [
    "English", "isiZulu", "isiXhosa", "Afrikaans", "Sesotho",
    "Setswana", "Sepedi", "Xitsonga", "Tshivenda", "siSwati", "isiNdebele",
]

# Offline MMS voice codes. Of these only "eng" and "tso" resolve on Hugging
# Face; the rest 404 and are kept as the mapping each language WOULD use,
# gated at synthesis time by speech._KNOWN_MMS_VOICES.
#
# English is "eng", not "afr". It was "afr" to give English a local accent
# by routing it through Afrikaans letter-to-sound rules — but that repo
# does not exist either, so the mapping silently removed English from the
# offline path as well. config.toml was corrected and this default was
# not, which left every caller that does not pass an explicit table (and
# any install without a config.toml) trying to fetch mms-tts-afr for
# English and getting no audio at all.
DEFAULT_MMS_CODES = {
    "Afrikaans": "afr", "English": "eng", "isiNdebele": "nbl", "isiXhosa": "xho",
    "isiZulu": "zul", "Sepedi": "nso", "Sesotho": "sot", "Setswana": "tsn",
    "siSwati": "ssw", "Tshivenda": "ven", "Xitsonga": "tso",
}
# The 14 pan-African languages the optional sklearn detector adds. Kept
# separate from LABELS, which is the eleven South African languages this
# app is built around and which the knowledge base is written in.
PAN_AFRICAN_LABELS = [
    "Amharic", "French", "Hausa", "Igbo", "Lingala", "Luganda", "Oromo",
    "Nigerian Pidgin", "Kirundi", "Shona", "Somali", "Swahili", "Tigrinya", "Yoruba",
]

# Offline MMS voices for those languages. Every code here was probed
# against Hugging Face on 2026-09-18 with an authenticated request and
# returned 200 — the same standard the South African table is held to,
# because an unverified code fails at synthesis time with an error that
# reads like a bug rather than a missing voice.
#
# Igbo and Lingala are absent on purpose: mms-tts-ibo and mms-tts-lin both
# 404, as do the alternative codes tried (ibb, ibo_Latn, ln, lingala). They
# are detected and answered in text, and honestly report no voice.
#
# Oromo is "orm", not the "gaz" its ISO 639-3 macrolanguage member would
# suggest — gaz 404s and orm resolves. Probed, not assumed.
DEFAULT_PAN_AFRICAN_MMS_CODES = {
    "Amharic": "amh", "French": "fra", "Hausa": "hau", "Luganda": "lug",
    "Oromo": "orm", "Nigerian Pidgin": "pcm", "Kirundi": "run", "Shona": "sna",
    "Somali": "som", "Swahili": "swh", "Tigrinya": "tir", "Yoruba": "yor",
}

DEFAULT_STT_CODES = {
    "Afrikaans": "af-ZA", "English": "en-ZA", "isiNdebele": "nr-ZA", "isiXhosa": "xh-ZA",
    "isiZulu": "zu-ZA", "Sepedi": "nso-ZA", "Sesotho": "st-ZA", "Setswana": "tn-ZA",
    "siSwati": "ss-ZA", "Tshivenda": "ve-ZA", "Xitsonga": "ts-ZA",
}


class Settings:
    """Loaded, resolved application settings used across modules."""

    def __init__(self, config_path: str | None = None):
        self._raw: dict = {}
        self.config_path = config_path or self._default_config_path()
        self._load()

    def _default_config_path(self) -> str:
        path = os.path.join(BASE_DIR, "config.toml")
        return path if os.path.exists(path) else "config.toml"

    def _load(self) -> None:
        if not os.path.exists(self.config_path):
            logging.warning(f"Configuration file not found at '{self.config_path}'. Using defaults.")
            return
        try:
            with open(self.config_path, "rb") as f:
                self._raw = tomllib.load(f)
            logging.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logging.error(f"Error parsing {self.config_path}: {e}")

    @property
    def gemini_model(self) -> str:
        return self._raw.get("model", {}).get("gemini_model", GEMINI_MODEL_DEFAULT)

    @property
    def thinking_level(self) -> str:
        """How much the model reasons before it starts answering.

        Gemini 3.x models think first and emit nothing until they finish,
        so this sets how long a user watches a spinner — streaming cannot
        help, because there is nothing to stream yet. Measured on
        gemini-3.5-flash with "explain calculus in xitsonga":

            default   11.9s to the first word, 13.0s total
            MINIMAL    6.6s to the first word, 12.8s total
            LOW       15.0s to the first word, 19.2s total

        All three answered in correct Xitsonga at 100% detection
        confidence, so the reasoning was not buying accuracy on this kind
        of question. MINIMAL is the default because this is an assistant
        answering everyday questions in eleven languages, not a reasoning
        benchmark, and halving the wait in front of an audience is worth
        more than thinking tokens nobody sees.

        Raise it to LOW, MEDIUM or HIGH for genuinely hard questions, or
        set "default" to let the model decide.
        """
        return str(self._raw.get("model", {}).get("thinking_level", "MINIMAL")).strip()

    @property
    def generation_max_attempts(self) -> int:
        """How many times to try a generation call before giving up.

        Gemini returns 503 UNAVAILABLE ("this model is currently
        experiencing high demand") under load, and 429 when rate-limited.
        Both are the provider saying "not now", not "not ever" — the same
        request typically succeeds immediately afterwards. Without a retry
        a single spike costs the user their whole turn.

        3 rather than more because someone is watching a spinner: the
        backoff is a fraction of a second, so the worst case adds about a
        second, while a longer ladder would make a genuine outage feel like
        a hang. 1 disables retrying.
        """
        return int(self._raw.get("model", {}).get("generation_max_attempts", 3))

    @property
    def similarity_threshold(self) -> float:
        return self._raw.get("retrieval", {}).get("default_similarity_threshold", 0.25)

    @property
    def min_query_coverage(self) -> float:
        """Minimum share of a question (IDF-weighted) the knowledge base
        must be able to represent before its answer counts as verified.
        See KnowledgeRetriever.query_coverage. Lower = more KB hits and
        more false positives; higher = fewer of both.
        """
        return self._raw.get("retrieval", {}).get("min_query_coverage", 0.7)

    @property
    def max_query_length(self) -> int:
        return self._raw.get("application", {}).get("max_query_length", 500)

    @property
    def assistant_avatar(self) -> str:
        return self._raw.get("application", {}).get("assistant_avatar", ASSISTANT_LOGO_PATH)

    @property
    def mms_codes(self) -> dict:
        """Voice codes for every language the app can detect, South African
        and pan-African, with config.toml taking precedence.

        MERGED rather than replaced. config.toml's table lists the eleven
        South African languages and nothing else, so a plain `.get(...,
        DEFAULT)` meant the twelve pan-African voices were invisible the
        moment that file existed — which is always. Merging also means an
        existing config.toml gains them without being edited, and an
        explicit entry still wins over the default it shadows.
        """
        configured = self._raw.get("languages", {}).get("mms_codes", {})
        return {**DEFAULT_MMS_CODES, **DEFAULT_PAN_AFRICAN_MMS_CODES, **configured}

    @property
    def stt_codes(self) -> dict:
        return self._raw.get("languages", {}).get("stt_codes", DEFAULT_STT_CODES)

    @property
    def tts_speaking_rate(self) -> float:
        return self._raw.get("speech", {}).get("speaking_rate", 0.85)

    @property
    def tts_noise_scale(self) -> float:
        return self._raw.get("speech", {}).get("noise_scale", 0.35)

    @property
    def pan_african_enabled(self) -> bool:
        """OFF by default, deliberately.

        Enabling this changes the detection path for EVERY message, not
        only the fourteen new languages — detection feeds greeting lookup,
        retrieval language preference and voice selection. Shipping it
        switched off means the code and the model are present and testable
        while behaviour is bit-for-bit unchanged until someone turns it on
        and checks it.
        """
        return self._raw.get("model", {}).get("enable_pan_african", False)

    @property
    def pan_african_model_path(self) -> str:
        default = os.path.join(BASE_DIR, "models", "language_detector_pan_african.joblib")
        return self._raw.get("model", {}).get("pan_african_path", default)

    @property
    def tts_backend(self) -> str:
        """"auto" | "edge" | "mms". See services/speech.select_backend."""
        return self._raw.get("speech", {}).get("backend", "auto")

    @property
    def min_detection_confidence(self) -> float:
        """Below this, the detected language is REPORTED but not acted on.

        Detection decides which language Gemini answers in, so a weak guess
        does not just mislabel a badge — it produces an answer in the wrong
        language. Observed on the demo machine: "what is car in xhosa" was
        detected as Afrikaans at 42% and came back in Afrikaans.

        The classifier is not at fault. It was trained on NCHLT news
        sentences and is being asked short English imperatives, which is
        out of distribution. In a live session everything at or above 64%
        was correct and the misfires sat at 35-42%.
        """
        return self._raw.get("model", {}).get("min_detection_confidence", 0.5)

    @property
    def coqui_sa_enabled(self) -> bool:
        """The multilingual South African VITS model. OFF by default.

        It is the only model found that speaks all eleven, and it closes
        the exact gap edge-tts and MMS leave — but three things make it a
        decision rather than a default:

          - It is cc-by-nc-4.0. This repository is MIT, so the model is
            fetched at runtime and never vendored; turning this on makes
            the resulting deployment non-commercial. Fine for an academic
            project, not fine for a product.
          - Its Hugging Face repo is gated. Approval is automatic, but the
            terms must be accepted once at the model page with HF_TOKEN
            set, or every fetch returns 403.
          - Its model card is an unfilled template — no training details,
            no evaluation, no named author. Nothing can tell you how it
            sounds except listening to it.

        Defaulting this on would make an MIT project quietly depend on a
        non-commercial model, which is the kind of thing that should never
        happen without someone choosing it.
        """
        return self._raw.get("speech", {}).get("enable_coqui_sa", False)

    @property
    def coqui_sa_speaker(self) -> str | None:
        """Which of the SA VITS model's 130 speakers to speak with.

        The ids are anonymous (`speaker_0` .. `speaker_129`) and carry no
        language or gender, so there is no principled way to choose one —
        it is picked by ear. Unset means the first in sorted order, which
        is deterministic rather than good.
        """
        value = self._raw.get("speech", {}).get("coqui_speaker", "")
        return str(value).strip() or None

    @property
    def local_voices(self) -> dict[str, str]:
        """VITS checkpoints this project trained itself, by language name.

        Read from [speech.local_voices]. Empty by default, which is the
        state on any machine that has not synced the model files — the
        speech layer drops paths that do not exist, so a stale entry
        degrades one language rather than stopping the app.
        """
        return self._raw.get("speech", {}).get("local_voices", {})

    @property
    def max_speech_characters(self) -> int:
        """How much of a reply to read aloud. 0 (the default) reads it all.

        This was capped at 400 characters to keep synthesis quick, which
        was the wrong trade for this app. The voice is not a flourish on
        top of text someone has already read — for a user who cannot
        easily read the screen it IS the answer, and half an answer is not
        an answer. An emergency number cut off before the number is worse
        than no audio.

        The cost is real: MMS runs a VITS forward pass per sentence on
        CPU, so a long reply takes roughly 60 seconds and yields two and a
        half minutes of audio. It runs after the text is rendered, so it
        delays nothing anyone is reading.
        """
        return int(self._raw.get("speech", {}).get("max_speech_characters", 0))

    @property
    def voice_fallback_related(self) -> bool:
        """Speak a language with a closely-related language's voice when it
        has none of its own. OFF by default: it is an approximation, and a
        silent approximation is worse than none.
        """
        return self._raw.get("speech", {}).get("fallback_related_voice", False)

    @property
    def voice_fallback_english(self) -> bool:
        """Speak in English when no other voice fits. OFF by default."""
        return self._raw.get("speech", {}).get("fallback_english_voice", False)

    @property
    def knowledge_base_path(self) -> str:
        return self._raw.get("retrieval", {}).get("data_path", os.path.join(BASE_DIR, "data", "african_data.csv"))

    @property
    def classifier_model_path(self) -> str:
        default = os.path.join(BASE_DIR, "models", "language_classifier.tflite")
        return self._raw.get("model", {}).get("classifier_path", default)

    @property
    def tokenizer_config_path(self) -> str:
        return self._raw.get("model", {}).get("tokenizer_config_path", os.path.join(BASE_DIR, "tokenizer_config.json"))


def load_config(config_path: str | None = None) -> Settings:
    return Settings(config_path)