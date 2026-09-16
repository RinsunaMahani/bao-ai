"""Bao AI - Language Detection.

The old codebase had two, incompatible language-ID paths: a keyword
heuristic (services/language_service.py, worked) and a TFLite/Keras
classifier (classifier.py, whose `.classify()` returned a class index with
no method to turn it back into a language string — console.py and app.py
both called a `.predict()` method that didn't exist on the class at all).

Rather than just fixing the crash, this makes the two paths what they
should always have been: two interchangeable implementations of the same
`LanguageDetector` interface, so the orchestrator never needs to know
which one it's talking to.

STATUS OF THE TFLITE BACKEND, as of this writing: it is now the default
(`prefer_ml=True` in bao/ui/streamlit_app.py and bao/ui/console_app.py),
not an opt-in experiment — but that took two rounds of finding and fixing
real, silent bugs before it earned that:

1. Preprocessing mismatch: the shipped code encoded text as
   `ord(c) % 256` per character; the model's own tokenizer_config.json
   proves it was trained on a 25,000-word, word-level vocabulary. Fixed
   by `_KerasWordTokenizer` below, reconstructing the real tokenization
   pipeline from that vocabulary file.
2. Class-order mismatch: even after fixing preprocessing, a manual
   spot-check still looked bad (1/4 correct). The actual cause, found by
   feeding one clearly-monolingual sentence per language through the real
   model and recording which output index fired for each: the code was
   mapping the model's output index to a language name using
   core.config.LABELS (a UI-ordering list), not the model's real
   training-time class order, which turned out to be alphabetical — see
   TFLITE_CLASS_ORDER below.

With both fixed, the deployed .tflite file (not just its Keras source)
was tested directly against three independent sets: 11/11 on formal
single-language sentences, 4/4 on short informal greetings, 5/5 on this
project's benchmark_bao.py query set (vs. the heuristic's 4/5 on the same
set). Full account, including how the underlying model's training claims
were independently corroborated first: REVIEW.md Sec 5.

The heuristic detector remains the automatic fallback if the model or
tokenizer files are ever missing — zero load time, no dependencies, and a
reasonable choice on its own terms, just not the more accurate one once
the ML path was actually working correctly.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from bao.core.config import LABELS
from bao.core.logging import get_logger

logger = get_logger(__name__)

# The classifier needs ONE thing from TensorFlow: a TFLite interpreter.
# Full TensorFlow is a ~600 MB dependency for it, and — the reason this
# matters in practice — TensorFlow publishes no wheels for Python 3.14, so
# on a current Python install `pip install tensorflow` fails outright with
# "No matching distribution found" and the ML classifier silently never
# runs.
#
# `ai-edge-litert` is Google's standalone LiteRT runtime: the same
# interpreter, a fraction of the size, and packaged for newer Pythons. It
# is tried first, with full TensorFlow as the fallback for environments
# that already have it.
#
# (requirements-ml.txt already noted that the pre-refactor pyproject
# declared ai-edge-litert while the code imported tensorflow. This closes
# that gap in the direction that actually installs.)
_TFLITE_RUNTIME = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None

if np is not None:
    try:
        from ai_edge_litert.interpreter import Interpreter as _TFLiteInterpreter
        _TFLITE_RUNTIME = "ai-edge-litert"
    except ImportError:
        try:
            import tensorflow as tf
            _TFLiteInterpreter = tf.lite.Interpreter
            _TFLITE_RUNTIME = "tensorflow"
        except ImportError:
            _TFLiteInterpreter = None

_HAS_TFLITE = _TFLITE_RUNTIME is not None


def tflite_runtime_name() -> str | None:
    """Which runtime provides the interpreter, or None. Reported so a
    missing classifier can be diagnosed without guessing.
    """
    return _TFLITE_RUNTIME


def has_tflite_support(model_path: str) -> bool:
    return _HAS_TFLITE and os.path.exists(model_path)


@dataclass
class DetectionResult:
    language: str
    confidence: float
    backend: str


class _KerasWordTokenizer:
    """Reimplements `tf.keras.preprocessing.text.Tokenizer.texts_to_sequences`
    + `pad_sequences`, reading the vocabulary from the model's own
    tokenizer_config.json rather than requiring the original Keras
    Tokenizer class to be importable (avoids version fragility from
    unpickling a serialized Tokenizer object across TF/Keras versions).

    This exists because of a real bug found via review and confirmed by
    inspecting the actual assets: the code previously shipped here encoded
    text as `ord(c) % 256` per character. But `tokenizer_config.json`
    documents `"char_level": false` and a 679,385-word vocabulary
    (`num_words: 25000`) — this is unambiguously a WORD-level tokenizer,
    not a character-level one. Character codes and word indices occupy
    completely different numerical spaces; feeding one where the model
    expects the other produces confident-looking, semantically meaningless
    output — worse than an error, because nothing crashes.

    IMPORTANT — what this fix does and does not establish: this
    reconstruction is not verified against the original training script
    (not available), so padding direction (pre/post) and truncation
    direction are Keras's defaults, not confirmed values. A small manual
    spot-check (a handful of hand-written sentences) after this fix did
    NOT show a clear accuracy improvement over the old, definitely-wrong
    preprocessing — which means the model's real quality is still
    unvalidated, independent of which preprocessing feeds it. This class
    replaces something proven wrong with something principled and
    documented, not with something proven correct. Treat the TFLite
    backend as experimental until `evaluate.py --detector-eval
    --use-ml-detector` has been run against a real labelled dataset (see
    eval/README.md) and produces a trustworthy accuracy number.
    """

    def __init__(self, tokenizer_config_path: str):
        with open(tokenizer_config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.word_index: dict[str, int] = cfg["word_index"]
        self.num_words: int | None = cfg.get("num_words")
        self.oov_token: str | None = cfg.get("oov_token")
        self.lower: bool = cfg.get("lower", True)
        self.filters: str = cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.split: str = cfg.get("split", " ")
        self._translate_table = str.maketrans({c: self.split for c in self.filters})
        self._oov_index = self.word_index.get(self.oov_token) if self.oov_token else None
        logger.info(f"Loaded word tokenizer: {len(self.word_index)} vocab entries, num_words={self.num_words}.")

    def _text_to_word_sequence(self, text: str) -> list[str]:
        if self.lower:
            text = text.lower()
        text = text.translate(self._translate_table)
        return [w for w in text.split(self.split) if w]

    def _texts_to_sequence(self, text: str) -> list[int]:
        vect: list[int] = []
        for word in self._text_to_word_sequence(text):
            index = self.word_index.get(word)
            if index is not None:
                if self.num_words and index >= self.num_words:
                    if self._oov_index is not None:
                        vect.append(self._oov_index)
                else:
                    vect.append(index)
            elif self._oov_index is not None:
                vect.append(self._oov_index)
        return vect

    def encode(self, text: str, maxlen: int = 35) -> np.ndarray:
        """Returns a (1, maxlen) float32 array ready for the model.

        padding='post', truncating='post' — determined empirically, not
        assumed. Keras's *default* is 'pre'/'pre', which was the original
        guess here; testing both directions against the real trained model
        with unambiguous, clearly-monolingual sentences showed the actual
        signature of correct vs. incorrect preprocessing: 'pre' produced
        near-uniform, low-confidence predictions (0.25-0.34, close to
        random across 11 classes), while 'post' produced near-certain
        predictions (0.97-1.00) that were also internally consistent (see
        the class-order note on TFLiteLanguageDetector below). That
        contrast — not a guess — is why 'post' is used here.
        """
        seq = self._texts_to_sequence(text)
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        pad_width = maxlen - len(seq)
        padded = seq + [0.0] * pad_width
        return np.array([padded], dtype=np.float32)


class LanguageDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> DetectionResult: ...


class HeuristicLanguageDetector(LanguageDetector):
    """Rule-based marker-word matching across all 11 official SA languages.
    No model to load, works instantly, and is transparent to debug — the
    default detector for exactly those reasons.
    """

    KEYWORD_MAP = {
        "isiZulu": ["sawubona", "sanibonani", "yebo", "ngicela", "usizo", "ngiyabonga", "unjani", "linjani"],
        "isiXhosa": ["molo", "molweni", "enkosi", "nceda", "unjani", "bhotani", "ewe", "ndiyabulela"],
        "siSwati": ["sawubona", "sanibona", "yebo", "ngiyabonga", "sicela", "unjani", "kantsi", "ngikhona"],
        "isiNdebele": ["lotjhani", "salibonani", "lotjha", "siyabonga", "ngiyathokoza", "ijano"],
        "Sepedi": ["dumela", "dumelang", "thusha", "hle", "bjang", "lekae", "ke a leboga", "thobela"],
        "Sesotho": ["lumela", "lumelang", "dumela", "dumelang", "kgotso", "jwang",
                    "ke a leboha", "thusa", "hle", "tsamaya"],
        "Setswana": ["dumela", "dumelang", "o kae", "le kae", "ke a leboga", "thusa", "ka kopo", "sentle"],
        "Xitsonga": ["avuxeni", "minjhani", "inkomu", "ahee", "ndza nkhensa", "nhlikanhi", "xewani"],
        "Tshivenda": ["ndaa", "nda", "ndi matsheloni", "ndi masiari", "ndi madekwana", "ndo livhuwa"],
        "Afrikaans": ["hallo", "goeie", "goeiedag", "dankie", "asseblief", "hoe", "gaan", "help", "totsiens"],
        "English": ["hello", "hi", "help", "please", "thank you", "good", "morning", "afternoon"],
    }

    def detect(self, text: str) -> DetectionResult:
        lowered = text.lower()
        scores = {lang: 0.05 for lang in LABELS}
        scores["English"] = 0.20  # slight baseline preference

        for lang, keywords in self.KEYWORD_MAP.items():
            for word in keywords:
                if re.search(r"\b" + re.escape(word) + r"\b", lowered):
                    scores[lang] += 0.70
                    break

        primary, score = max(scores.items(), key=lambda kv: kv[1])
        return DetectionResult(language=primary, confidence=round(min(score, 0.95), 2), backend="heuristic")


TFLITE_CLASS_ORDER = [
    "Afrikaans", "English", "isiNdebele", "isiXhosa", "isiZulu",
    "Sepedi", "Sesotho", "Setswana", "siSwati", "Tshivenda", "Xitsonga",
]
"""The trained model's actual output class order — alphabetical, and NOT
the same as core.config.LABELS (which is ordered for UI/config purposes,
not training). This was the second half of a two-part bug: even after
fixing preprocessing, passing LABELS here as the index-to-name mapping
silently mislabeled every prediction, because the two orderings don't
match. Confirmed empirically by feeding one clearly-monolingual sentence
per language through the real model and recording which output index
fired — all 11 landed on a distinct index with 0.97-1.00 confidence, and
the resulting index order was exactly alphabetical (case-insensitive),
consistent with a standard sorted-classes / LabelEncoder training setup.
Do not "fix" this back to LABELS — LABELS is deliberately a different,
UI-appropriate order and isn't wrong for its own purpose.
"""


class TFLiteLanguageDetector(LanguageDetector):
    """On-device ML classifier (quantized TFLite, falls back to the Keras
    source model if TFLite loading fails). Opt-in: requires tensorflow and
    the trained model file, and is meaningfully slower to construct than
    the heuristic since it loads model weights.
    """

    def __init__(self, model_path: str, fallback_keras_path: str | None = None,
                 tokenizer_config_path: str | None = None, labels: list[str] = TFLITE_CLASS_ORDER):
        if not _HAS_TFLITE:
            raise ImportError("tensorflow is required for TFLiteLanguageDetector.")
        self.labels = labels
        self.interpreter = None
        self.keras_model = None
        self.is_tflite = False
        self._tokenizer: _KerasWordTokenizer | None = None
        self._initialize(model_path, fallback_keras_path, tokenizer_config_path)

    def _initialize(self, model_path: str, fallback_keras_path: str | None, tokenizer_config_path: str | None) -> None:
        # The model's own tokenizer_config.json (word_index, char_level=False,
        # num_words=25000) proves this is a WORD-level classifier — a fact
        # discovered via external review and confirmed here by inspecting
        # the asset directly, not assumed. The model can't be meaningfully
        # used without loading that same tokenizer, so it's a hard
        # requirement to construct this class, not an optional extra.
        if not tokenizer_config_path or not os.path.exists(tokenizer_config_path):
            raise FileNotFoundError(
                f"Tokenizer config not found at {tokenizer_config_path!r}. The TFLite/Keras "
                "language model requires it to preprocess text correctly and cannot be used "
                "without it — see the module docstring for why."
            )
        self._tokenizer = _KerasWordTokenizer(tokenizer_config_path)

        if os.path.exists(model_path):
            try:
                self.interpreter = _TFLiteInterpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.is_tflite = True
                logger.info(f"Loaded TFLite language model from {model_path}.")
                return
            except Exception as e:
                logger.warning(f"TFLite interpreter init failed: {e}. Trying Keras fallback.")

        if fallback_keras_path and os.path.exists(fallback_keras_path):
            # Keras needs full TensorFlow, which the LiteRT-only install
            # deliberately does not have. Import it here rather than at
            # module level so the common path (TFLite) never pays for it,
            # and so a LiteRT-only environment gets a clear message instead
            # of a NameError from a `tf` that was never bound.
            try:
                import tensorflow as tf_keras
            except ImportError as e:
                raise RuntimeError(
                    "Keras fallback needs full TensorFlow, which is not installed. "
                    "The TFLite path is the supported one — check that "
                    f"{fallback_keras_path!r} is not being used in place of a "
                    ".tflite model."
                ) from e

            with tf_keras.keras.utils.custom_object_scope({"quantization_config": None}):
                self.keras_model = tf_keras.keras.models.load_model(fallback_keras_path)
            logger.info(f"Loaded fallback Keras model from {fallback_keras_path}.")
            return

        raise RuntimeError("Neither TFLite nor Keras language model could be loaded.")

    def detect(self, text: str) -> DetectionResult:
        input_data = self._tokenizer.encode(text, maxlen=35)

        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])
        else:
            output = self.keras_model.predict(input_data, verbose=0)

        probabilities = output[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        language = self.labels[predicted_class] if predicted_class < len(self.labels) else "Unknown"
        # Report the engine that ACTUALLY ran, not the one this class is
        # named after. Previously this always said "tflite" even when the
        # Keras fallback produced the prediction — which mattered more than
        # cosmetically: tests/test_language_detector.py asserts
        # `backend == "tflite"` precisely to prove the TFLite path ran, and
        # this bug would have let a Keras fallback satisfy that assertion
        # falsely, defeating the check added in the previous review round.
        backend = "tflite" if self.is_tflite else "keras"
        return DetectionResult(language=language, confidence=confidence, backend=backend)


# The pan-African bundle is trained on ISO 639-3 codes, which is correct for
# a label encoder and wrong for everything downstream. Left as codes, the
# interface reads "Language: swa" and — worse — the Gemini system prompt
# reads "Respond naturally in swa", which is an instruction in a code the
# model has to guess at rather than a language it knows by name.
#
# Mapped here, at the boundary where the model's labels become the
# application's, so nothing further down has to know the difference.
PAN_AFRICAN_LANGUAGE_NAMES = {
    "amh": "Amharic",
    "fra": "French",
    "hau": "Hausa",
    "ibo": "Igbo",
    "lin": "Lingala",
    "lug": "Luganda",
    "orm": "Oromo",
    "pcm": "Nigerian Pidgin",
    "run": "Kirundi",
    "sna": "Shona",
    "som": "Somali",
    "swa": "Swahili",
    "tir": "Tigrinya",
    "yor": "Yoruba",
}


class SklearnLanguageDetector(LanguageDetector):
    """Character n-gram TF-IDF + a linear classifier, loaded from a joblib
    bundle.

    Added as a THIRD backend rather than as a replacement for the TFLite
    LSTM. The two are not obviously ordered — each is better at something —
    and the honest way to choose is to run both against the same held-out
    set rather than to argue about it:

      - Character n-grams (`char_wb`) genuinely handle unseen words,
        typos and slang better than a word-level vocabulary. A live
        session had a user type "axuxeni" for "avuxeni"; the word-level
        path could only emit <OOV> and fell through to English, where
        character n-grams would still have seen the Xitsonga shape.
      - The LSTM reads word ORDER, which a bag of n-grams discards. That
        was the stated justification for choosing it over a bag-of-words
        model in the first place, specifically for separating Sepedi,
        Sesotho and Setswana.

    Which effect dominates on this corpus is an empirical question, and
    `compare_detectors.py` answers it. Until it has been answered on a
    held-out set, neither backend should be removed.

    Expected bundle (joblib): {"pipeline": <fitted sklearn Pipeline>,
    "labels": <list matching pipeline.classes_>}. The labels are read from
    the fitted estimator for the same reason as the sign classifier —
    `classes_` order is what `predict_proba` columns follow, and a
    separately-sorted list agreeing with it is a coincidence, not a
    guarantee.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._pipeline = None
        self._labels: list[str] = []

        try:
            import joblib
        except ImportError:
            logger.warning("joblib not installed — sklearn detector unavailable.")
            return

        if not os.path.exists(model_path):
            logger.info(f"No sklearn detector bundle at {model_path}.")
            return

        try:
            bundle = joblib.load(model_path)
            self._pipeline = bundle["pipeline"]
            self._labels = [str(label) for label in bundle["labels"]]
        except Exception as e:
            logger.warning(f"Could not load sklearn detector: {e}")
            self._pipeline = None
            return

        # A bundle whose labels are still encoded integers would "work" and
        # report languages called "0" and "7". Refuse it.
        if all(label.isdigit() for label in self._labels):
            logger.error(
                "sklearn detector bundle has encoded labels, not language names — "
                "rebuild it with scripts/build_pan_african_bundle.py."
            )
            self._pipeline = None
            return

        if len(self._labels) != len(self._pipeline.classes_):
            logger.error(
                "sklearn detector labels do not match the fitted classes — refusing "
                "to load rather than mislabel every prediction."
            )
            self._pipeline = None

    def is_available(self) -> bool:
        return self._pipeline is not None

    def detect(self, text: str) -> DetectionResult:
        if self._pipeline is None or not text or not text.strip():
            return DetectionResult(language="English", confidence=0.0, backend="none")
        try:
            probabilities = self._pipeline.predict_proba([text])[0]
        except Exception as e:
            logger.warning(f"sklearn detection failed: {e}")
            return DetectionResult(language="English", confidence=0.0, backend="none")

        best = int(probabilities.argmax())
        code = self._labels[best]
        return DetectionResult(
            # Unmapped codes fall through unchanged rather than being
            # dropped, so retraining with extra languages still works —
            # it just shows a code until a name is added above.
            language=PAN_AFRICAN_LANGUAGE_NAMES.get(code, code),
            confidence=float(probabilities[best]),
            backend="sklearn",
        )


class CompositeLanguageDetector(LanguageDetector):
    """A primary detector with a fallback consulted only when the primary
    is unsure.

    Bao is South-Africa-first, so the eleven-language classifier is always
    asked first and its answer is used whenever it is confident. The
    pan-African model covers fourteen languages that do not overlap with
    those eleven, and is consulted only when the primary falls below its
    threshold — never to overrule a confident primary answer.

    Why not "take whichever is more confident": the two models are trained
    on different corpora and are not calibrated against each other, so
    their probabilities are not comparable. Measured on the shipped
    pan-African model:

        South African / English input  ->  16-23% confidence (wrong label)
        genuine pan-African input      ->  37-95% confidence (right label)

    So a secondary answer is only accepted above `secondary_min`, which
    sits above everything observed in the first group and below everything
    observed in the second. That margin comes from eleven probe inputs, not
    a held-out set — it is deliberately conservative, and erring means
    falling back to today's behaviour rather than guessing.
    """

    def __init__(
        self,
        primary: LanguageDetector,
        secondary: LanguageDetector,
        primary_min: float = 0.5,
        secondary_min: float = 0.35,
        enabled: bool = True,
    ):
        self.primary = primary
        self.secondary = secondary
        self.primary_min = primary_min
        self.secondary_min = secondary_min
        # Flippable at runtime so the UI can offer a toggle. The alternative
        # — rebuilding the pipeline when the setting changes — would mean
        # invalidating Streamlit's @st.cache_resource and reloading the
        # TFLite model on every flip, which is slow and easy to get wrong
        # mid-demo. Disabled simply means the secondary is never consulted,
        # so the primary's behaviour is bit-for-bit what it would be if the
        # composite did not exist.
        self.enabled = enabled

    def detect(self, text: str) -> DetectionResult:
        result = self.primary.detect(text)
        if not self.enabled:
            return result
        if result.confidence >= self.primary_min:
            return result

        fallback = self.secondary.detect(text)
        if fallback.confidence < self.secondary_min:
            return result

        logger.info(
            f"Primary detector unsure ({result.language} {result.confidence:.2f}); "
            f"pan-African model says {fallback.language} ({fallback.confidence:.2f})."
        )
        return fallback


def get_language_detector(prefer_ml: bool = False, model_path: str | None = None,
                           fallback_keras_path: str | None = None,
                           tokenizer_config_path: str | None = None) -> LanguageDetector:
    """Factory used by the orchestrator. Falls back to the heuristic
    detector on any failure to load the ML backend, so choosing
    `prefer_ml=True` can never take the whole app down — it can only fall
    back to the heuristic if the model assets aren't present.

    The function's own default is `prefer_ml=False` (a plain library call
    shouldn't silently trigger a multi-second model load) — but both
    bao/ui/streamlit_app.py and bao/ui/console_app.py explicitly pass
    `prefer_ml=True`, which is the actual application default. See this
    module's docstring for why: the TFLite backend is validated (20/20
    across three test sets), not just bug-fixed.
    """
    if prefer_ml and model_path and has_tflite_support(model_path):
        try:
            return TFLiteLanguageDetector(model_path, fallback_keras_path, tokenizer_config_path)
        except Exception as e:
            logger.warning(f"Falling back to heuristic detector: {e}")
    return HeuristicLanguageDetector()
