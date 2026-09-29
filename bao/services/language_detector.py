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
import threading
import warnings
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
        self.num_words: int | None = cfg.get("num_words")
        self.oov_token: str | None = cfg.get("oov_token")
        # Only words the model can actually receive are kept. Keras maps a
        # word ranked at or beyond num_words to the OOV token, exactly as
        # it maps a word it has never seen, so dropping those entries
        # changes no encoding - and it drops 654,386 of the 679,385: 75 MB
        # of resident memory down to 6 MB, measured. tests/
        # test_tokenizer_parity.py checks the encodings against Keras
        # itself, rare words included.
        vocabulary: dict[str, int] = cfg["word_index"]
        self.vocabulary_size = len(vocabulary)
        if self.num_words:
            vocabulary = {w: i for w, i in vocabulary.items() if i < self.num_words}
        self.word_index: dict[str, int] = vocabulary
        self.lower: bool = cfg.get("lower", True)
        self.filters: str = cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.split: str = cfg.get("split", " ")
        self._translate_table = str.maketrans({c: self.split for c in self.filters})
        self._oov_index = self.word_index.get(self.oov_token) if self.oov_token else None
        logger.info(
            f"Loaded word tokenizer: {self.vocabulary_size} vocab entries, "
            f"{len(self.word_index)} usable (num_words={self.num_words})."
        )

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


_TARGET_LANGUAGE_RE = None


def named_target_language(query: str) -> str | None:
    """The language a question ASKS FOR, as in "explain calculus in
    Xitsonga" — or None when it names none.

    This is a different question from "what language is this written in",
    and conflating them is what made cross-lingual requests unreliable. The
    query above is written in English, so detection correctly reports
    English, the system instruction then reads "Primary response language:
    English", and the model weighs that header above the request buried in
    the sentence. Measured on gemini-3.5-flash-lite: "explain gravity in
    Afrikaans" came back in English on both attempts, and "in xitsonga"
    came back in English then in something closer to Afrikaans.

    The user stated their language explicitly. That should outrank a guess
    about the language they happened to type the request in.

    Deliberately narrow, in two ways.

    It needs the preposition immediately before a known language name, so
    "how many official languages does South Africa have" is unaffected.

    And it needs the name to END the phrase — followed by the end of the
    text, punctuation, or a function word — because many of these names
    double as adjectives. "I am interested in French cuisine", "a degree in
    English literature" and "recipes popular in Somali culture" all put a
    language name straight after "in", and all five sentences like them
    were read as requests to switch the reply language until this was
    added. The difference is what comes next: a request is followed by
    nothing, or by "please", "for a grade 10 learner", "with examples",
    "how it works"; a mention is followed by the noun the name describes.

    Matching is case-insensitive because people type "xitsonga" and
    "Xitsonga" interchangeably, and it accepts the everyday names in
    LANGUAGE_ALIASES ("in zulu", "in tsonga") for the same reason.
    """
    global _TARGET_LANGUAGE_RE
    if _TARGET_LANGUAGE_RE is None:
        from bao.core.config import LABELS, LANGUAGE_ALIASES, PAN_AFRICAN_LABELS

        # Longest first, so "Nigerian Pidgin" is not shadowed by a shorter
        # name that happens to be a prefix of it.
        names = sorted(LABELS + PAN_AFRICAN_LABELS + list(LANGUAGE_ALIASES), key=len, reverse=True)
        pattern = "|".join(re.escape(n.lower()) for n in names)
        # What may follow the name for it to count as a request. Function
        # words only — a noun here means the name is describing that noun.
        # "and" is left out on purpose: "fluent in isiZulu and English" is
        # a description of a person, not an instruction.
        followers = (
            "please|pls|plz|language|languages|only|for|so|too|instead|now|"
            "thanks|thank|again|if|because|rather|as|with|without|"
            "what|how|why|when|where|who|which|about"
        )
        _TARGET_LANGUAGE_RE = re.compile(
            rf"\bin\s+({pattern})"
            rf"(?=\s*$|\s*[.,!?;:)\]\"'»”’]|\s+(?:{followers})\b)",
            re.IGNORECASE,
        )

    match = _TARGET_LANGUAGE_RE.search(query)
    if not match:
        return None

    from bao.core.config import LABELS, LANGUAGE_ALIASES, PAN_AFRICAN_LABELS

    found = " ".join(match.group(1).lower().split())
    for name in LABELS + PAN_AFRICAN_LABELS:
        if name.lower() == found:
            return name
    return LANGUAGE_ALIASES.get(found)


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
        # TFLite inference is three stateful calls against one interpreter
        # — set_tensor, invoke, get_tensor — so it is not reentrant. The
        # web app serves each session on its own thread and shares this
        # object between them, and two sessions detecting at the same
        # moment made LiteRT raise "There is at least 1 reference to
        # internal data in the interpreter": measured with three threads,
        # two of them died outright.
        #
        # A lock rather than an interpreter per session, because inference
        # is about a millisecond while loading the model costs seconds and
        # ~13 MB. Serialising a millisecond is free; duplicating the model
        # per visitor is not.
        self._lock = threading.Lock()
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
            with self._lock:
                self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
                self.interpreter.invoke()
                # Copied inside the lock. get_tensor returns a view onto
                # the interpreter's own buffer, so releasing the lock first
                # would let the next thread overwrite the numbers this one
                # is about to read — and LiteRT refuses to invoke at all
                # while such a reference is outstanding.
                output = self.interpreter.get_tensor(
                    self.output_details[0]["index"]).copy()
        else:
            with self._lock:
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


def _minor(version: str) -> tuple[int, ...]:
    """(major, minor) of a version string, ignoring the patch level."""
    return tuple(int(part) for part in re.findall(r"\d+", version)[:2])


def sklearn_minor_mismatches(caught) -> list[tuple[str, str]]:
    """(pickled, installed) scikit-learn versions that differ in MAJOR or
    MINOR version, from warnings recorded while unpickling.

    scikit-learn warns on ANY version difference when loading a pickle,
    patch-level included. A patch difference is routine — CI installs the
    newest 1.9.x while a bundle saved under 1.9.0 is still read faithfully —
    so treating every warning as a fault turned the suite red each time a
    patch shipped. A minor difference is the real risk: on Python 3.10, pip
    can only install scikit-learn 1.7.2, and a 1.9 pickle loaded by 1.7 is
    the unsafe direction, where failure is silent rather than raised.

    One rule, used by the loader below and by the test that guards it.
    """
    mismatches = []
    for record in caught:
        if type(record.message).__name__ != "InconsistentVersionWarning":
            continue
        pickled = getattr(record.message, "original_sklearn_version", "")
        installed = getattr(record.message, "current_sklearn_version", "")
        if _minor(pickled) != _minor(installed):
            mismatches.append((pickled, installed))
    return sorted(set(mismatches))


def _report_version_skew(caught) -> None:
    """Silent for a patch difference, loud for a minor one — and every
    OTHER warning raised while loading is passed through untouched.

    The bundle is loaded under catch_warnings so that the version warnings
    can be classified, which means everything raised during the load is
    captured. Dropping the rest would have silenced unrelated warnings
    along with the ones this is meant to judge — tidier output bought by
    hiding things. Only the version warnings are this function's business.
    """
    for record in caught:
        if type(record.message).__name__ != "InconsistentVersionWarning":
            warnings.warn_explicit(
                record.message, record.category, record.filename, record.lineno)
    for pickled, installed in sklearn_minor_mismatches(caught):
        logger.warning(
            f"The pan-African detector was saved with scikit-learn {pickled} and "
            f"is being read by {installed}. Across minor versions a pickled model "
            "is not guaranteed to load faithfully, and it fails silently when it "
            "does not. Install scikit-learn from the pinned range in "
            "requirements.txt, or re-save the bundle and re-run its tests: "
            "python scripts/build_pan_african_bundle.py --reexport"
        )


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
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bundle = joblib.load(model_path)
            self._pipeline = bundle["pipeline"]
            self._labels = [str(label) for label in bundle["labels"]]
        except Exception as e:
            logger.warning(f"Could not load sklearn detector: {e}")
            self._pipeline = None
            return
        _report_version_skew(caught)

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

        South African / English input  ->  13-44% confidence (wrong label)
        genuine pan-African input      ->  43-89% confidence (right label)

    So a secondary answer is only accepted above a threshold that sits
    above the first group and below the second, rather than by comparing
    the two numbers directly.

    TWO THRESHOLDS, because one is not enough — and this corrects a rule
    that was wrong for a measurable reason rather than a debatable one.

    The original design consulted the secondary ONLY when the primary fell
    below `primary_min`, on the assumption that a classifier is unsure
    about input that is not in its languages. That assumption is false. The
    primary is an 11-class softmax with no "none of the above" output, so
    out-of-distribution input does not make it hesitate — it makes it pick
    the nearest of its eleven, confidently. Measured on one sentence per
    language (scripts/probe_language_routing.py):

        Luganda  -> Xitsonga  100%        Hausa           -> English  97%
        Yoruba   -> Xitsonga   99%        Nigerian Pidgin -> English  85%
        Igbo     -> isiZulu    64%        French          -> isiNdebele 63%

    Every one of those clears `primary_min`, so the secondary was never
    asked, and seven of the fourteen languages were answered in a language
    the user had not written in. The gate was reading the wrong signal:
    primary confidence says nothing about whether the input is
    South African, while secondary confidence does.

    So `secondary_strong` (0.5) accepts a strong secondary answer outright,
    even over a confident primary. `secondary_min` (0.35) keeps the
    original, more cautious path for a moderate secondary answer, which is
    still only taken when the primary is itself unsure.

    Evidence for 0.5, from 24 South African inputs (11 sentences and the 13
    single-word probes in tests/test_pan_african.py) and 14 pan-African
    sentences:

        highest secondary confidence on ANY South African input   44%
        lowest secondary confidence where it was RIGHT             43%

    0.5 clears the first with margin. The 44% case is English read as
    Nigerian Pidgin, which is the one genuinely hard pair here — Nigerian
    Pidgin is English-lexifier, so it is confusable with English by
    construction rather than by accident, and keeping English is the right
    way to resolve it for this app.

    Known cost, stated rather than hidden: Luganda scores 43% and so falls
    below the bar, leaving it detected as Xitsonga. Admitting it would mean
    dropping the threshold under English's 44% and relabelling English as
    Nigerian Pidgin, which is a worse trade for an app whose lingua franca
    is English. 24 of 25 languages route correctly.

    These figures come from one sentence per language, not a held-out set.
    They are enough to show the old rule was wrong and to choose between
    two candidate thresholds; they are not a calibration claim.
    """

    def __init__(
        self,
        primary: LanguageDetector,
        secondary: LanguageDetector,
        primary_min: float = 0.5,
        secondary_min: float = 0.35,
        secondary_strong: float = 0.5,
        enabled: bool = True,
    ):
        self.primary = primary
        self.secondary = secondary
        self.primary_min = primary_min
        self.secondary_min = secondary_min
        self.secondary_strong = secondary_strong
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

        # Asked unconditionally. Gating this call on the primary being
        # unsure is what hid seven languages: the primary is confidently
        # wrong on input outside its eleven, so it never asked.
        fallback = self.secondary.detect(text)

        # A strong secondary answer is evidence the input is not South
        # African at all, which is a claim the primary cannot make about
        # itself. It therefore outranks a confident primary.
        if fallback.confidence >= self.secondary_strong:
            if result.confidence >= self.primary_min:
                logger.info(
                    f"Primary confident ({result.language} {result.confidence:.2f}) but "
                    f"pan-African model is stronger ({fallback.language} "
                    f"{fallback.confidence:.2f}); taking the pan-African answer."
                )
            return fallback

        # Below that, the cautious original path: a moderate secondary
        # answer is only taken when the primary is unsure too.
        if result.confidence >= self.primary_min:
            return result
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
            ml = TFLiteLanguageDetector(model_path, fallback_keras_path, tokenizer_config_path)
        except Exception as e:
            logger.warning(f"Falling back to heuristic detector: {e}")
            return HeuristicLanguageDetector()

        # The heuristic is kept as a RUNTIME fallback, not merely a
        # load-time one, because the two fail on opposite inputs.
        #
        # The classifier is word-level over a 25,000-word vocabulary built
        # from NCHLT news sentences. A single greeting is not in it, so the
        # tokenizer emits nothing but OOV markers and the model returns its
        # prior — the same answer for every such input. Measured: nine
        # different greetings across eight languages ("Sawubona", "Molo",
        # "Avuxeni", "Thobela", "Lumela", "Ndaa", "Lotjhani", "Goeiedag",
        # "Hello") ALL came back as siSwati at exactly 35%. That is not a
        # detection, it is a constant, and the confidence floor then
        # correctly refused to act on it — so greetings, which is what
        # people actually open a conversation with, were all answered in
        # English.
        #
        # Those same words are precisely what the keyword map contains, and
        # it gets all nine right. So: the model leads on anything it can
        # read, and the keyword list answers the short input it cannot.
        #
        # Thresholds are deliberate, not inherited:
        #
        #   secondary_strong is unreachable (1.01). A marker word must
        #   never overrule a confident model. "Kan jy my help" is Afrikaans
        #   at 100%, but "help" appears in both the Afrikaans and English
        #   keyword lists and the English baseline tips it to English at
        #   90% — allowing a strong-secondary override would turn a correct
        #   answer into a wrong one.
        #
        #   secondary_min is 0.5, which sits above the keyword detector's
        #   no-match scores (0.05, or 0.20 for its slight English baseline)
        #   and below a real marker-word hit (0.75). So it is consulted
        #   only when it actually recognised a word, never when it is
        #   guessing.
        return CompositeLanguageDetector(
            primary=ml,
            secondary=HeuristicLanguageDetector(),
            primary_min=0.5,
            secondary_min=0.5,
            secondary_strong=1.01,
        )
    return HeuristicLanguageDetector()
