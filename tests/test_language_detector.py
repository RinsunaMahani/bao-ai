import pytest

from bao.core.config import Settings
from bao.services.language_detector import HeuristicLanguageDetector, get_language_detector


def test_english_default():
    detector = HeuristicLanguageDetector()
    result = detector.detect("How do I reset my password?")
    assert result.language == "English"
    assert result.backend == "heuristic"


def test_zulu_keyword_detected():
    detector = HeuristicLanguageDetector()
    result = detector.detect("Sawubona, ngicela usizo")
    assert result.language == "isiZulu"
    assert result.confidence > 0.5


def test_afrikaans_keyword_detected():
    detector = HeuristicLanguageDetector()
    result = detector.detect("Dankie vir die hulp")
    assert result.language == "Afrikaans"


def test_factory_falls_back_to_heuristic_without_ml_assets():
    # No model_path given -> must not attempt (or crash on) the ML backend.
    detector = get_language_detector(prefer_ml=True, model_path=None)
    assert isinstance(detector, HeuristicLanguageDetector)


def test_confidence_is_bounded():
    detector = HeuristicLanguageDetector()
    for text in ["", "hello", "sawubona molo dumela dankie"]:
        result = detector.detect(text)
        assert 0.0 <= result.confidence <= 0.95


_settings = Settings()


def _tflite_actually_works() -> bool:
    """Attempts real construction rather than checking any proxy (file
    existence, or even `import tensorflow` succeeding) for whether the ML
    path is usable. Confirmed necessary the hard way: an uninstalled
    tensorflow can leave a broken namespace-package stub behind where
    `import tensorflow` succeeds but `tf.lite` doesn't exist — so even an
    import-success check isn't reliable. Only actually building the
    detector and asking it what backend it landed on tells the truth.
    """
    try:
        from bao.services.language_detector import get_language_detector

        detector = get_language_detector(
            prefer_ml=True,
            model_path=_settings.classifier_model_path,
            tokenizer_config_path=_settings.tokenizer_config_path,
        )
        return detector.detect("test").backend == "tflite"
    except Exception:
        return False


_tflite_available = _tflite_actually_works()


class _StubKerasModel:
    """Minimal stand-in for a loaded Keras model — returns a fixed
    probability vector so the Keras-fallback reporting branch can be
    exercised without loading the real 38 MB model or needing a
    deliberately corrupted .tflite file on disk.
    """

    def predict(self, input_data, verbose=0):
        import numpy as np

        probs = np.zeros((1, 11), dtype=np.float32)
        probs[0][1] = 0.99  # index 1 == "English" in TFLITE_CLASS_ORDER
        return probs


@pytest.mark.skipif(
    not _tflite_available,
    reason="TFLite detector could not be constructed and confirmed working "
           "— see _tflite_actually_works().",
)
class TestTFLiteLanguageDetector:
    """Regression tests for two real, stacked bugs found via review and
    fixed by direct empirical testing against the real trained model:
    (1) preprocessing used character codes instead of the model's actual
    word-level tokenizer, and (2) even after fixing that, the class-index-
    to-language-name mapping used the app's general-purpose LABELS order
    instead of the model's actual (alphabetical) training-time class
    order. Both bugs were silent — no crash, no shape error, just
    confidently wrong output — which is exactly why they need pinned
    tests, not just a one-time manual check.
    """

    def test_padding_direction_matches_training(self):
        """post-padding must produce high-confidence, correct predictions
        on unambiguous sentences; this is a proxy for 'preprocessing
        matches what the model was actually trained on' since there's no
        access to the original training script to confirm directly.
        """
        from bao.services.language_detector import get_language_detector

        detector = get_language_detector(
            prefer_ml=True, model_path=_settings.classifier_model_path,
            tokenizer_config_path=_settings.tokenizer_config_path,
        )
        result = detector.detect("Die regering het nuwe beleid aangekondig vir onderwys in hierdie land")
        # Asserted BEFORE checking the language: if this ever silently
        # falls back to the heuristic (e.g. a half-broken tensorflow
        # install — reproduced and confirmed as a real failure mode, not
        # hypothetical), the test must fail with an obvious "wrong
        # backend" message, not a confusing "wrong language" one that
        # looks like the ML model itself failed when it never even ran.
        assert result.backend == "tflite", (
            f"Detector silently fell back to '{result.backend}' — this test doesn't "
            "validate the ML model if the ML model never actually ran."
        )
        assert result.language == "Afrikaans"
        assert result.confidence > 0.9  # near-random (~0.3) would indicate wrong padding direction

    def test_class_order_is_alphabetical_not_ui_labels_order(self):
        """Pins the exact reverse-engineered class order — if this ever
        starts failing, someone "fixed" TFLITE_CLASS_ORDER back to LABELS,
        which is the bug this test exists to catch.
        """
        from bao.services.language_detector import TFLITE_CLASS_ORDER

        assert TFLITE_CLASS_ORDER == sorted(TFLITE_CLASS_ORDER, key=str.lower)

    def test_reports_tflite_backend_when_tflite_actually_ran(self):
        """Pins the fix for a real bug: DetectionResult previously
        hardcoded backend="tflite" even when the Keras fallback produced
        the prediction. That mattered because the assertions above use
        `backend == "tflite"` to prove the TFLite path ran — the bug would
        have let a Keras fallback satisfy them falsely, quietly defeating
        the whole point of that check.
        """
        from bao.services.language_detector import get_language_detector

        detector = get_language_detector(
            prefer_ml=True, model_path=_settings.classifier_model_path,
            tokenizer_config_path=_settings.tokenizer_config_path,
        )
        assert detector.is_tflite is True
        assert detector.detect("hello there").backend == "tflite"

    def test_backend_string_follows_is_tflite_flag(self):
        """Directly exercises the reporting logic for the Keras-fallback
        case without needing a genuinely corrupt .tflite file on disk:
        flip the flag the reporting branches on and confirm the reported
        backend follows it. If someone reverts to a hardcoded string, this
        fails.
        """
        from bao.services.language_detector import get_language_detector

        detector = get_language_detector(
            prefer_ml=True, model_path=_settings.classifier_model_path,
            tokenizer_config_path=_settings.tokenizer_config_path,
        )
        original = detector.is_tflite
        try:
            detector.is_tflite = False
            detector.keras_model = _StubKerasModel()
            assert detector.detect("hello there").backend == "keras"
        finally:
            detector.is_tflite = original
            detector.keras_model = None

    def test_disambiguates_the_closely_related_sotho_tswana_group(self):
        """Answers a question raised in review: is the Sepedi/Sesotho
        confusion a model weakness or a heuristic one?

        Sepedi, Sesotho and Setswana are the most closely related group in
        the label set and all three share the greeting "Dumela" — which is
        exactly why the keyword heuristic collapses them. Three sentences
        per language, including all three "Dumela" greetings, must be
        separated correctly. This being 9/9 is what establishes that the
        confusion is a limitation of keyword matching, not of the trained
        model, and it's the concrete justification for the ML detector
        being the default rather than a nice-to-have.
        """
        from bao.services.language_detector import get_language_detector

        detector = get_language_detector(
            prefer_ml=True, model_path=_settings.classifier_model_path,
            tokenizer_config_path=_settings.tokenizer_config_path,
        )
        cases = [
            ("Sepedi", "Dumela, nka thusha bjang ka system e?"),
            ("Sepedi", "Mmuso o begile maano a mafsa a thuto nageng ye"),
            ("Sepedi", "Ke leboga thuso ya gago kudu"),
            ("Sesotho", "Dumela, o kae kajeno?"),
            ("Sesotho", "Mmuso o phatlaladitse maano a macha a thuto naheng ena"),
            ("Sesotho", "Ke leboha thuso ya hao haholo"),
            ("Setswana", "Dumela, o tsogile jang?"),
            ("Setswana", "Puso e itsisitse maano a mantsha a thuto mo lefatsheng le"),
            ("Setswana", "Ke leboga thuso ya gago thata"),
        ]
        wrong = []
        for expected, text in cases:
            result = detector.detect(text)
            assert result.backend == "tflite", f"fell back to '{result.backend}'"
            if result.language != expected:
                wrong.append((text, expected, result.language))
        assert not wrong, f"Sotho-Tswana disambiguation regressed: {wrong}"

    def test_beats_heuristic_on_the_known_dumela_ambiguity(self):
        """The one case the heuristic detector is documented to get
        wrong (see REVIEW.md) — Sepedi "dumela" misread as Sesotho due to
        shared keywords. The whole point of having an ML backend is that
        it shouldn't share the heuristic's specific blind spots.
        """
        from bao.services.language_detector import get_language_detector

        detector = get_language_detector(
            prefer_ml=True, model_path=_settings.classifier_model_path,
            tokenizer_config_path=_settings.tokenizer_config_path,
        )
        result = detector.detect("Dumela, nka thusha bjang ka system e?")
        assert result.backend == "tflite", f"Detector silently fell back to '{result.backend}'"
        assert result.language == "Sepedi"
