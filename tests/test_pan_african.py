"""The pan-African detector must be invisible until deliberately enabled.

Adding a second detector changes the detection path for EVERY message, not
only the fourteen new languages — detection feeds greeting lookup,
retrieval language preference and voice selection. So the property that
matters most is not "does it work" but "does it change nothing when off".
"""

import pytest

from bao.bootstrap import build_orchestrator
from bao.core.config import LABELS, Settings
from bao.services.language_detector import (
    CompositeLanguageDetector,
    DetectionResult,
    HeuristicLanguageDetector,
    SklearnLanguageDetector,
)

# Trigger words for all eleven South African languages. If enabling the
# second detector changes any of these, it is not safe to enable.
SA_PROBES = [
    ("avuxeni", "Xitsonga"), ("sawubona", "isiZulu"), ("molo", "isiXhosa"),
    ("dumela", "Sesotho"), ("thobela", "Sepedi"), ("lumela", "Sesotho"),
    ("nda", "Tshivenda"), ("goeiedag", "Afrikaans"), ("lotjhani", "isiNdebele"),
    ("sanibonani", "isiZulu"), ("hello how can you help me", "English"),
    ("o kae", "Setswana"), ("kantsi", "siSwati"),
]

# Note for anyone extending this: siSwati shares almost its whole keyword
# set with isiZulu ("sawubona", "yebo", "unjani", "ngiyabonga"), so under
# the heuristic backend isiZulu wins every shared trigger and siSwati is
# only reachable through its distinctive words. That is a limitation of the
# keyword fallback, not of this test — the ML classifier separates them.


class _Stub:
    def __init__(self, language, confidence, backend="stub"):
        self.result = DetectionResult(language=language, confidence=confidence, backend=backend)

    def detect(self, text):
        return self.result


# --- The default: off ---------------------------------------------------


def test_disabled_by_default():
    assert Settings().pan_african_enabled is False


def test_default_build_leaves_the_composite_inert():
    """The composite is now constructed whenever a bundle exists, so the UI
    can toggle it without rebuilding the pipeline. The guarantee therefore
    moves from "not wrapped" to "wrapped but inert" — which has to be
    checked behaviourally rather than by type.
    """
    _, orchestrator = build_orchestrator()
    detector = orchestrator.language_detector
    if isinstance(detector, CompositeLanguageDetector):
        assert detector.enabled is False, "must start disabled by default"
        # Inert means the primary's answer passes through untouched.
        for probe in ("sawubona", "hello there", "avuxeni"):
            assert detector.detect(probe) == detector.primary.detect(probe)


def test_toggling_the_composite_changes_nothing_else():
    """Flipping .enabled must not disturb the primary detector, since the
    same object serves every South African language.
    """
    _, orchestrator = build_orchestrator()
    detector = orchestrator.language_detector
    if not isinstance(detector, CompositeLanguageDetector):
        pytest.skip("no pan-African bundle present")

    before = detector.primary.detect("sawubona")
    detector.enabled = True
    try:
        assert detector.detect("sawubona") == before, "SA languages unaffected"
    finally:
        detector.enabled = False


# --- The composite's arbitration rule -----------------------------------


def test_confident_primary_is_never_overruled(monkeypatch):
    """The eleven-language model is the authority for its own domain. A
    secondary answer must not displace a confident primary one, however
    sure the secondary sounds.
    """
    composite = CompositeLanguageDetector(
        primary=_Stub("isiZulu", 0.75), secondary=_Stub("swa", 0.99)
    )
    assert composite.detect("sawubona").language == "isiZulu"


def test_unsure_primary_defers_to_a_confident_secondary():
    composite = CompositeLanguageDetector(
        primary=_Stub("English", 0.20), secondary=_Stub("yor", 0.80)
    )
    assert composite.detect("some yoruba text").language == "yor"


def test_unsure_primary_keeps_its_answer_when_secondary_is_also_unsure():
    """Measured: the pan-African model scores 16-23% on South African and
    English input. Below the secondary threshold the primary answer stands,
    which is what stops English being relabelled as Nigerian Pidgin.
    """
    composite = CompositeLanguageDetector(
        primary=_Stub("English", 0.20), secondary=_Stub("pcm", 0.23)
    )
    assert composite.detect("hello there").language == "English"


# --- With the real bundle, if present ------------------------------------


@pytest.fixture
def real_composite():
    settings = Settings()
    pan = SklearnLanguageDetector(settings.pan_african_model_path)
    if not pan.is_available():
        pytest.skip("no pan-African bundle present")
    return CompositeLanguageDetector(primary=HeuristicLanguageDetector(), secondary=pan)


def test_bundle_labels_are_names_not_encoded_integers():
    settings = Settings()
    pan = SklearnLanguageDetector(settings.pan_african_model_path)
    if not pan.is_available():
        pytest.skip("no pan-African bundle present")
    result = pan.detect("Habari gani rafiki wangu")
    assert not result.language.isdigit(), (
        "the classifier trains on encoded labels; a bundle built without the "
        "label encoder reports languages called '0' and '7'"
    )
    # The detector now maps codes to display names, so check the property
    # this test exists for — that the label survived decoding — against the
    # mapping's own value set rather than against raw ISO codes.
    from bao.services.language_detector import PAN_AFRICAN_LANGUAGE_NAMES

    assert result.language in set(PAN_AFRICAN_LANGUAGE_NAMES.values())


@pytest.mark.parametrize("text,expected", SA_PROBES)
def test_enabling_it_does_not_change_south_african_detection(real_composite, text, expected):
    """The regression that would actually hurt: an isiZulu message routed
    to the pan-African model and returned as Swahili, taking the greeting
    row and the voice with it.
    """
    baseline = HeuristicLanguageDetector().detect(text)
    assert baseline.language == expected, "precondition: baseline unchanged"
    assert real_composite.detect(text).language == expected, (
        f"{text!r} changed from {expected} once the pan-African detector was enabled"
    )


def test_every_sa_language_is_covered_by_the_probes():
    covered = {expected for _, expected in SA_PROBES}
    missing = [label for label in LABELS if label not in covered]
    assert not missing, f"no probe for: {missing}"


def test_composite_detections_are_not_re_judged_by_the_reply_floor():
    """Two thresholds gated the same decision on incompatible scales.

    CompositeLanguageDetector accepts a pan-African answer above 0.35
    (calibrated: that model scores 16-23% on input that is not its language
    and 37-95% on input that is). The orchestrator then required 0.50 to act
    on any detection — calibrated for the LSTM, whose confidences are
    distributed differently.

    Result was a dead band: "sannu" was correctly identified as Hausa at
    43%, accepted by the detector, then discarded and answered in English.
    """
    from bao.bootstrap import build_orchestrator

    _, orchestrator = build_orchestrator()
    orchestrator.language_detector = _Stub("hau", 0.43, backend="sklearn")
    orchestrator.min_detection_confidence = 0.5
    result = orchestrator.handle("sannu", force_offline=True)
    assert result.reply_language == "hau", "a vouched-for detection must survive"


def test_the_floor_still_applies_to_the_primary_detector():
    """The exemption is for the composite's own gate, not a way around the
    floor generally — a weak LSTM guess must still be declined.
    """
    from bao.bootstrap import build_orchestrator

    _, orchestrator = build_orchestrator()
    orchestrator.language_detector = _Stub("Afrikaans", 0.42, backend="tflite")
    orchestrator.min_detection_confidence = 0.5
    result = orchestrator.handle("what is car in xhosa", force_offline=True)
    assert result.reply_language == "English"


def test_detector_reports_language_names_not_iso_codes():
    """The bundle's labels are ISO 639-3 codes, which is right for a label
    encoder and wrong for everything downstream: the badge read
    "Language: swa", and the Gemini system prompt read "Respond naturally
    in swa" — an instruction in a code rather than a language name.
    """
    settings = Settings()
    pan = SklearnLanguageDetector(settings.pan_african_model_path)
    if not pan.is_available():
        pytest.skip("no pan-African bundle present")

    result = pan.detect("jambo habari yako rafiki")
    assert result.language == "Swahili", f"got {result.language!r}"
    assert len(result.language) > 3, "an ISO code leaked through"


def test_every_bundle_label_has_a_display_name():
    """A retrain that adds a language must not silently start showing codes
    again, so the mapping is checked against the bundle rather than assumed.
    """
    from bao.services.language_detector import PAN_AFRICAN_LANGUAGE_NAMES

    settings = Settings()
    pan = SklearnLanguageDetector(settings.pan_african_model_path)
    if not pan.is_available():
        pytest.skip("no pan-African bundle present")

    import joblib

    labels = joblib.load(settings.pan_african_model_path)["labels"]
    missing = [c for c in labels if c not in PAN_AFRICAN_LANGUAGE_NAMES]
    assert not missing, f"no display name for: {missing}"
