"""The pan-African detector must be invisible until deliberately enabled.

Adding a second detector changes the detection path for EVERY message, not
only the fourteen new languages — detection feeds greeting lookup,
retrieval language preference and voice selection. So the property that
matters most is not "does it work" but "does it change nothing when off".
"""

from pathlib import Path

import pytest

from bao.bootstrap import build_orchestrator
from bao.core.config import LABELS, Settings
from bao.services.language_detector import (
    CompositeLanguageDetector,
    DetectionResult,
    HeuristicLanguageDetector,
    SklearnLanguageDetector,
    get_language_detector,
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


def test_disabled_by_default(default_settings):
    """Asserts the DEFAULT, not this machine's config.toml.

    A deployment that switches the pan-African languages on is a choice,
    and it must not make the suite claim the shipped default changed. So
    this reads Settings with no config file present, which is what a fresh
    clone sees.
    """
    assert default_settings.pan_african_enabled is False


def test_default_build_leaves_the_composite_inert(default_settings):
    """The composite is now constructed whenever a bundle exists, so the UI
    can toggle it without rebuilding the pipeline. The guarantee therefore
    moves from "not wrapped" to "wrapped but inert" — which has to be
    checked behaviourally rather than by type.
    """
    _, orchestrator = build_orchestrator(default_settings)
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


def test_confident_primary_beats_a_MODERATE_secondary(monkeypatch):
    """The eleven-language model is the authority for its own domain, so a
    middling secondary answer must not displace a confident primary one.

    This is the half of the original rule that survived measurement. The
    other half did not - see the test below.
    """
    composite = CompositeLanguageDetector(
        primary=_Stub("isiZulu", 0.75), secondary=_Stub("swa", 0.45)
    )
    assert composite.detect("sawubona").language == "isiZulu"


def test_confident_primary_IS_overruled_by_a_strong_secondary():
    """The rule this replaces said a confident primary is never overruled.
    That was wrong, and measurably so.

    The primary is an 11-class softmax with no "none of the above" output.
    Input in a language it was never trained on does not make it hesitate;
    it makes it pick the nearest of its eleven and report high confidence.
    Measured, one sentence per language:

        Luganda -> Xitsonga 100%     Yoruba -> Xitsonga 99%
        Hausa   -> English   97%     Nigerian Pidgin -> English 85%

    Every one of those cleared primary_min, so the secondary was never
    consulted and seven of the fourteen languages were answered in a
    language the user had not written in. Primary confidence simply does
    not carry the information the gate needed.

    Secondary confidence does: across 24 South African inputs it never
    exceeds 44%, and where it is right about a pan-African language it
    scores 43-89%. So a secondary above secondary_strong outranks the
    primary however sure the primary sounds.
    """
    composite = CompositeLanguageDetector(
        primary=_Stub("Xitsonga", 1.00), secondary=_Stub("lug", 0.87)
    )
    assert composite.detect("some luganda text").language == "lug"


def test_south_african_input_never_trips_the_strong_secondary_rule():
    """The safety property the new rule has to keep.

    44% is the highest the pan-African model scored on ANY South African
    input tested - English read as Nigerian Pidgin, the one genuinely
    confusable pair, since Nigerian Pidgin is English-lexifier. The
    threshold sits above it so English stays English.
    """
    composite = CompositeLanguageDetector(
        primary=_Stub("English", 0.99), secondary=_Stub("pcm", 0.44)
    )
    assert composite.detect("hello how can you help me").language == "English"


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


@pytest.fixture(scope="module")
def real_composite():
    """Module-scoped for the same reason as full_composite: the bundle is
    9 MB and this feeds a thirteen-case parametrised test.
    """
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
    # No language named in the question: "what is car in xhosa" used to be
    # the example here, and now correctly asks for isiXhosa outright.
    result = orchestrator.handle("what is a car", force_offline=True)
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


# --- The seven languages the old gate hid --------------------------------

# Full sentences, because the primary is an LSTM trained on NCHLT news
# sentences. Kept alongside scripts/probe_language_routing.py, which walks
# the same inputs through the whole orchestrator; these pin the detection
# half in CI, where building an orchestrator per language is too slow.
_PREVIOUSLY_MISROUTED = [
    ("French", "Bonjour, pouvez-vous m'aider à trouver des informations sur la santé."),
    ("Hausa", "Sannu, ina neman taimako game da harkokin lafiya a yankinmu."),
    ("Igbo", "Ndewo, biko nyere m aka banyere ozi gbasara ahụike na obodo anyị."),
    ("Nigerian Pidgin", "How you dey, abeg I wan know about health care for our area."),
    ("Swahili", "Habari yako, naomba msaada kuhusu huduma za afya katika eneo letu."),
    ("Yoruba", "Bawo ni, jọwọ ran mi lọwọ nipa alaye nipa ilera ni agbegbe wa."),
]


@pytest.fixture(scope="module")
def full_composite():
    """The real classifier paired with the real pan-African bundle.

    Module-scoped because both are expensive to construct — a 9 MB joblib
    bundle and a TFLite interpreter — and the parametrised tests below
    would otherwise rebuild both once per case. Safe to share: detection is
    stateless, nothing here mutates the detectors.
    """
    settings = Settings()
    pan = SklearnLanguageDetector(settings.pan_african_model_path)
    if not pan.is_available():
        pytest.skip("no pan-African bundle present")
    primary = get_language_detector(
        prefer_ml=True,
        model_path=settings.classifier_model_path,
        tokenizer_config_path=settings.tokenizer_config_path,
    )
    return CompositeLanguageDetector(primary=primary, secondary=pan)


@pytest.mark.parametrize("expected,text", _PREVIOUSLY_MISROUTED)
def test_languages_hidden_by_the_old_primary_gate_are_detected(
    expected, text, full_composite
):
    """Each of these was answered in a South African language.

    The primary was confidently wrong on all of them - Yoruba read as
    Xitsonga at 99%, Hausa as English at 97% - so it never fell below
    primary_min and the pan-African model was never consulted. The bug was
    not that the secondary was inaccurate; it was right about 14 of 14. The
    bug was that it was not asked.
    """
    assert full_composite.detect(text).language == expected


def test_luganda_is_a_known_miss_and_stays_documented():
    """Luganda scores 43%, just under the 44% ceiling that English hits as
    Nigerian Pidgin. Admitting it means relabelling English, which is the
    worse trade for an app whose lingua franca is English.

    Pinned as a KNOWN miss rather than left silent, so that if a better
    bundle ever lifts it above the bar this test fails and says so - a
    limitation that stops being true should not go unnoticed.
    """
    settings = Settings()
    pan = SklearnLanguageDetector(settings.pan_african_model_path)
    if not pan.is_available():
        pytest.skip("no pan-African bundle present")
    result = pan.detect("Oli otya, nsaba obuyambi ku bikwata ku by'obulamu mu kitundu kyaffe.")
    assert result.language == "Luganda", "the secondary identifies it correctly"
    assert result.confidence < 0.5, (
        "Luganda now clears secondary_strong - remove it from the known-miss "
        "list in the CompositeLanguageDetector docstring and from this test"
    )


def test_the_bundle_is_read_by_the_scikit_learn_series_that_saved_it():
    """A joblib bundle records the scikit-learn that pickled it. Across
    MINOR versions a pickled estimator is not guaranteed to load
    faithfully, and when it does not, it fails silently rather than
    raising.

    Patch differences are allowed on purpose. scikit-learn warns on any
    difference, and the earlier version of this test failed on all of
    them - which would have turned CI red every time a 1.9.x patch
    shipped, since CI installs the newest one. The real risk is the one
    CI's own install log showed: on Python 3.10, pip can only get 1.7.2.

    If this fails, install scikit-learn from the pinned range in
    requirements.txt, or re-save the bundle and re-run these tests:

        python scripts/build_pan_african_bundle.py --reexport
    """
    import warnings

    import joblib

    from bao.services.language_detector import sklearn_minor_mismatches

    path = Settings().pan_african_model_path
    if not Path(path).exists():
        pytest.skip("no pan-African bundle present")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joblib.load(path)

    assert not sklearn_minor_mismatches(caught), sklearn_minor_mismatches(caught)


def _version_warning(pickled, installed):
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(InconsistentVersionWarning(
            estimator_name="Pipeline", current_sklearn_version=installed,
            original_sklearn_version=pickled))
    return caught


def test_a_patch_difference_is_not_treated_as_a_fault():
    """Deterministic, whatever scikit-learn happens to be installed."""
    from bao.services.language_detector import sklearn_minor_mismatches

    assert sklearn_minor_mismatches(_version_warning("1.9.0", "1.9.1")) == []


def test_a_minor_difference_is():
    from bao.services.language_detector import sklearn_minor_mismatches

    assert sklearn_minor_mismatches(_version_warning("1.9.0", "1.7.2")) == [("1.9.0", "1.7.2")]
    assert sklearn_minor_mismatches(_version_warning("1.9.0", "2.0.0")) == [("1.9.0", "2.0.0")]