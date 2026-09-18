"""The voice fallback tiers.

Four of eleven languages have a real voice. What happens for the other
seven is a design decision, and these pin it — including the three
substitutions that were proposed and rejected, because a rejected option
that isn't recorded gets proposed again.
"""

import pytest

import bao.services.speech as speech_module
from bao.services.speech import (
    RELATED_LANGUAGE_VOICES,
    resolve_voice_language,
    select_backend,
    voice_coverage,
)


@pytest.fixture(autouse=True)
def both_backends(monkeypatch):
    """Test the policy, not this machine's installed packages."""
    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", True)
    monkeypatch.setattr(speech_module, "_HAS_MMS_BACKEND", True)


# --- Tier 1 -------------------------------------------------------------


@pytest.mark.parametrize("language", ["English", "isiZulu", "Afrikaans", "Xitsonga"])
def test_languages_with_a_real_voice_use_it_and_say_nothing(language):
    spoken, note = resolve_voice_language(language)
    assert spoken == language
    assert note is None, "a native voice needs no explanation"


# --- Tier 4 is the default ----------------------------------------------


@pytest.mark.parametrize("language", [
    "isiXhosa", "Sesotho", "Setswana", "Sepedi", "Tshivenda", "siSwati", "isiNdebele",
])
def test_languages_without_a_voice_are_silent_by_default(language):
    """Off by default because a substitution is an approximation, and a
    silent approximation is worse than no audio.
    """
    spoken, note = resolve_voice_language(language)
    assert spoken is None
    assert language in note


# --- Tier 2, and what was deliberately excluded from it ------------------


@pytest.mark.parametrize("language", ["isiXhosa", "siSwati", "isiNdebele"])
def test_nguni_languages_may_borrow_the_isizulu_voice(language):
    spoken, note = resolve_voice_language(language, allow_related=True)
    assert spoken == "isiZulu"
    assert "not the same" in note, "the substitution must be disclosed"


@pytest.mark.parametrize("language", ["Sesotho", "Setswana", "Sepedi"])
def test_sotho_tswana_has_no_related_fallback(language):
    """A proposed table routed these to each other. All three return 404 on
    Hugging Face, so there is no voice anywhere in the family to borrow —
    the rows were unimplementable, not merely imperfect.
    """
    assert language not in RELATED_LANGUAGE_VOICES
    spoken, _ = resolve_voice_language(language, allow_related=True)
    assert spoken is None


def test_tshivenda_does_not_borrow_the_xitsonga_voice():
    """Also proposed, and rejected on linguistic grounds: Venda is its own
    branch of Bantu, Tsonga is Tswa-Ronga, and Venda orthography uses dental
    diacritics (ṱ ḓ ṋ ḽ) that Tsonga lacks entirely. A character-level VITS
    model mangles characters it never saw during training.
    """
    assert "Tshivenda" not in RELATED_LANGUAGE_VOICES
    spoken, _ = resolve_voice_language("Tshivenda", allow_related=True)
    assert spoken is None


# --- Tier 3 --------------------------------------------------------------


def test_english_fallback_is_opt_in_and_disclosed():
    spoken, note = resolve_voice_language("Tshivenda", allow_english=True)
    assert spoken == "English"
    assert "spoken in English" in note


def test_related_is_preferred_over_english():
    """A closely-related language is a better approximation than a
    completely unrelated one, so Tier 2 must be tried before Tier 3.
    """
    spoken, _ = resolve_voice_language("isiXhosa", allow_related=True, allow_english=True)
    assert spoken == "isiZulu"


# --- Reporting -----------------------------------------------------------


def test_coverage_reports_every_language():
    from bao.core.config import LABELS

    coverage = voice_coverage()
    assert set(coverage) == set(LABELS)
    assert sum(1 for tier in coverage.values() if tier == "native") == 4


# --- Coverage and synthesis must agree ----------------------------------


@pytest.mark.parametrize("with_local_checkpoint", [False, True])
def test_coverage_never_claims_a_tier_synthesis_will_not_honour(
    tmp_path, with_local_checkpoint
):
    """The sidebar may not promise audio the audio path will not produce.

    This is the regression that motivated deriving voice_coverage() from
    resolve_voice_language() instead of re-deriving the tiers from the same
    tables. The two were independent walks over overlapping data and they
    drifted: coverage consulted LOCAL_VOICE_MODELS and
    related_language_voices(), the resolver consulted neither. Registering
    a locally trained Sepedi checkpoint made the interface report
    "Sepedi — native (locally trained)" and "Sesotho — related (Sepedi)"
    while speak() returned no audio for either.

    Parametrised over the checkpoint because the two walks AGREED when
    nothing was registered — which is why the whole existing suite passed
    while the feature was broken. The disagreement only appears in the
    state the feature exists for.
    """
    from bao.core.config import LABELS

    original = dict(speech_module.LOCAL_VOICE_MODELS)
    try:
        if with_local_checkpoint:
            checkpoint = tmp_path / "sepedi_vits"
            checkpoint.mkdir()
            speech_module.load_local_voices({"Sepedi": str(checkpoint)})
        else:
            speech_module.load_local_voices({})

        coverage = voice_coverage()
        for language in LABELS:
            tier = coverage[language]
            native, _ = resolve_voice_language(language)
            related, _ = resolve_voice_language(language, allow_related=True)

            if tier.startswith("native"):
                assert native == language, (
                    f"coverage calls {language} {tier!r} but the resolver "
                    f"returns {native!r} for it"
                )
            elif tier.startswith("related"):
                assert related is not None and related != language, (
                    f"coverage offers {language} a related voice but the "
                    f"resolver returns {related!r}"
                )
                assert f"related ({related})" == tier
            else:
                assert tier == "text only"
                assert related is None, (
                    f"coverage calls {language} text-only but the resolver "
                    f"would speak it with {related!r}"
                )
    finally:
        speech_module.load_local_voices(original)


def test_a_registered_checkpoint_reaches_the_synthesis_path(tmp_path):
    """Registration has to change what synthesis DOES, not only what the
    sidebar says.

    Sepedi has no edge locale and mms-tts-nso 404s, so before a checkpoint
    is registered every layer agrees there is no voice. After, all three
    must agree there is: the resolver speaks it natively, the backend
    router sends it to the VITS engine, and coverage reports it as locally
    trained.
    """
    original = dict(speech_module.LOCAL_VOICE_MODELS)
    try:
        speech_module.load_local_voices({})
        assert resolve_voice_language("Sepedi")[0] is None
        assert select_backend("Sepedi", "auto") == "mms"  # engine exists, weights do not

        checkpoint = tmp_path / "sepedi_vits"
        checkpoint.mkdir()
        speech_module.load_local_voices({"Sepedi": str(checkpoint)})

        assert resolve_voice_language("Sepedi") == ("Sepedi", None)
        assert select_backend("Sepedi", "auto") == "mms"
        assert voice_coverage()["Sepedi"] == "native (locally trained)"
    finally:
        speech_module.load_local_voices(original)


# --- Pan-African voices --------------------------------------------------


def test_the_pan_african_languages_that_have_a_voice_get_one():
    """Twelve of the fourteen extra languages have an MMS voice, and the
    app mapped none of them before: config.toml's mms_codes table listed
    only the eleven South African languages, so every pan-African reply was
    silent no matter what was installed.

    Codes probed against Hugging Face rather than inferred - Oromo is
    "orm", not the "gaz" its ISO 639-3 macrolanguage member suggests.
    """
    from bao.core.config import PAN_AFRICAN_LABELS, Settings

    codes = Settings().mms_codes
    coverage = voice_coverage(codes, languages=PAN_AFRICAN_LABELS)

    spoken = {lang for lang, tier in coverage.items() if tier.startswith("native")}
    assert spoken == set(PAN_AFRICAN_LABELS) - {"Igbo", "Lingala"}


@pytest.mark.parametrize("language", ["Igbo", "Lingala"])
def test_igbo_and_lingala_report_no_voice_rather_than_failing_late(language):
    """mms-tts-ibo and mms-tts-lin 404, as do the alternative codes tried.

    Recorded as an absent voice so the interface says so up front, instead
    of the request failing at synthesis time with an error that reads like
    a bug. A language with no voice is a fact about the world; a language
    that looks supported and then produces nothing is a defect.
    """
    from bao.core.config import Settings

    assert Settings().mms_codes.get(language) is None
    spoken, note = resolve_voice_language(language, Settings().mms_codes)
    assert spoken is None
    assert language in note


def test_edge_voices_are_preferred_where_microsoft_has_the_locale():
    """Four of the fourteen have a real Microsoft neural voice. Those must
    route to edge rather than MMS, for the same reason English does: a
    native neural voice beats a generic multilingual one.
    """
    for language in ("Amharic", "French", "Somali", "Swahili"):
        assert select_backend(language, "auto") == "edge", language
