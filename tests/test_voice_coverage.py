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


# --- The multilingual South African VITS model ---------------------------


@pytest.fixture
def coqui_on(monkeypatch):
    """Pretends coqui-tts is installed and the model is switched on.

    The real thing needs a gated 150 MB download, so what is testable here
    is the routing: which languages it claims, and which it must not take
    from a better voice. Audio quality is not testable anywhere in this
    suite and is not claimed to be.
    """
    monkeypatch.setattr(speech_module, "_HAS_COQUI_BACKEND", True)
    monkeypatch.setattr(speech_module, "_COQUI_ENABLED", True)


def test_it_is_off_unless_configured(default_settings):
    """Off by default for a licence reason, not a technical one: the model
    is cc-by-nc-4.0 and this repository is MIT, so turning it on makes a
    deployment non-commercial. That is a choice someone has to make.

    Reads the default rather than this machine's config.toml, which may
    legitimately have made that choice — see tests/conftest.py.
    """
    assert default_settings.coqui_sa_enabled is False
    assert default_settings.coqui_sa_speaker is None


def test_asking_for_it_without_the_library_leaves_it_off(monkeypatch):
    """A switch that reports success while doing nothing is the failure
    this project keeps finding. Enabling without coqui-tts installed must
    return False, not True.
    """
    monkeypatch.setattr(speech_module, "_HAS_COQUI_BACKEND", False)
    monkeypatch.setattr(speech_module, "_COQUI_ENABLED", False)
    assert speech_module.enable_coqui_sa(True) is False
    assert speech_module.has_coqui_backend() is False


@pytest.mark.parametrize("language", [
    "isiXhosa", "Sesotho", "Setswana", "Sepedi", "Tshivenda", "siSwati", "isiNdebele",
])
def test_it_covers_exactly_the_languages_nothing_else_reaches(coqui_on, language):
    """These seven are the whole reason for the model: edge-tts has three
    South African locales, MMS has one, and this has all eleven.
    """
    spoken, note = resolve_voice_language(language)
    assert spoken == language
    assert note is None, "a voice in the user's own language needs no apology"
    assert select_backend(language, "auto") == "coqui"


@pytest.mark.parametrize("language", ["English", "Afrikaans", "isiZulu"])
def test_it_does_not_displace_microsofts_voices(coqui_on, language):
    """Microsoft's neural voices are better than a community NCHLT model
    for the three languages both cover. The model's job is the seven
    neither covers, not every language it happens to list.
    """
    assert select_backend(language, "auto") == "edge"


def test_it_outranks_borrowing_a_related_languages_voice(coqui_on):
    """isiXhosa was spoken with the isiZulu voice, which is Nguni and
    shares the click letters but is still a different language being
    mispronounced. A real isiXhosa voice must win, and must do so without
    the disclosure note a substitution carries.
    """
    borrowed, note = resolve_voice_language("isiXhosa", allow_related=True)
    assert borrowed == "isiXhosa", "an own-language voice beats a borrowed one"
    assert note is None


def test_xitsonga_keeps_its_verified_mms_voice(coqui_on):
    """Xitsonga is the one South African language with a real MMS repo.
    A verified voice is not replaced by an unevaluated one.
    """
    assert select_backend("Xitsonga", "auto") == "mms"


# --- Characters the SA VITS checkpoint cannot represent -------------------


@pytest.mark.parametrize("raw,expected", [
    # Sepedi: dropping the s-caron turns "thusho" into "thuo".
    ("thušo", "thusho"),
    ("tša sekolo", "tsha sekolo"),
    # Tshivenda dental consonants, which vanished entirely.
    ("ṱoḓa", "toda"),
    ("ḽa na ṋea", "la na nea"),
    # Sesotho long vowels.
    ("ōē", "oe"),
])
def test_unrepresentable_characters_are_folded_not_dropped(raw, expected):
    """Coqui discards a symbol outside its vocabulary, which is the worst
    option available: it does not fail, it changes the word.

    The checkpoint's 138-symbol vocabulary has no š, no ṱ/ḓ/ṋ/ḽ and no
    ō/ē, so three of the languages it exists to serve were being
    mispronounced by the model meant to fix their silence.
    """
    from bao.services.speech import fold_to_coqui_vocabulary

    assert fold_to_coqui_vocabulary(raw) == expected


def test_folding_leaves_ordinary_text_untouched():
    """It must not become a general text mangler: every other language
    here is written in characters the model already has.
    """
    from bao.services.speech import fold_to_coqui_vocabulary

    for text in ("Molo, ndicela uncedo", "Sawubona, unjani na", "Goeiedag almal"):
        assert fold_to_coqui_vocabulary(text) == text
