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
