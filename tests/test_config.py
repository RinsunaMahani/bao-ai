from bao.core.config import LABELS, Settings


def test_mms_codes_cover_every_canonical_label():
    """Regression test for a real bug found via review: config.toml's
    mms_codes/stt_codes keys ("IsiZulu", "SiSwati", etc.) previously used
    different capitalization than LABELS ("isiZulu", "siSwati"). Since
    services/speech.py looks languages up with a plain `dict.get()`, that
    mismatch was a silent lookup miss — not a crash — that fell back to
    English audio for 5 of the 11 languages regardless of what was
    actually detected. Confirmed by testing the actual dict lookup before
    fixing, not just eyeballing the casing.
    """
    settings = Settings()
    mms_codes = settings.mms_codes
    stt_codes = settings.stt_codes

    missing_from_mms = [lang for lang in LABELS if lang not in mms_codes]
    missing_from_stt = [lang for lang in LABELS if lang not in stt_codes]

    assert not missing_from_mms, f"Languages missing a working MMS/TTS code: {missing_from_mms}"
    assert not missing_from_stt, f"Languages missing a working STT code: {missing_from_stt}"


def test_default_mms_and_stt_codes_also_cover_every_label():
    """Same check against the in-code fallback maps (used when config.toml
    is missing or doesn't define [languages]), so both sources of truth
    are pinned, not just the one that happens to be loaded right now.
    """
    from bao.core.config import DEFAULT_MMS_CODES, DEFAULT_STT_CODES

    assert all(lang in DEFAULT_MMS_CODES for lang in LABELS)
    assert all(lang in DEFAULT_STT_CODES for lang in LABELS)


def test_english_maps_to_a_voice_that_exists():
    """DEFAULT_MMS_CODES mapped English to "afr" to give it a local accent
    by routing English through Afrikaans letter-to-sound rules.

    facebook/mms-tts-afr does not exist, so that mapping silently removed
    English from the offline path entirely. config.toml was corrected and
    this default was not, which left every caller that does not pass an
    explicit table - and any install without a config.toml - fetching a
    404 and getting no audio.
    """
    from bao.core.config import DEFAULT_MMS_CODES
    from bao.services.speech import _KNOWN_MMS_VOICES

    assert DEFAULT_MMS_CODES["English"] == "eng"
    assert DEFAULT_MMS_CODES["English"] in _KNOWN_MMS_VOICES


def test_the_shipped_config_and_the_code_default_agree_on_english():
    """The two drifted once and nothing noticed. A mismatch here means one
    of them is wrong and the app behaves differently depending on whether
    config.toml is present.
    """
    from bao.core.config import DEFAULT_MMS_CODES, Settings

    assert Settings().mms_codes["English"] == DEFAULT_MMS_CODES["English"]
