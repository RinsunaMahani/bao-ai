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
