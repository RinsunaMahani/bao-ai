"""Shared fixtures.

The speech layer keeps three pieces of module-global state — which local
checkpoints are registered, whether the multilingual SA VITS model is on,
and which speaker it uses. They are global because voice coverage is a
property of the machine rather than of one orchestrator, and because the
UI has to be able to ask "what can be spoken?" without holding a pipeline.

That is a reasonable design and a hostile one to test: `build_orchestrator()`
writes all three from config.toml, so the first test that builds a pipeline
silently reconfigures every test after it. Enabling the VITS model in
config.toml turned 24 unrelated tests red for exactly this reason — they
were asserting that a language has no voice, which stopped being true
halfway through the session.

Resetting around every test makes each one describe the state it sets up,
and keeps the suite's result independent of config.toml. A test that wants
these switches on turns them on itself.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_speech_globals():
    """Snapshots and restores the speech module's global switches."""
    import bao.services.speech as speech

    saved_voices = dict(speech.LOCAL_VOICE_MODELS)
    saved_enabled = speech._COQUI_ENABLED
    saved_speaker = speech.COQUI_SA_SPEAKER
    try:
        yield
    finally:
        speech.LOCAL_VOICE_MODELS.clear()
        speech.LOCAL_VOICE_MODELS.update(saved_voices)
        speech._COQUI_ENABLED = saved_enabled
        speech.COQUI_SA_SPEAKER = saved_speaker


@pytest.fixture
def default_settings(tmp_path):
    """Settings as a fresh install sees them, ignoring this machine's
    config.toml.

    Tests about DEFAULTS have to read the code's defaults, not whatever the
    local deployment happens to have switched on. Pointing Settings at a
    path that does not exist leaves every value at its fallback, which is
    exactly the question those tests are asking.
    """
    from bao.core.config import Settings

    return Settings(str(tmp_path / "absent-config.toml"))
