"""The web app, driven as a user drives it.

Everything else in this suite tests the pipeline behind the page. Nothing
rendered the page itself, so UI changes were checked only by compiling
them — and a bug that only exists in how Streamlit runs a script cannot be
seen that way.

That is not hypothetical. "New conversation" called st.rerun() from inside
its handler, which aborted the run before the settings further down the
sidebar were drawn, and Streamlit drops the state of widgets a run never
reaches. So the button silently switched "Speak replies" back on and reset
"I'll speak in" — the settings it promised to keep. Its unit test passed,
because it used a plain dict; AppTest, which runs the real script, caught
it on the first try.

Version note, measured rather than assumed: the widget-state loss happens
on Streamlit 1.59, the version on the machine this app is presented from,
and not on 1.64, which CI installs. So the settings tests below catch that
regression on 1.59 and simply pass on 1.64 either way. The callback fix
is correct on both, and is Streamlit's own recommended pattern.

These tests need no network. The questions are answered from the curated
knowledge base, and speech is switched off before anything is asked.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

# Absolute, built from this file. AppTest resolved a relative path against
# the working directory in Streamlit 1.59 and against the CALLING FILE in
# 1.64 - so "bao/ui/streamlit_app.py" worked locally and pointed at
# tests/bao/ui/... under the Streamlit CI installs, failing every test here.
APP = str(Path(__file__).resolve().parent.parent / "bao" / "ui" / "streamlit_app.py")
# The first run builds the whole pipeline (classifier, knowledge base,
# detector bundle), which takes several seconds on a cold cache.
TIMEOUT = 300


@pytest.fixture
def voice_controls(monkeypatch):
    """Makes the voice settings interactive whatever is installed.

    "Speak replies" is disabled, and "I'll speak in" not drawn at all, on a
    machine without speech libraries - CI among them. Streamlit 1.64's
    AppTest refuses to change a disabled widget (1.59 allowed it), so the
    settings tests either errored or, worse, could only exercise the one
    setting that survived the original bug by coincidence. Declaring the
    backends present lets every environment check every setting. Nothing
    is synthesized: speech is switched off before anything is asked.
    """
    import bao.services.speech as speech

    monkeypatch.setattr(speech, "_HAS_EDGE_BACKEND", True)
    monkeypatch.setattr(speech, "_HAS_STT_BACKEND", True)


@pytest.fixture
def page(voice_controls):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.sidebar.checkbox(key="speech_enabled").uncheck().run()
    return at


def _settings(at) -> dict:
    keys = ("speech_enabled", "stt_language", "pan_african_enabled")
    return {k: at.session_state[k] for k in keys if k in at.session_state}


def test_the_page_renders_without_errors(page):
    assert not page.exception
    assert any(b.label == "New conversation" for b in page.sidebar.button)


def test_a_greeting_is_answered_in_its_own_language(page):
    page.chat_input[0].set_value("Avuxeni").run()

    assert not page.exception, [str(e.value) for e in page.exception]
    replies = [m for m in page.chat_message if m.name == "assistant"]
    assert replies, "no assistant reply was rendered"
    assert "Avuxeni" in replies[-1].markdown[0].value
    badges = [c.value for c in page.caption if c.value.startswith("Language:")]
    assert badges and "Xitsonga" in badges[-1]


def test_new_conversation_keeps_the_visitors_settings(page):
    """The regression this file exists for."""
    page.sidebar.selectbox(key="stt_language").set_value("isiZulu").run()
    page.chat_input[0].set_value("Avuxeni").run()
    before = _settings(page)
    assert before["speech_enabled"] is False
    assert before["stt_language"] == "isiZulu"

    [b for b in page.sidebar.button if b.label == "New conversation"][0].click().run()

    assert not page.exception, [str(e.value) for e in page.exception]
    assert _settings(page) == before, "a new conversation must not reset settings"
    assert not page.chat_message, "the transcript should be empty"
    assert len(page.session_state["orchestrator"].memory) == 0


def test_clearing_uploads_keeps_the_visitors_settings(page):
    page.session_state["orchestrator"].document_retriever.add_document(
        "notes.txt", "Registration opens in January.")
    page.run()
    before = _settings(page)

    [b for b in page.sidebar.button if b.label == "Clear uploaded documents"][0].click().run()

    assert not page.exception, [str(e.value) for e in page.exception]
    assert len(page.session_state["orchestrator"].document_retriever) == 0
    assert _settings(page) == before


def test_the_page_survives_a_missing_logo(monkeypatch, voice_controls):
    """Every Docker image excluded docs/, where the logo lives, and
    Streamlit raised MediaFileStorageError on the first page load - so the
    containerised app never rendered at all. The logo is decoration; its
    absence must not take the page down.
    """
    import bao.core.config as config

    monkeypatch.setattr(config, "ASSISTANT_LOGO_PATH", "/app/docs/assets/absent-logo.jpg")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.checkbox(key="speech_enabled").uncheck().run()
    at.chat_input[0].set_value("Avuxeni").run()   # exercises the chat avatar too

    assert not at.exception, [str(e.value) for e in at.exception]
    assert [m for m in at.chat_message if m.name == "assistant"]
