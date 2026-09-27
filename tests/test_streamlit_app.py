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

These tests need no network. The questions are answered from the curated
knowledge base, and speech is switched off before anything is asked.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "bao/ui/streamlit_app.py"
# The first run builds the whole pipeline (classifier, knowledge base,
# detector bundle), which takes several seconds on a cold cache.
TIMEOUT = 300


@pytest.fixture
def page():
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
    speak_in = [s for s in page.sidebar.selectbox if s.key == "stt_language"]
    if speak_in:  # only drawn when speech recognition is installed
        speak_in[0].set_value("isiZulu").run()
    page.chat_input[0].set_value("Avuxeni").run()
    before = _settings(page)
    assert before["speech_enabled"] is False

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
