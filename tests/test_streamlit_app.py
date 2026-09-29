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
    # The page starts the voice libraries loading on a background thread.
    # With them declared present but not actually installed, that thread's
    # import fails and switches the backend off again, at whatever moment
    # it happens to finish: in the Docker image it disabled the checkbox
    # between two steps of a test. Nothing here needs the real import.
    monkeypatch.setattr(speech, "preload_in_background", lambda: None)


@pytest.fixture
def page(voice_controls):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.sidebar.checkbox(key="speech_enabled").uncheck().run()
    return at


def _settings(at) -> dict:
    keys = ("speech_enabled", "stt_language", "pan_african_enabled", "reply_in")
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
    page.sidebar.selectbox(key="reply_in").set_value("Xitsonga").run()
    page.chat_input[0].set_value("Avuxeni").run()
    before = _settings(page)
    assert before["speech_enabled"] is False
    assert before["stt_language"] == "isiZulu"
    assert before["reply_in"] == "Xitsonga"

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


# --- "Reply in" --------------------------------------------------------------


def test_reply_in_offers_only_what_the_app_can_currently_detect():
    from bao.core.config import LABELS, PAN_AFRICAN_LABELS
    from bao.ui.streamlit_app import REPLY_AUTO, reply_language_options

    assert reply_language_options(False) == [REPLY_AUTO, *LABELS]
    assert reply_language_options(True) == [REPLY_AUTO, *LABELS, *PAN_AFRICAN_LABELS]


def test_automatic_means_no_override():
    from bao.ui.streamlit_app import REPLY_AUTO, reply_override

    assert reply_override(REPLY_AUTO) is None
    assert reply_override(None) is None       # a session from before the picker
    assert reply_override("Sesotho") == "Sesotho"


def test_the_badge_does_not_present_a_choice_as_a_detection():
    """"Language: Sesotho" would read as the detector's finding - on a
    greeting ("Dumela") the detector cannot actually tell apart.
    """
    from bao.ai.orchestrator import PipelineResult
    from bao.ui.streamlit_app import _format_detection_badge

    result = PipelineResult(
        text="Lumela! Nka o thusa jwang?", detected_language="Sesotho",
        confidence=1.0, detection_backend="override", source="knowledge_base",
        latency_ms=1.0, reply_language="Sesotho", language_was_overridden=True,
    )
    badge = _format_detection_badge(result)
    assert badge.startswith("Replying in Sesotho (chosen in the sidebar)")
    assert "Language:" not in badge and "%" not in badge


def _result(**overrides):
    from bao.ai.orchestrator import PipelineResult

    fields = dict(
        text="...", detected_language="Sesotho", confidence=1.0,
        detection_backend="override", source="gemini", latency_ms=1.0,
        reply_language="Sesotho", language_was_overridden=True,
    )
    fields.update(overrides)
    return PipelineResult(**fields)


def test_the_badge_says_when_the_message_overruled_the_sidebar():
    from bao.ui.streamlit_app import _format_detection_badge

    badge = _format_detection_badge(
        _result(reply_language="isiZulu", language_was_requested=True)
    )
    assert badge.startswith("Replying in isiZulu, as your message asks (sidebar: Sesotho)")
    assert "you asked for" not in badge, "said once, not twice"


def test_the_badge_does_not_claim_a_language_the_answer_is_not_in():
    """Live: "Replying in Sesotho" above a Xitsonga greeting. When a
    curated answer can only be served as written, the badge says so.
    """
    from bao.ui.streamlit_app import _format_detection_badge

    badge = _format_detection_badge(
        _result(source="knowledge_base", reply_language="Swahili",
                detected_language="Swahili", text_language="Xitsonga")
    )
    assert "this answer is in Xitsonga" in badge

    same = _format_detection_badge(_result(source="knowledge_base", text_language="Sesotho"))
    assert "this answer is in" not in same


def test_the_badge_names_a_backup_model():
    from bao.ui.streamlit_app import _format_detection_badge

    badge = _format_detection_badge(_result(fallback_model="gemini-3.5-flash-lite"))
    assert "gemini-3.5-flash-lite, because the main model was busy" in badge
    assert "busy" not in _format_detection_badge(_result())


def test_a_chosen_language_is_greeted_in_it_on_the_page(page):
    """The live report, driven through the real page: picker on Sesotho,
    "avuxeni" typed. Served from the curated rows, so no network is used.
    """
    page.sidebar.selectbox(key="reply_in").set_value("Sesotho").run()
    page.chat_input[0].set_value("avuxeni").run()

    assert not page.exception, [str(e.value) for e in page.exception]
    reply = [m for m in page.chat_message if m.name == "assistant"][-1]
    assert reply.markdown[0].value.startswith("Lumela!")
    assert "this answer is in" not in reply.caption[0].value


@pytest.mark.parametrize(("choice", "greeting"), [
    ("Sepedi", "Thobela!"),
    ("Sesotho", "Lumela!"),
    ("Setswana", "Dumela!"),
])
def test_reply_in_settles_a_greeting_three_languages_share(page, choice, greeting):
    """The case the picker exists for. "Dumela" is a greeting in Sepedi,
    Sesotho and Setswana, so no detector can know which the visitor
    speaks; the picker lets them say. Each language has its own curated
    row, so the answer proves the choice reached retrieval, not just the
    badge.
    """
    page.sidebar.selectbox(key="reply_in").set_value(choice).run()
    page.chat_input[0].set_value("Dumela").run()

    assert not page.exception, [str(e.value) for e in page.exception]
    reply = [m for m in page.chat_message if m.name == "assistant"][-1]
    assert reply.markdown[0].value.startswith(greeting)
    assert f"Replying in {choice} (chosen in the sidebar)" in reply.caption[0].value


def test_a_choice_that_is_no_longer_offered_falls_back_to_automatic(page):
    """Turning the 14 extra languages off removes them from the picker.
    A visitor who had chosen one must land on automatic rather than keep
    a language the app has just stopped offering. Streamlit 1.59.2 does
    this by itself, so there is no code of ours to test here - this pins
    the behaviour, so a Streamlit release that changes it fails CI instead
    of the demo.
    """
    toggles = [c for c in page.sidebar.checkbox if c.key == "pan_african_enabled"]
    if not toggles:
        pytest.skip("pan-African bundle not present")
    toggles[0].check().run()
    page.sidebar.selectbox(key="reply_in").set_value("Swahili").run()
    assert page.session_state["reply_in"] == "Swahili"

    page.sidebar.checkbox(key="pan_african_enabled").uncheck().run()

    assert not page.exception, [str(e.value) for e in page.exception]
    from bao.ui.streamlit_app import REPLY_AUTO
    assert page.session_state["reply_in"] == REPLY_AUTO
    assert "Swahili" not in page.sidebar.selectbox(key="reply_in").options
