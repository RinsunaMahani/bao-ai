"""Transient provider failures must not cost the user their turn.

Observed live on gemini-3.6-flash: a question came back as "I ran into a
problem generating a response just now" while the identical request
succeeded moments later. The API had answered

    503 UNAVAILABLE - this model is currently experiencing high demand

which is the provider saying "not now", not "not ever". The client treated
it as fatal and the turn was lost.
"""
from __future__ import annotations

import pytest

from bao.ai.client import GeminiClient, _is_transient
from bao.core.config import Settings
from bao.core.exceptions import GenerationError, GenerationUnavailableError


class _ApiError(Exception):
    """Stands in for google.genai.errors.APIError, which carries `.code`."""

    def __init__(self, code: int, message: str = "boom"):
        super().__init__(f"{code} {message}")
        self.code = code


class _Timeout(Exception):
    """A network failure has no status code; the type name is the signal."""


# --- which failures are worth retrying ----------------------------------


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_provider_side_failures_are_transient(code):
    assert _is_transient(_ApiError(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_request_side_failures_are_not(code):
    """A malformed or unauthorised request fails the same way every time.
    Retrying it just makes the user wait longer for the same answer.
    """
    assert not _is_transient(_ApiError(code))


def test_network_errors_are_transient_without_a_status_code():
    assert _is_transient(_Timeout("read timed out"))


def test_status_code_is_read_not_pattern_matched():
    """The message is prose and can be reworded upstream at any time; the
    code is the contract. A 400 whose text happens to mention 503 must
    still be treated as permanent.
    """
    assert not _is_transient(_ApiError(400, "not a 503, genuinely your fault"))


# --- the retry itself ----------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    c = GeminiClient.__new__(GeminiClient)
    c.settings = Settings()
    c.model_name = "test-model"
    c.max_attempts = 3
    # Set explicitly rather than inherited from config.toml: a deployment
    # changing its latency tuning must not change what these tests assert.
    c.thinking_level = "MINIMAL"
    c.client = object()
    monkeypatch.setattr(GeminiClient, "_sleep_before_retry", lambda self, attempt: None)
    return c


def _responses(client, monkeypatch, outcomes):
    """Drives models.generate_content through a scripted list of outcomes."""
    calls = {"n": 0}

    class _Models:
        def generate_content(self, **kwargs):
            outcome = outcomes[calls["n"]]
            calls["n"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return type("R", (), {"text": outcome})()

    monkeypatch.setattr(client, "client", type("C", (), {"models": _Models()})())
    return calls


def test_a_503_is_retried_and_the_turn_survives(client, monkeypatch):
    calls = _responses(client, monkeypatch, [_ApiError(503), "the real answer"])
    assert client.generate("hello") == "the real answer"
    assert calls["n"] == 2, "should have retried exactly once"


def test_a_permanent_error_is_not_retried(client, monkeypatch):
    calls = _responses(client, monkeypatch, [_ApiError(400), "unreachable"])
    with pytest.raises(GenerationError) as excinfo:
        client.generate("hello")
    assert not isinstance(excinfo.value, GenerationUnavailableError)
    assert calls["n"] == 1, "a 400 must fail on the first attempt"


def test_exhausted_retries_raise_the_distinguishable_error(client, monkeypatch):
    """Callers need to tell "the provider is busy, try again" apart from
    "something about this request is wrong" - the two need different
    sentences on screen.
    """
    calls = _responses(client, monkeypatch, [_ApiError(503)] * 3)
    with pytest.raises(GenerationUnavailableError):
        client.generate("hello")
    assert calls["n"] == 3


def test_retrying_can_be_switched_off(client, monkeypatch):
    client.max_attempts = 1
    calls = _responses(client, monkeypatch, [_ApiError(503)])
    with pytest.raises(GenerationUnavailableError):
        client.generate("hello")
    assert calls["n"] == 1


# --- streaming, where retrying is only sometimes legal -------------------


def _stream(client, monkeypatch, scripts):
    calls = {"n": 0}

    class _Models:
        def generate_content_stream(self, **kwargs):
            script = scripts[calls["n"]]
            calls["n"] += 1
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield type("Chunk", (), {"text": item})()

    monkeypatch.setattr(client, "client", type("C", (), {"models": _Models()})())
    return calls


def test_a_stream_that_fails_before_any_output_is_retried(client, monkeypatch):
    calls = _stream(client, monkeypatch, [[_ApiError(503)], ["all ", "good"]])
    assert "".join(client.generate_stream("hi")) == "all good"
    assert calls["n"] == 2


def test_a_stream_that_already_emitted_text_is_NOT_retried(client, monkeypatch):
    """The case that is easy to get wrong and ugly when you do.

    Chunks are rendered as they arrive, so once any text has reached the
    caller it is on screen. Restarting the request would append a second
    copy of the answer underneath the first rather than replace it. A
    stream that has begun can only be failed, never retried - even though
    the error is one that would otherwise justify a retry.
    """
    calls = _stream(client, monkeypatch, [["half an ", _ApiError(503)], ["unreachable"]])
    received = []
    with pytest.raises(GenerationUnavailableError):
        for piece in client.generate_stream("hi"):
            received.append(piece)
    assert received == ["half an "], "what was already shown stays shown"
    assert calls["n"] == 1, "must not start a second stream"


# --- when the provider names a wait, retrying makes things worse ---------


class _RateLimited(Exception):
    def __init__(self, seconds="13"):
        super().__init__(
            f"429 RESOURCE_EXHAUSTED. Quota exceeded, limit: 5. "
            f"Please retry in {seconds}.902226599s. 'retryDelay': '{seconds}s'"
        )
        self.code = 429


@pytest.mark.parametrize("raw,expected", [
    ("Please retry in 13.902226599s.", 13.902226599),
    ("{'retryDelay': '13s'}", 13.0),
    ('"retryDelay": "7s"', 7.0),
    ("no delay mentioned", None),
])
def test_the_providers_own_wait_is_read_when_given(raw, expected):
    from bao.ai.client import _server_retry_delay

    assert _server_retry_delay(Exception(raw)) == expected


def test_a_long_wait_is_not_burned_through_with_quick_retries(client, monkeypatch):
    """The free tier allows five requests a minute and asks for ~13
    seconds. Three sub-second retries do not wait that out; they spend
    three more of the five, so the retry makes the exact condition it is
    reacting to worse and then reports failure anyway.
    """
    calls = _responses(client, monkeypatch, [_RateLimited(), "unreachable"])
    with pytest.raises(GenerationUnavailableError) as excinfo:
        client.generate("hello")
    assert calls["n"] == 1, "must not spend more quota on a known-long wait"
    assert excinfo.value.retry_after == pytest.approx(13.9, abs=0.1)


def test_a_short_wait_is_still_retried(client, monkeypatch):
    """A one-second hint is worth waiting out; the rule is about long
    waits, not about any wait at all.
    """
    calls = _responses(client, monkeypatch, [_RateLimited(seconds="1"), "answer"])
    assert client.generate("hello") == "answer"
    assert calls["n"] == 2


def test_the_wait_reaches_the_user_as_a_number():
    """"Try again in a moment" invites an immediate retry that fails again.
    A number tells them how long to actually leave it.
    """
    from bao.ai.prompts import generation_busy_message

    assert "13 seconds" in generation_busy_message(13.4)
    assert "1 second." in generation_busy_message(0.6)
    assert "moment" in generation_busy_message(None)


# --- how long the user watches a spinner ---------------------------------


def test_the_thinking_level_reaches_the_request(client):
    """Gemini 3.x models emit NOTHING until reasoning finishes, so this is
    what decides how long someone watches a spinner - streaming cannot
    shorten it, because there is nothing to stream yet.

    Measured on gemini-3.5-flash with "explain calculus in xitsonga":
    MINIMAL reached the first word in 6.6s against 11.9s on the model
    default, and in the full app the first word arrived in 2.7s where it
    had been 21.7s. All levels answered in correct Xitsonga at 100%, so
    the reasoning was buying no accuracy on this kind of question.
    """
    client.thinking_level = "MINIMAL"
    config = client._config("be brief", 0.3)
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == "MINIMAL"


def test_default_leaves_the_model_to_decide(client):
    """An explicit "default" must send no thinking_config at all, rather
    than sending one that happens to mean the same thing.
    """
    client.thinking_level = "default"
    assert client._config(None, 0.3).thinking_config is None


def test_an_unusable_thinking_level_does_not_break_generation(client):
    """The field is rejected outright by models that do not support it,
    and older SDKs have no ThinkingConfig at all. Neither should cost the
    app its generation path - a latency tuning knob must not be able to
    take the answer down with it.
    """
    client.thinking_level = "NOT_A_REAL_LEVEL"
    config = client._config(None, 0.3)  # must not raise
    assert config.temperature == 0.3
