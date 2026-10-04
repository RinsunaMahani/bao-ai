"""Transient provider failures must not cost the user their turn.

Observed live on gemini-3.6-flash: a question came back as "I ran into a
problem generating a response just now" while the identical request
succeeded moments later. The API had answered

    503 UNAVAILABLE - this model is currently experiencing high demand

which is the provider saying "not now", not "not ever". The client treated
it as fatal and the turn was lost.
"""
from __future__ import annotations

import threading

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
    # No backup unless a test adds one, so the single-model tests below
    # keep meaning what they say whatever config.toml lists.
    c.fallback_models = []
    c._local = threading.local()
    c.max_attempts = 3
    # Set explicitly rather than inherited from config.toml: a deployment
    # changing its latency tuning must not change what these tests assert.
    c.thinking_level = "MINIMAL"
    c.max_output_tokens = 4096
    c.timeout_seconds = 60.0
    c.client = object()
    monkeypatch.setattr(GeminiClient, "_sleep_before_retry", lambda self, attempt: None)
    return c


def _responses(client, monkeypatch, outcomes):
    """Drives models.generate_content through a scripted list of outcomes."""
    calls = {"n": 0, "models": []}

    class _Models:
        def generate_content(self, **kwargs):
            outcome = outcomes[calls["n"]]
            calls["n"] += 1
            calls["models"].append(kwargs["model"])
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
    calls = {"n": 0, "models": []}

    class _Models:
        def generate_content_stream(self, **kwargs):
            script = scripts[calls["n"]]
            calls["n"] += 1
            calls["models"].append(kwargs["model"])
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


# --- a backup model, when the main one stays busy -----------------------
#
# Observed live on 2026-09-28: gemini-3.5-flash answered "high demand" on
# all three attempts, a second apart, and the visitor got the busy message.
# Free-tier quotas and queues are per model, so another model is a real
# second chance rather than the same request again.


def test_a_busy_main_model_hands_over_to_the_backup(client, monkeypatch):
    client.fallback_models = ["backup-model"]
    calls = _responses(client, monkeypatch, [_ApiError(503)] * 3 + ["backup answer"])

    assert client.generate("hello") == "backup answer"
    assert calls["models"] == ["test-model"] * 3 + ["backup-model"]
    assert client.last_fallback_model() == "backup-model", "the badge needs to know"


def test_a_long_quota_wait_goes_straight_to_the_backup(client, monkeypatch):
    """The main model's quota is spent, and the backup has its own. The
    no-quick-retries rule still holds for the main model: one call, then
    move on.
    """
    client.fallback_models = ["backup-model"]
    calls = _responses(client, monkeypatch, [_RateLimited(), "backup answer"])

    assert client.generate("hello") == "backup answer"
    assert calls["models"] == ["test-model", "backup-model"]


def test_the_main_models_answer_clears_the_backup_flag(client, monkeypatch):
    client.fallback_models = ["backup-model"]
    _responses(client, monkeypatch, [_ApiError(503)] * 3 + ["backup answer", "main answer"])
    client.generate("first")
    assert client.last_fallback_model() == "backup-model"

    client.generate("second")
    assert client.last_fallback_model() is None


def test_a_broken_backup_still_reports_the_main_model_as_busy(client, monkeypatch):
    """A retired backup model name fails with a 404. Reporting that as "I
    ran into a problem" would hide the actual cause, which is that the main
    model was busy and the user should simply try again.
    """
    client.fallback_models = ["retired-model"]
    _responses(client, monkeypatch, [_ApiError(503)] * 3 + [_ApiError(404)])

    with pytest.raises(GenerationUnavailableError):
        client.generate("hello")
    assert client.last_fallback_model() is None


def test_both_models_busy_is_still_the_busy_error(client, monkeypatch):
    client.fallback_models = ["backup-model"]
    calls = _responses(client, monkeypatch, [_ApiError(503)] * 6)

    with pytest.raises(GenerationUnavailableError):
        client.generate("hello")
    assert calls["n"] == 6
    assert client.last_fallback_model() is None


@pytest.mark.parametrize("code", [404, 403, 400])
def test_a_main_model_that_refuses_the_request_hands_over(client, monkeypatch, code):
    """404: the model was retired. 400: it rejects a setting, as 3.7 and
    3.8 Flash reject the "minimal" thinking level. Retrying the same model
    cannot help, and it is not tried twice; another model may accept the
    identical request.
    """
    client.fallback_models = ["backup-model"]
    calls = _responses(client, monkeypatch, [_ApiError(code), "backup answer"])

    assert client.generate("hello") == "backup answer"
    assert calls["models"] == ["test-model", "backup-model"]
    assert client.last_fallback_model() == "backup-model"


def test_a_broken_backup_does_not_stop_the_list(client, monkeypatch):
    """It used to: the first backup that failed for any reason other than
    being busy ended the chain, so every backup after it was dead
    configuration.
    """
    client.fallback_models = ["retired-model", "second-backup"]
    calls = _responses(client, monkeypatch, [_ApiError(503)] * 3 + [_ApiError(404), "second answer"])

    assert client.generate("hello") == "second answer"
    assert calls["models"][-2:] == ["retired-model", "second-backup"]
    assert client.last_fallback_model() == "second-backup"


def test_an_empty_answer_from_the_main_model_does_not_hand_over(client, monkeypatch):
    """An empty answer can be a refusal. Asking another model the same
    thing would be a way round it.
    """
    client.fallback_models = ["backup-model"]
    calls = _responses(client, monkeypatch, ["", "backup answer"])

    with pytest.raises(GenerationError):
        client.generate("hello")
    assert calls["models"] == ["test-model"]


def test_every_model_refusing_reports_the_main_models_error(client, monkeypatch):
    """Not "busy": trying again will not help, and the message should not
    say it will."""
    from bao.core.exceptions import ModelRejectedError

    client.fallback_models = ["backup-model"]
    _responses(client, monkeypatch, [_ApiError(404, "main retired"), _ApiError(404, "backup retired")])

    with pytest.raises(ModelRejectedError, match="main retired"):
        client.generate("hello")
    assert client.last_fallback_model() is None


def test_a_refused_main_model_and_a_busy_backup_is_busy(client, monkeypatch):
    """Trying again later can work, so that is what the user is told."""
    client.fallback_models = ["backup-model"]
    _responses(client, monkeypatch, [_ApiError(404)] + [_ApiError(503)] * 3)

    with pytest.raises(GenerationUnavailableError):
        client.generate("hello")


def test_a_stream_hands_over_from_a_retired_model(client, monkeypatch):
    client.fallback_models = ["backup-model"]
    calls = _stream(client, monkeypatch, [[_ApiError(404)], ["from ", "backup"]])

    assert "".join(client.generate_stream("hi")) == "from backup"
    assert calls["models"] == ["test-model", "backup-model"]
    assert client.last_fallback_model() == "backup-model"


# --- no connection at all ---------------------------------------------------
#
# Measured: with Wi-Fi off the SDK raises httpx.ConnectError ("getaddrinfo
# failed") immediately. Treated as busy, it was retried three times on each
# of the three models and reported 22 seconds later as the provider being
# busy. Matched by type name, like the real one.


class ConnectError(Exception):
    """Stands in for httpx.ConnectError."""


def test_no_connection_is_one_attempt_and_no_backups(client, monkeypatch):
    from bao.core.exceptions import NetworkUnavailableError

    client.fallback_models = ["backup-model", "second-backup"]
    calls = _responses(client, monkeypatch, [ConnectError("getaddrinfo failed")] * 9)

    with pytest.raises(NetworkUnavailableError):
        client.generate("hello")
    assert calls["n"] == 1, "every backup would need the same connection"
    assert client.last_call_was_unreachable()


def test_a_stream_with_no_connection_is_one_attempt(client, monkeypatch):
    from bao.core.exceptions import NetworkUnavailableError

    client.fallback_models = ["backup-model"]
    calls = _stream(client, monkeypatch, [[ConnectError("getaddrinfo failed")]] * 6)

    with pytest.raises(NetworkUnavailableError):
        list(client.generate_stream("hi"))
    assert calls["n"] == 1
    assert client.last_call_was_unreachable()


def test_the_no_connection_record_lasts_until_the_next_turn(client, monkeypatch):
    _responses(client, monkeypatch, [ConnectError("down"), "back again"])
    with pytest.raises(GenerationError):
        client.generate("first")
    assert client.last_call_was_unreachable()

    client.forget_last_call()
    assert not client.last_call_was_unreachable()

    assert client.generate("second") == "back again"
    assert not client.last_call_was_unreachable()


def test_a_stream_hands_over_before_any_output(client, monkeypatch):
    client.fallback_models = ["backup-model"]
    calls = _stream(client, monkeypatch, [[_ApiError(503)]] * 3 + [["from ", "backup"]])

    assert "".join(client.generate_stream("hi")) == "from backup"
    assert calls["models"][-1] == "backup-model"
    assert client.last_fallback_model() == "backup-model"


def test_a_stream_that_already_emitted_text_does_not_hand_over(client, monkeypatch):
    """Same rule as retrying: a second model's answer would appear under
    the first one's half-answer on screen.
    """
    client.fallback_models = ["backup-model"]
    calls = _stream(client, monkeypatch, [["half an ", _ApiError(503)], ["unreachable"]])

    with pytest.raises(GenerationUnavailableError):
        list(client.generate_stream("hi"))
    assert calls["models"] == ["test-model"]


def test_which_model_answered_is_tracked_per_thread(client, monkeypatch):
    """One client serves every browser session, and Streamlit runs each
    session's turn on its own thread. One visitor's backup answer must not
    appear on another visitor's badge.
    """
    client.fallback_models = ["backup-model"]
    _responses(client, monkeypatch, [_ApiError(503)] * 3 + ["backup answer"])
    client.generate("hello")

    seen = []
    other = threading.Thread(target=lambda: seen.append(client.last_fallback_model()))
    other.start()
    other.join()
    assert seen == [None]
    assert client.last_fallback_model() == "backup-model"


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


def test_replies_have_a_length_ceiling_and_no_tool_calling(client):
    """Bounded output, and no automatic function calling: no tools are
    passed, so the model is given no agency this app does not use.
    """
    config = client._config(None, 0.3)
    assert config.max_output_tokens == 4096
    assert config.automatic_function_calling.disable is True


def test_zero_leaves_the_length_to_the_model(client):
    client.max_output_tokens = 0
    assert client._config(None, 0.3).max_output_tokens is None


def test_requests_time_out_rather_than_hang(monkeypatch):
    """The SDK's default is no timeout, so a stalled connection held a
    visitor's turn indefinitely.
    """
    import bao.ai.client as client_module

    captured = {}

    class _FakeSdkClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(client_module.genai, "Client", _FakeSdkClient)
    GeminiClient(Settings(), api_key="not-a-real-key")
    assert captured["http_options"].timeout == 60_000, "milliseconds"


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
