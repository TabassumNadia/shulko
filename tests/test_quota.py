"""
Tests for the quota guard.

These cover the failure that actually broke this pipeline in practice.
The Gemini free tier allows 5 generateContent requests a minute, per
model. One invoice needs a vision call, two classification calls per line
item, a verification call per item and a grounded search, and the graph
runs line items concurrently — so the requests arrive in a burst well
past the limit and the run died mid-pipeline on a 429.

Nothing here calls the network. The point is to pin the *policy*: how
requests are paced, what gets retried, and what a failure does to the
rest of the run.
"""

from __future__ import annotations

import time

import pytest

from backend.core.config import Settings
from backend.core.llm import (
    RetryOnRateLimit,
    _FallbackLimiter,
    get_rate_limiter,
    is_daily_quota_error,
    is_rate_limit_error,
    retry_delay_seconds,
)


# ----------------------------------------------------------- error detection
@pytest.mark.parametrize("message", [
    "429 RESOURCE_EXHAUSTED",
    "Error calling model (RESOURCE_EXHAUSTED): 429",
    "GoogleRateLimitError: quota exceeded",
    "You exceeded your current quota, please check your plan",
])
def test_rate_limit_errors_are_recognised(message):
    assert is_rate_limit_error(Exception(message))


@pytest.mark.parametrize("message", [
    "400 INVALID_ARGUMENT: bad prompt",
    "404 model not found",
    "connection reset by peer",
])
def test_other_errors_are_not_treated_as_rate_limits(message):
    """A dead model id must fail on the first attempt.

    Retrying it would spend quota to reach the same failure, and hide the
    real cause behind four identical stack traces.
    """
    assert not is_rate_limit_error(Exception(message))


# ------------------------------------------------------------ retry backoff
def test_retry_delay_prefers_the_servers_own_number():
    """The API says how long its window has left. Believe it."""
    exc = Exception("429 RESOURCE_EXHAUSTED {'retryDelay': '15s'}")
    delay = retry_delay_seconds(exc, default=99.0)
    # The server's 15s plus a small margin, not the caller's guess.
    assert 15.0 < delay <= 17.0


def test_retry_delay_falls_back_when_none_is_given():
    assert retry_delay_seconds(Exception("429 quota"), default=4.0) == 4.0


class _Flaky:
    """A runnable that fails with the given errors, then succeeds."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def test_retry_recovers_from_a_transient_rate_limit(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    flaky = _Flaky([Exception("429 RESOURCE_EXHAUSTED")])

    assert RetryOnRateLimit(flaky, attempts=4).invoke("x") == "ok"
    assert flaky.calls == 2


def test_retry_gives_up_and_reraises_after_the_last_attempt(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    flaky = _Flaky([Exception("429 quota")] * 10)

    with pytest.raises(Exception, match="429"):
        RetryOnRateLimit(flaky, attempts=3).invoke("x")
    assert flaky.calls == 3


def test_retry_does_not_swallow_a_real_error(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    flaky = _Flaky([ValueError("malformed prompt")])

    with pytest.raises(ValueError):
        RetryOnRateLimit(flaky, attempts=4).invoke("x")
    assert flaky.calls == 1, "a non-quota error must not be retried"


# ------------------------------------------------------------- the buckets
def test_one_bucket_per_model_not_per_role():
    """Google meters the quota per model id, so the bucket is keyed that way.

    Two roles on the same model must share one budget, or together they
    would spend double that model's quota.
    """
    assert get_rate_limiter("gemini-3.5-flash") is get_rate_limiter("gemini-3.5-flash")
    assert get_rate_limiter("gemini-3.5-flash") is not get_rate_limiter(
        "gemini-3.5-flash-lite"
    )


def test_bucket_is_paced_from_the_configured_rpm():
    limiter = get_rate_limiter("test-model-for-pacing")
    if limiter is None:
        pytest.skip("installed LangChain has no InMemoryRateLimiter")

    expected = Settings().gemini_rpm / 60.0
    assert limiter.requests_per_second == pytest.approx(expected)
    # A burst of exactly one minute's worth: the quota is "N per minute",
    # not "one every 60/N seconds". Sizing this at 1 would put a full gap
    # before even the first call.
    assert limiter.max_bucket_size == Settings().gemini_rpm


def test_fallback_limiter_spaces_calls_out():
    """Used when LangChain has no limiter, and by the grounding client."""
    limiter = _FallbackLimiter(rpm=600)          # 0.1s apart, quick to test
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    # First call is free; the next two wait their turn.
    assert time.monotonic() - start >= 0.18


def test_default_rpm_matches_the_free_tier():
    """The default must be safe on the tier a fresh clone will be on."""
    assert Settings().gemini_rpm <= 5


def test_workers_do_not_exceed_the_per_minute_budget():
    """More threads than the quota allows only means longer queues."""
    settings = Settings()
    assert 1 <= settings.max_workers <= settings.gemini_rpm


# ------------------------------------------------------- daily vs per-minute
def test_daily_and_per_minute_exhaustion_are_told_apart():
    """Both arrive as a 429; only the quotaId distinguishes them."""
    per_minute = Exception(
        "429 RESOURCE_EXHAUSTED "
        "{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}"
    )
    per_day = Exception(
        "429 RESOURCE_EXHAUSTED "
        "{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}"
    )

    assert is_rate_limit_error(per_minute) and not is_daily_quota_error(per_minute)
    assert is_rate_limit_error(per_day) and is_daily_quota_error(per_day)


def test_a_daily_exhaustion_fails_fast_instead_of_retrying(monkeypatch):
    """Retrying a spent day cannot succeed, and the API asks for a
    minute-long wait each time -- three of those added nearly two minutes
    to a request that was doomed from the first attempt."""
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)

    flaky = _Flaky([Exception(
        "429 {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
        "'retryDelay': '52s'}"
    )] * 5)

    with pytest.raises(RuntimeError, match="DAILY quota"):
        RetryOnRateLimit(flaky, attempts=4).invoke("x")

    assert flaky.calls == 1, "a spent day must not be retried"
    assert slept == [], "and must not sleep before giving up"


def test_the_daily_message_names_a_way_forward(monkeypatch):
    """An error the user cannot act on is only half an error message."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    flaky = _Flaky([Exception("429 {'quotaId': 'RequestsPerDay-FreeTier'}")])

    with pytest.raises(RuntimeError) as caught:
        RetryOnRateLimit(flaky, attempts=4).invoke("x")

    message = str(caught.value)
    assert "midnight" in message          # when it comes back
    assert ".env" in message              # what to change meanwhile
    assert "billing" in message           # the permanent fix
