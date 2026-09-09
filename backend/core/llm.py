"""
LLM provider factory.

One place decides which model serves which job, so swapping a provider or
a model is a config change rather than a hunt through the agents.

    vision      reads invoice images, returns JSON
    reasoning   HS classification, the hard judgement
    judge       verification and routing; cheap, and it only checks

Every model id comes from settings, never a literal in this file. That is
not over-engineering: Google retired both text-embedding-004 and
gemini-2.0-flash during this project's lifetime, and a hardcoded id turns
each retirement into a 404 raised deep inside an agent. When a model is
shut down, edit .env and restart.

Run `python -m backend.core.llm` to list the models your key can actually
reach — the fastest way to answer "is this id still alive?".

Every call in the process is also throttled through one shared token
bucket. See `get_rate_limiter` below for why that is not optional.
"""

from __future__ import annotations

import functools
import threading
import time

from backend.core.config import get_settings

# Temperatures are deliberate, not defaults.
#   0.0 for extraction, judging and routing: same input, same answer.
#   0.2 for classification: a little room to weigh competing headings.
TEMPERATURES = {"vision": 0.0, "reasoning": 0.2, "judge": 0.0}


def model_for(role: str) -> str:
    """Resolve a role to a model id from settings."""
    settings = get_settings()
    return {
        "vision": settings.vision_model,
        "reasoning": settings.chat_model,
        "judge": settings.judge_model,
    }.get(role, settings.chat_model)


# ------------------------------------------------------------ quota
@functools.lru_cache(maxsize=8)
def get_rate_limiter(model: str):
    """A token bucket for one model id, shared by every caller using it.

    The free tier allows 5 generateContent requests a minute. A single
    invoice needs a vision call, two classification calls per line item,
    a verification call per item and a grounded search — well past five —
    and the graph runs items concurrently, so they arrive in a burst.
    Without throttling the run dies mid-pipeline on a 429, which is how
    this pipeline actually failed before this existed.

    The bucket is keyed by MODEL ID, because that is how Google meters
    it: the 429 names `model: gemini-3.5-flash` and counts that model's
    requests alone. So the classifier and the verifier draw on separate
    budgets when they run on different models, and roles that share a
    model correctly share one bucket — `lru_cache` returns the same
    limiter for the same id.

    Keying it per *role* instead would be wrong in both directions: it
    would let two roles on one model spend double that model's quota,
    while making two models queue behind each other for no reason.

    Returns None if the installed LangChain has no rate limiter, in which
    case `acquire_quota` falls back to a local implementation.
    """
    try:
        from langchain_core.rate_limiters import InMemoryRateLimiter
    except ImportError:
        return None

    rpm = max(get_settings().gemini_rpm, 1)
    return InMemoryRateLimiter(
        requests_per_second=rpm / 60.0,
        # Check often enough that a waiting thread starts promptly once a
        # token is available, instead of sleeping through its own turn.
        check_every_n_seconds=0.1,
        # Allow a burst of exactly one minute's worth. The quota is "5 per
        # minute", not "one every twelve seconds", so a bucket this size
        # is a faithful model of it rather than a risk.
        #
        # Sizing it at 1 instead — the obvious cautious choice — forces a
        # twelve-second gap between every call including the first, and
        # turned a single typed question into a three-minute wait for no
        # quota benefit at all.
        max_bucket_size=rpm,
    )


class _FallbackLimiter:
    """Minimum spacing between calls, for when LangChain has no limiter.

    Only used by the direct google-genai grounding call and on older
    LangChain versions. Deliberately simple: a lock and a timestamp.
    """

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


@functools.lru_cache(maxsize=8)
def _fallback_limiter(model: str) -> _FallbackLimiter:
    return _FallbackLimiter(get_settings().gemini_rpm)


def acquire_quota(model: str | None = None) -> None:
    """Block until this process may make one more request against `model`.

    LangChain models are given their limiter directly and throttle
    themselves. This is for the paths that bypass LangChain — the
    google-genai grounding client — so they draw from the same per-model
    budget instead of quietly spending it behind the limiter's back.
    """
    model = model or get_settings().grounding_model
    limiter = get_rate_limiter(model)
    if limiter is not None:
        limiter.acquire(blocking=True)
    else:
        _fallback_limiter(model).acquire()


def is_rate_limit_error(exc: BaseException) -> bool:
    """True for a quota/429 error, whichever exception class carries it."""
    text = f"{type(exc).__name__} {exc}"
    return any(
        marker in text
        for marker in ("429", "RESOURCE_EXHAUSTED", "RateLimit", "quota")
    )


def is_daily_quota_error(exc: BaseException) -> bool:
    """True when the DAY's allowance is gone, not just this minute's.

    Google reports both through the same 429, separated only by the
    quotaId: `...RequestsPerMinute...` recovers in seconds, while
    `...RequestsPerDay...` recovers at midnight Pacific and will not
    succeed on any retry today.

    Worth telling apart, because retrying a daily exhaustion is pure
    waste: the API answers each attempt with a `retryDelay` of a minute
    or so, and honouring three of those added nearly two minutes to a
    request that was never going to succeed.
    """
    return "PerDay" in str(exc)


class RetryOnRateLimit:
    """Wraps a runnable so `.invoke` retries rate-limit errors only.

    The limiter makes a 429 unlikely; it does not make it impossible.
    Quota windows are enforced server-side and do not align exactly with
    our local clock, and a daily cap ignores pacing entirely. Retrying
    turns a run-ending exception into a slower answer.

    Only rate-limit errors are retried. A malformed prompt or a dead
    model id fails on the first attempt, because retrying it would burn
    quota to arrive at the same failure. LangChain's own `.with_retry`
    filters by exception *type*, and the SDK raises quota errors under
    several classes across versions, so the test is on the message.

    The wait honours the retryDelay the API returns when it names one:
    the server knows how long its own window has left better than a
    backoff curve does.
    """

    def __init__(self, runnable, attempts: int) -> None:
        self._runnable = runnable
        self._attempts = max(attempts, 1)

    def __getattr__(self, name):
        # Anything not overridden here — bind, schema helpers — still
        # reaches the wrapped runnable.
        return getattr(self._runnable, name)

    def invoke(self, *args, **kwargs):
        last: BaseException | None = None
        for attempt in range(self._attempts):
            try:
                return self._runnable.invoke(*args, **kwargs)
            except Exception as exc:
                if not is_rate_limit_error(exc):
                    raise
                if is_daily_quota_error(exc):
                    raise RuntimeError(
                        f"The free-tier DAILY quota for "
                        f"{getattr(self._runnable, 'model', 'this model')} is "
                        "used up; it resets at midnight US Pacific. Retrying "
                        "will not help today.\n"
                        "Either switch model in .env (each model id has its "
                        "own daily allowance, so CHAT_MODEL=gemini-3.5-flash-lite "
                        "often still works), or enable billing on the key."
                    ) from exc
                last = exc
                if attempt == self._attempts - 1:
                    break
                delay = retry_delay_seconds(exc, default=2.0 * (2 ** attempt))
                print(f"[llm] rate limited, retrying in {delay:.0f}s "
                      f"({attempt + 1}/{self._attempts - 1})")
                time.sleep(delay)
        raise last  # type: ignore[misc]


def retry_delay_seconds(exc: BaseException, default: float) -> float:
    """The server's own retryDelay if it named one, else `default`.

    Gemini returns `retryDelay: '15s'` in the error body. Waiting exactly
    that long beats guessing, and beats waiting far longer than needed.
    """
    import re

    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    if match:
        # A second of margin, because the window closes on their clock.
        return float(match.group(1)) + 1.0
    return default


def with_retry(runnable):
    """Retry a runnable on rate-limit errors with exponential backoff."""
    return RetryOnRateLimit(runnable, get_settings().llm_max_attempts)


@functools.lru_cache(maxsize=8)
def get_llm(role: str = "reasoning"):
    """Return a chat model for a named role.

    Cached because building a client per call is slow, and LangSmith then
    records each one as a separate root run instead of one trace.

    Raises:
        RuntimeError: if no API key is configured, naming the variable to
            set rather than surfacing a stack trace.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env "
            "(free key at aistudio.google.com/apikey)."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = model_for(role)
    kwargs = {
        "model": model,
        "google_api_key": settings.google_api_key,
        "max_retries": 2,
        # A hung call is worse than a failed one: the user watches a
        # spinner with no way to tell whether it will ever return.
        "timeout": settings.llm_timeout,
    }
    # Share the process-wide bucket so concurrent line items queue behind
    # one quota rather than racing each other into a 429.
    limiter = get_rate_limiter(model)
    if limiter is not None:
        kwargs["rate_limiter"] = limiter
    # Some models ("-lite", "-preview") use fixed sampling and warn loudly
    # if a temperature is passed. Omitting it keeps the logs readable.
    if not any(tag in model for tag in ("-lite", "-preview")):
        kwargs["temperature"] = TEMPERATURES.get(role, 0.2)

    return ChatGoogleGenerativeAI(**kwargs)


def structured(role: str, schema, method: str | None = None):
    """A model that must answer in the shape of a pydantic schema.

    Structured output is what keeps agent-to-agent hand-offs honest: the
    next node receives typed fields, never a paragraph it has to re-parse.

    Text-only agents use the library default, which works. The vision
    path does NOT use this at all: with_structured_output drives the SDK's
    automatic-function-calling loop, and on a multimodal call that loop
    stalls indefinitely. See backend/agents/extractor.py, which asks for
    JSON directly and parses it instead.
    """
    llm = get_llm(role)
    if method:
        try:
            return with_retry(llm.with_structured_output(schema, method=method))
        except Exception as exc:
            print(f"[llm] {method} unavailable for {model_for(role)}: {exc}")
    return with_retry(llm.with_structured_output(schema))


def list_available_models() -> list[str]:
    """Model ids this API key can currently use for generateContent."""
    from google import genai

    client = genai.Client(api_key=get_settings().google_api_key)
    names = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            names.append(model.name)
    return sorted(names)


if __name__ == "__main__":
    settings = get_settings()
    print("Configured:")
    print(f"  embeddings {settings.embedding_provider}", end="")
    print(f" ({settings.fastembed_model})" if settings.embedding_provider == "fastembed"
          else f" ({settings.embedding_model})")
    print(f"  vision    {settings.vision_model}")
    print(f"  reasoning {settings.chat_model}")
    print(f"  judge     {settings.judge_model}")

    if not settings.google_api_key:
        raise SystemExit("\nGOOGLE_API_KEY is not set in .env")

    print("\nAvailable to this key:")
    try:
        available = list_available_models()
    except Exception as exc:
        raise SystemExit(f"  could not list models: {exc}")

    for name in available:
        print(f"  {name}")

    configured = {settings.vision_model, settings.chat_model, settings.judge_model}
    missing = {
        m for m in configured
        if m not in available and f"models/{m}" not in available
    }
    if missing:
        print(f"\n⚠  Not reachable with this key: {', '.join(sorted(missing))}")
        print("   Pick a live id above and set it in .env.")
    else:
        print("\n✓ Every configured model is reachable.")
