"""Tests for huginn.llm_retry — retry, backoff, fallback, context overflow.

All sleeps are mocked out so the suite runs in well under a second.
If you want to skip the project-wide coverage gate, run with:

    pytest agent/tests/test_llm_retry.py --override-ini="addopts="
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from huginn.llm_retry import (
    CONTEXT_OVERFLOW_SHRINK,
    FALLBACK_MODELS,
    INITIAL_BACKOFF,
    MAX_529_RETRIES,
    MAX_BACKOFF,
    MIN_MAX_TOKENS,
    FallbackTriggeredError,
    _exponential_backoff,
    _get_retry_after,
    _get_status_code,
    _is_auth_error,
    _is_context_overflow,
    _is_overloaded,
    _is_rate_limit,
    _is_transient_network,
    _jitter,
    call_with_fallback,
    parse_context_overflow,
    persistent_retry,
    with_retry,
)
import huginn.llm_retry as lr


# ── fake exceptions & helpers ────────────────────────────────────


class FakeHttpError(Exception):
    """Mimics SDK HTTP errors — set whatever attrs the retry code probes."""

    def __init__(self, message: str = "", **attrs):
        super().__init__(message)
        for key, val in attrs.items():
            setattr(self, key, val)


# these exist purely so type(exc).__name__ triggers the name-based branches
class FakeTimeoutError(Exception):
    pass


class FakeConnectionError(Exception):
    pass


class FakeAuthenticationError(Exception):
    pass


class FakeContextOverflow(Exception):
    pass


class _HeadersWithoutGet:
    """Headers object that only supports iteration, not .get().

    Forces _get_retry_after down its fallback loop path.
    """

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def items(self):
        return self._pairs


def _seq_factory(items):
    """Turn a list of values/exceptions into a coro_factory.

    Each call advances to the next item. Exceptions get raised,
    everything else is the return value.
    """
    seq = list(items)
    pos = [0]

    def factory():
        async def _coro():
            i = pos[0]
            pos[0] += 1
            if i >= len(seq):
                raise RuntimeError("test factory ran out of items")
            item = seq[i]
            if isinstance(item, BaseException):
                raise item
            return item

        return _coro()

    return factory


def _always_ok(value="ok"):
    def factory():
        async def _coro():
            return value

        return _coro()

    return factory


def _always_raise(exc):
    def factory():
        async def _coro():
            raise exc

        return _coro()

    return factory


# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_sleep(monkeypatch):
    """Stub _sleep_with_log so with_retry tests don't actually wait."""
    m = AsyncMock()
    monkeypatch.setattr(lr, "_sleep_with_log", m)
    return m


@pytest.fixture
def no_asyncio_sleep(monkeypatch):
    """Stub asyncio.sleep for persistent_retry tests."""
    async def _noop(_seconds=0):
        pass

    monkeypatch.setattr(lr.asyncio, "sleep", _noop)
    return _noop


# ── _get_status_code ─────────────────────────────────────────────


class TestGetStatusCode:
    def test_status_code_attr(self):
        assert _get_status_code(FakeHttpError("err", status_code=429)) == 429

    def test_status_attr(self):
        assert _get_status_code(FakeHttpError("err", status=503)) == 503

    def test_response_status_code(self):
        resp = SimpleNamespace(status_code=500)
        assert _get_status_code(FakeHttpError("err", response=resp)) == 500

    def test_response_status_attr(self):
        resp = SimpleNamespace(status=401)
        assert _get_status_code(FakeHttpError("err", response=resp)) == 401

    def test_missing_everything(self):
        assert _get_status_code(FakeHttpError("plain")) is None

    def test_garbage_value(self):
        # int() should blow up gracefully
        assert _get_status_code(FakeHttpError("err", status_code="nope")) is None


# ── _get_retry_after ─────────────────────────────────────────────


class TestGetRetryAfter:
    def test_direct_attr(self):
        assert _get_retry_after(FakeHttpError("err", retry_after=5.0)) == 5.0

    def test_response_header_lowercase(self):
        resp = SimpleNamespace(headers={"retry-after": "3"})
        assert _get_retry_after(FakeHttpError("err", response=resp)) == 3.0

    def test_response_header_capitalized(self):
        resp = SimpleNamespace(headers={"Retry-After": "10"})
        assert _get_retry_after(FakeHttpError("err", response=resp)) == 10.0

    def test_headers_without_get_method(self):
        # exercises the AttributeError fallback loop
        headers = _HeadersWithoutGet([("Retry-After", "7")])
        resp = SimpleNamespace(headers=headers)
        assert _get_retry_after(FakeHttpError("err", response=resp)) == 7.0

    def test_nothing_to_find(self):
        assert _get_retry_after(FakeHttpError("err")) is None

    def test_non_numeric(self):
        assert _get_retry_after(FakeHttpError("err", retry_after="soon")) is None


# ── exception classification ─────────────────────────────────────


class TestClassification:
    def test_rate_limit_by_code(self):
        assert _is_rate_limit(FakeHttpError("err", status_code=429))

    def test_rate_limit_by_name(self):
        class RateLimitError(Exception):
            pass

        assert _is_rate_limit(RateLimitError("slow down"))

    def test_overloaded_by_code(self):
        assert _is_overloaded(FakeHttpError("err", status_code=529))

    def test_overloaded_by_text(self):
        assert _is_overloaded(FakeHttpError("service overloaded, try again"))

    def test_overloaded_by_name(self):
        class OverloadedError(Exception):
            pass

        assert _is_overloaded(OverloadedError())

    def test_context_overflow_true(self):
        exc = FakeHttpError("maximum context length is 8192 tokens")
        assert _is_context_overflow(exc)

    def test_context_overflow_false(self):
        assert not _is_context_overflow(FakeHttpError("all good"))

    def test_transient_5xx(self):
        assert _is_transient_network(FakeHttpError("err", status_code=503))
        assert _is_transient_network(FakeHttpError("err", status_code=502))

    def test_transient_excludes_529(self):
        # 529 goes through the overloaded branch, not transient
        assert not _is_transient_network(FakeHttpError("err", status_code=529))

    def test_transient_by_name(self):
        assert _is_transient_network(FakeTimeoutError())
        assert _is_transient_network(FakeConnectionError())

    def test_auth_by_code(self):
        assert _is_auth_error(FakeHttpError("err", status_code=401))

    def test_auth_by_name(self):
        assert _is_auth_error(FakeAuthenticationError())

    def test_auth_by_text(self):
        assert _is_auth_error(FakeHttpError("invalid api key"))


# ── _exponential_backoff ─────────────────────────────────────────


class TestExponentialBackoff:
    def test_known_sequence(self):
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        for attempt, want in enumerate(expected, 1):
            assert _exponential_backoff(attempt) == want

    def test_hits_cap(self):
        assert _exponential_backoff(7) == MAX_BACKOFF
        assert _exponential_backoff(100) == MAX_BACKOFF

    def test_custom_base(self):
        assert _exponential_backoff(1, base=2.0) == 2.0
        assert _exponential_backoff(3, base=2.0) == 8.0


# ── _jitter ──────────────────────────────────────────────────────


class TestJitter:
    def test_stays_in_range(self):
        for _ in range(200):
            val = _jitter(10.0)
            assert 7.5 <= val <= 12.5

    def test_zero_stays_zero(self):
        assert _jitter(0.0) == 0.0

    def test_never_negative(self):
        for _ in range(200):
            assert _jitter(0.4) >= 0.0

    def test_custom_ratio(self):
        for _ in range(200):
            val = _jitter(10.0, jitter_ratio=0.5)
            assert 5.0 <= val <= 15.0


# ── parse_context_overflow ───────────────────────────────────────


class TestParseContextOverflow:
    def test_openai_style(self):
        exc = FakeHttpError("maximum context length is 8192 tokens")
        assert parse_context_overflow(exc) == 4096

    def test_anthropic_style(self):
        exc = FakeHttpError("context window is 200000 tokens")
        assert parse_context_overflow(exc) == 100000

    def test_max_is_pattern(self):
        exc = FakeHttpError("the max is 4096")
        assert parse_context_overflow(exc) == 2048

    def test_maximum_allowed_pattern(self):
        exc = FakeHttpError("maximum allowed 8192 tokens")
        assert parse_context_overflow(exc) == 4096

    def test_keyword_only_no_number(self):
        # looks like overflow but no digits to extract
        exc = FakeContextOverflow("something broke")
        result = parse_context_overflow(exc)
        assert result == max(MIN_MAX_TOKENS, int(4096 * CONTEXT_OVERFLOW_SHRINK))

    def test_context_length_in_message(self):
        exc = FakeHttpError("context length exceeded the allowed limit")
        assert parse_context_overflow(exc) is not None

    def test_error_code_attribute(self):
        exc = FakeHttpError("err", code="context_length_exceeded")
        assert parse_context_overflow(exc) is not None

    def test_completely_unrelated(self):
        assert parse_context_overflow(FakeHttpError("network timeout")) is None

    def test_tiny_window_respects_floor(self):
        exc = FakeHttpError("context window is 100 tokens")
        # 100 * 0.5 = 50, but MIN_MAX_TOKENS kicks in
        assert parse_context_overflow(exc) == MIN_MAX_TOKENS


# ── FallbackTriggeredError ───────────────────────────────────────


class TestFallbackTriggeredError:
    def test_default_reason(self):
        err = FallbackTriggeredError()
        assert err.reason == "529_overloaded"

    def test_custom_message_preserves_reason(self):
        err = FallbackTriggeredError("custom overload msg")
        assert err.reason == "529_overloaded"
        assert "custom overload msg" in str(err)

    def test_is_an_exception(self):
        assert isinstance(FallbackTriggeredError(), Exception)


# ── with_retry ────────────────────────────────────────────────────


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        # no retry needed, factory called exactly once
        result = await with_retry(_always_ok("done"), max_attempts=3)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_429_retries_correct_count(self, mock_sleep):
        exc = FakeHttpError("rate limited", status_code=429)
        factory = _seq_factory([exc, exc, "ok"])
        result = await with_retry(factory, max_attempts=5)
        assert result == "ok"
        # two failures means two sleeps before the successful third call
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_429_uses_retry_after_header(self, mock_sleep):
        exc = FakeHttpError("429", status_code=429, retry_after=5.0)
        factory = _seq_factory([exc, "ok"])
        await with_retry(factory, max_attempts=3)
        # server asked for 5s, we honor it (±10% jitter)
        slept = mock_sleep.call_args_list[0].args[0]
        assert slept == pytest.approx(5.0, abs=0.6)

    @pytest.mark.asyncio
    async def test_429_falls_back_to_exponential(self, mock_sleep):
        exc = FakeHttpError("429", status_code=429)
        factory = _seq_factory([exc, exc, "ok"])
        await with_retry(factory, max_attempts=5)
        # no retry-after → exponential backoff, attempt 1 → 1.0s ±25%
        first_sleep = mock_sleep.call_args_list[0].args[0]
        assert first_sleep == pytest.approx(1.0, abs=0.3)

    @pytest.mark.asyncio
    async def test_429_exhausts_max_attempts(self, mock_sleep):
        exc = FakeHttpError("429", status_code=429)
        with pytest.raises(FakeHttpError) as exc_info:
            await with_retry(_always_raise(exc), max_attempts=3)
        assert exc_info.value is exc
        # attempts 1 and 2 sleep, attempt 3 hits the limit and bails
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_529_triggers_fallback_after_three(self, mock_sleep):
        exc = FakeHttpError("overloaded", status_code=529)
        with pytest.raises(FallbackTriggeredError):
            await with_retry(_always_raise(exc), max_attempts=10)
        # three 529s: sleep between 1→2 and 2→3, third one raises before sleeping
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_529_counter_resets_on_other_error(self, mock_sleep):
        e529 = FakeHttpError("529", status_code=529)
        e500 = FakeHttpError("500", status_code=500)
        # 529, 529, 500 (resets streak), 529, 529, 529 → fallback
        factory = _seq_factory([e529, e529, e500, e529, e529, e529])
        with pytest.raises(FallbackTriggeredError):
            await with_retry(factory, max_attempts=10)
        assert mock_sleep.call_count == 5

    @pytest.mark.asyncio
    async def test_529_max_attempts_below_fallback_threshold(self, mock_sleep):
        exc = FakeHttpError("529", status_code=529)
        # only 2 tries, streak maxes at 2 which is below MAX_529_RETRIES(3)
        with pytest.raises(FakeHttpError):
            await with_retry(_always_raise(exc), max_attempts=2)
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_context_overflow_calls_adjuster(self, mock_sleep):
        exc = FakeHttpError("maximum context length is 8192 tokens")
        factory = _seq_factory([exc, "ok"])
        adjuster = MagicMock(return_value=4096)
        result = await with_retry(
            factory, max_attempts=3, max_tokens_adjuster=adjuster
        )
        assert result == "ok"
        adjuster.assert_called_once_with(4096)

    @pytest.mark.asyncio
    async def test_context_overflow_without_adjuster(self, mock_sleep):
        # no adjuster passed — should still retry and succeed
        exc = FakeHttpError("context window is 200000 tokens")
        factory = _seq_factory([exc, "ok"])
        result = await with_retry(factory, max_attempts=3)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_transient_5xx_retries(self, mock_sleep):
        exc = FakeHttpError("bad gateway", status_code=502)
        factory = _seq_factory([exc, exc, "ok"])
        result = await with_retry(factory, max_attempts=5)
        assert result == "ok"
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_transient_by_exception_name(self, mock_sleep):
        factory = _seq_factory([FakeTimeoutError(), "ok"])
        result = await with_retry(factory, max_attempts=3)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_401_refresh_then_retry(self, mock_sleep):
        exc = FakeHttpError("unauthorized", status_code=401)
        factory = _seq_factory([exc, "ok"])
        with patch(
            "huginn.security.auth.handle_401_error",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await with_retry(factory, max_attempts=3)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_401_refresh_fails_raises(self, mock_sleep):
        exc = FakeHttpError("unauthorized", status_code=401)
        with patch(
            "huginn.security.auth.handle_401_error",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(FakeHttpError):
                await with_retry(_always_raise(exc), max_attempts=3)

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self, mock_sleep):
        exc = ValueError("bad input from caller")
        with pytest.raises(ValueError):
            await with_retry(_always_raise(exc), max_attempts=5)
        # should never have slept
        assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_source_appears_in_logs(self, mock_sleep, caplog):
        exc = FakeHttpError("429", status_code=429)
        factory = _seq_factory([exc, "ok"])
        with caplog.at_level(logging.WARNING, logger="huginn.llm_retry"):
            await with_retry(factory, source="batch-job-42", max_attempts=3)
        assert any("batch-job-42" in r.getMessage() for r in caplog.records)


# ── call_with_fallback ───────────────────────────────────────────


class TestCallWithFallback:
    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self, mock_sleep):
        async def llm_fn(prompt, model):
            return f"{model}-reply"

        result = await call_with_fallback("hi", "claude-sonnet-4-6", llm_fn)
        assert result == "claude-sonnet-4-6-reply"

    @pytest.mark.asyncio
    async def test_falls_back_to_cheaper_model(self, mock_sleep):
        seen_models = []

        async def llm_fn(prompt, model):
            seen_models.append(model)
            if model == "claude-sonnet-4-6":
                raise FakeHttpError("overloaded", status_code=529)
            return f"{model}-reply"

        result = await call_with_fallback("hi", "claude-sonnet-4-6", llm_fn)
        assert result == "claude-haiku-4-reply"
        assert "claude-sonnet-4-6" in seen_models
        assert "claude-haiku-4" in seen_models

    @pytest.mark.asyncio
    async def test_no_fallback_model_direct_call(self):
        async def llm_fn(prompt, model):
            return f"{model}-direct"

        result = await call_with_fallback("hi", "some-random-model", llm_fn)
        assert result == "some-random-model-direct"

    @pytest.mark.asyncio
    async def test_no_fallback_model_propagates_error(self):
        async def llm_fn(prompt, model):
            raise FakeHttpError("boom", status_code=500)

        with pytest.raises(FakeHttpError):
            await call_with_fallback("hi", "some-random-model", llm_fn)

    @pytest.mark.asyncio
    async def test_fallback_model_also_overloaded(self, mock_sleep):
        async def llm_fn(prompt, model):
            raise FakeHttpError("overloaded", status_code=529)

        # primary triggers fallback, fallback also 529s → second
        # FallbackTriggeredError propagates out
        with pytest.raises(FallbackTriggeredError):
            await call_with_fallback("hi", "claude-sonnet-4-6", llm_fn)


# ── persistent_retry ─────────────────────────────────────────────


class TestPersistentRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self, no_asyncio_sleep):
        result = await persistent_retry(_always_ok("done"), max_hours=0.001)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_retries_rate_limit_then_succeeds(self, no_asyncio_sleep):
        exc = FakeHttpError("429", status_code=429)
        factory = _seq_factory([exc, exc, "ok"])
        result = await persistent_retry(factory, max_hours=0.001)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_transient_then_succeeds(self, no_asyncio_sleep):
        exc = FakeHttpError("503", status_code=503)
        factory = _seq_factory([exc, "ok"])
        result = await persistent_retry(factory, max_hours=0.001)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self, no_asyncio_sleep):
        with pytest.raises(ValueError):
            await persistent_retry(_always_raise(ValueError("nope")), max_hours=0.001)

    @pytest.mark.asyncio
    async def test_deadline_exceeded_raises_timeout(self, no_asyncio_sleep):
        exc = FakeHttpError("429", status_code=429)
        # max_hours=0 → deadline is already past, loop never enters
        with pytest.raises(TimeoutError):
            await persistent_retry(_always_raise(exc), max_hours=0.0)

    @pytest.mark.asyncio
    async def test_swallows_fallback_signal_and_retries(self, no_asyncio_sleep):
        fb = FallbackTriggeredError("529 streak")
        factory = _seq_factory([fb, fb, "ok"])
        result = await persistent_retry(factory, max_hours=0.001)
        assert result == "ok"
