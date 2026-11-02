"""A pending turn survives exhausted primary and fallback quotas, without replaying tools."""

import copy
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_retry_state import TurnRetryState
from hermes_cli.config import atomic_config_write
from hermes_constants import get_hermes_home
from run_agent import AIAgent


class Clock:
    now = 1_900_000_000.0
    elapsed = 1000.0

    def advance(self, seconds):
        self.now += seconds
        self.elapsed += seconds


class ProviderError(Exception):
    def __init__(self, status, reset_at=None, now=0):
        message = "Rate limit exceeded" if status == 429 else "Invalid credentials or unpaid balance"
        super().__init__(message)
        self.status_code = status
        self.body = {}
        headers = {"Retry-After": str(reset_at - now)} if isinstance(reset_at, (int, float)) else {}
        self.response = SimpleNamespace(headers=headers)


@pytest.fixture
def rig(monkeypatch, request):
    clock = Clock()
    monkeypatch.setattr(time, "time", lambda: clock.now)
    monkeypatch.setattr(time, "monotonic", lambda: clock.elapsed)
    config_path = get_hermes_home() / "config.yaml"
    atomic_config_write(config_path, {"fallback": {"wait_for_reset": True, "reset_poll_seconds": 60}})
    options = getattr(request, "param", {})
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-primary", provider=options.get("provider", "openrouter"),
            model=options.get("model", "primary-test-model"),
            base_url="https://primary.invalid/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            fallback_model=[{
                "provider": options.get("fallback_provider", "openai"),
                "model": options.get("fallback_model", options.get("model", "backup-test-model")),
                "base_url": "https://backup.invalid/v1", "api_key": "test-backup",
            }],
        )
    backup = MagicMock(base_url="https://backup.invalid/v1", api_key="test-backup")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: (backup, k.get("model")))
    monkeypatch.setattr(agent, "_try_recover_primary_transport", lambda *a, **k: False)
    monkeypatch.setattr("agent.turn_api_error.max_retries_exhausted_result", lambda *a, **k: {"failed": True})
    monkeypatch.setattr("agent.turn_api_error.nonretryable_client_error_result", lambda *a, **k: {"failed": True})
    agent._current_streamed_assistant_text = ""
    statuses, waits = [], []
    monkeypatch.setattr(agent, "_emit_diagnostic_status", statuses.append)
    monkeypatch.setattr(agent, "_emit_diagnostic_wait", lambda text: None)

    def sleep(agent, seconds, retry, **kwargs):
        waits.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", sleep)
    yield SimpleNamespace(agent=agent, clock=clock, waits=waits, statuses=statuses, config_path=config_path)
    agent.close()


def exhaust_primary(rig, reset_after=120):
    reset = rig.clock.now + reset_after if isinstance(reset_after, (int, float)) else reset_after
    assert rig.agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=reset)
    assert rig.agent.model == "backup-test-model"


def settle(rig, reset_after=300, *, status=429):
    reset = rig.clock.now + reset_after if isinstance(reset_after, (int, float)) else reset_after
    error = ProviderError(status, reset, rig.clock.now)
    classified = classify_api_error(error, provider=rig.agent.provider, model=rig.agent.model)
    messages = [
        {"role": "user", "content": "Continue the task"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "done-tool", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "done-tool", "content": "already completed"},
    ]
    before = copy.deepcopy(messages)
    retry = TurnRetryState(primary_recovery_attempted=True)
    verdict = settle_unrecovered_error(
        rig.agent, api_error=error, classified=classified, _retry=retry,
        status_code=status, error_msg=str(error), error_context=classified.error_context,
        is_context_length_error=False, is_rate_limited=status == 429, _is_zai_coding_overload=False,
        _provider=rig.agent.provider, _base=rig.agent.base_url, _model=rig.agent.model,
        messages=messages, api_messages=copy.deepcopy(messages), api_kwargs={}, active_system_prompt=None,
        conversation_history=[], approx_tokens=0, retry_count=3, max_retries=3,
        compression_attempts=0, api_call_count=1,
    )
    assert messages == before, "completed tool results and prior conversation must survive"
    return verdict, retry


def test_both_quotas_exhausted_waits_for_primary_and_restarts_pending_request(rig):
    exhaust_primary(rig)
    verdict, retry = settle(rig)
    assert verdict.action == "break"
    assert rig.waits == [120]
    assert rig.agent.model == "primary-test-model"
    assert rig.agent._fallback_index == 0
    assert retry.restart_with_rebuilt_messages
    assert verdict.retry_count == 0
    assert any("quota" in line.lower() and "primary" in line.lower() for line in rig.statuses)


def test_earlier_fallback_reset_runs_then_next_reset_restores_primary(rig):
    start = rig.clock.now
    exhaust_primary(rig, 300)
    verdict, _ = settle(rig, 30)
    assert verdict.action == "break"
    assert rig.waits == [30]
    assert rig.agent.model == "backup-test-model"
    verdict, _ = settle(rig, 500)
    assert verdict.action == "break"
    assert rig.clock.now == start + 300
    assert rig.waits == [30, 270]
    assert rig.agent.model == "primary-test-model"

    # The backup is still benched. A new primary quota must not hit it early.
    assert not rig.agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=rig.clock.now + 50)
    verdict, _ = settle(rig, 50)
    assert verdict.action == "break"
    assert rig.waits[-1] == 50
    assert rig.agent.model == "primary-test-model"


def test_primary_wins_when_both_reset_together(rig):
    exhaust_primary(rig, 90)
    verdict, _ = settle(rig, 90)
    assert verdict.action == "break"
    assert rig.agent.model == "primary-test-model"
    assert rig.waits == [90]


def test_primary_already_reopened_needs_no_wait(rig):
    exhaust_primary(rig, 30)
    rig.clock.advance(31)
    verdict, _ = settle(rig, 90)
    assert verdict.action == "break"
    assert rig.agent.model == "primary-test-model"
    assert rig.waits == []


@pytest.mark.parametrize("reset", [None, "invalid", float("nan"), float("inf")])
def test_unknown_or_invalid_reset_polls_after_positive_delay(rig, reset):
    exhaust_primary(rig, reset)
    verdict, _ = settle(rig, None)
    assert verdict.action == "break"
    assert rig.waits == [60]
    assert rig.agent.model == "primary-test-model"


def test_disabled_recovery_keeps_terminal_error_behavior(rig):
    atomic_config_write(rig.config_path, {"fallback": {"wait_for_reset": False, "reset_poll_seconds": 60}})
    exhaust_primary(rig)
    verdict, _ = settle(rig)
    assert verdict.action == "return"
    assert verdict.result["failed"]
    assert rig.waits == []


@pytest.mark.parametrize("status", [401, 402, 403, 400])
def test_permanent_failure_never_waits_for_quota(rig, status):
    exhaust_primary(rig)
    verdict, _ = settle(rig, status=status)
    assert verdict.action == "return"
    assert rig.waits == []


def test_delivered_partial_answer_is_not_replayed(rig):
    exhaust_primary(rig)
    rig.agent._current_streamed_assistant_text = "Already delivered part of the answer"
    verdict, _ = settle(rig)
    assert verdict.action == "return"
    assert rig.waits == []


def test_cancel_while_waiting_returns_interrupted_result(rig, monkeypatch):
    exhaust_primary(rig)
    result = {"interrupted": True, "completed": False}
    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **k: result)
    verdict, _ = settle(rig)
    assert verdict.action == "return"
    assert verdict.result is result


def test_steering_during_wait_rebuilds_without_retrying_stale_request(rig, monkeypatch):
    exhaust_primary(rig)

    def steer(agent, seconds, retry, **kwargs):
        retry.restart_with_redirected_messages = True

    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", steer)
    verdict, retry = settle(rig)
    assert verdict.action == "break"
    assert retry.restart_with_redirected_messages
    assert not retry.restart_with_rebuilt_messages


def test_live_turn_survives_more_quota_cycles_than_the_failure_restart_limit(rig, monkeypatch):
    from openai.types.chat import ChatCompletion

    agent = rig.agent
    agent._disable_streaming = True
    calls = []
    quota_failures = 2 * (agent._api_max_retries + 2)

    def inference(kwargs):
        calls.append((agent.model, copy.deepcopy(kwargs["messages"])))
        if len(calls) <= quota_failures:
            delay = 60 if agent.model == "primary-test-model" else 180
            raise ProviderError(429, rig.clock.now + delay, rig.clock.now)
        return ChatCompletion(
            id="fixture-completion", object="chat.completion", created=int(rig.clock.now), model=agent.model,
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Task complete"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", inference)
    result = agent.run_conversation("Complete the fixture task")
    assert result.get("completed"), result
    assert result["final_response"] == "Task complete"
    assert len(calls) == quota_failures + 1
    assert len(rig.waits) > agent._api_max_retries
    for _, messages in calls:
        assert sum(m.get("content") == "Complete the fixture task" for m in messages) == 1


@pytest.mark.parametrize("rig", [{"provider": "openai", "model": "shared-test-model"}], indirect=True)
def test_same_model_on_two_explicit_endpoints_keeps_separate_quota_windows(rig):
    assert rig.agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=rig.clock.now + 120)
    assert rig.agent.base_url == "https://backup.invalid/v1"
    verdict, _ = settle(rig, 30)
    assert verdict.action == "break"
    assert rig.agent.base_url == "https://backup.invalid/v1"
    verdict, _ = settle(rig, 300)
    assert verdict.action == "break"
    assert rig.agent.base_url == "https://primary.invalid/v1"
    assert rig.waits == [30, 90]


def test_unknown_reset_uses_configured_poll_interval(rig):
    atomic_config_write(rig.config_path, {"fallback": {"wait_for_reset": True, "reset_poll_seconds": 17}})
    exhaust_primary(rig, None)
    verdict, _ = settle(rig, None)
    assert verdict.action == "break"
    assert rig.waits == [17]
    assert rig.agent.model == "backup-test-model"


def test_no_fallback_configured_can_wait_on_primary_alone(rig):
    rig.agent._fallback_chain = []
    verdict, _ = settle(rig, 120)
    assert verdict.action == "break"
    assert rig.agent.model == "primary-test-model"
    assert rig.waits == [120]


@pytest.mark.parametrize("rig", [{
    "fallback_provider": "openai-codex", "fallback_model": "openai/backup-test-model",
}], indirect=True)
def test_configured_model_alias_retains_its_runtime_quota_deadline(rig):
    exhaust_primary(rig, 120)
    verdict, _ = settle(rig, 30)
    assert verdict.action == "break"
    assert rig.waits == [30]
    assert rig.agent.model == "backup-test-model"
    assert not rig.agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=rig.clock.now + 90)
    # Both routes now reopen together, so primary wins.
    verdict, _ = settle(rig, 90)
    assert verdict.action == "break"
    assert rig.agent.model == "primary-test-model"


def test_fallback_skipped_for_pool_quota_is_retried_when_its_pool_reopens(rig, monkeypatch):
    from agent.credential_pool import CredentialPool, PooledCredential, STATUS_EXHAUSTED

    entry = PooledCredential(
        provider="openai", id="backup-key", label="fixture", auth_type="api_key", priority=0,
        source="manual", access_token="test-key", last_status=STATUS_EXHAUSTED,
        last_status_at=rig.clock.now, last_error_code=429, last_error_reset_at=rig.clock.now + 30,
    )
    pool = CredentialPool("openai", [entry])
    monkeypatch.setattr("agent.custom_entry_pool.load_runtime_pool",
                        lambda provider, *a, **k: pool if provider == "openai" else None)
    assert not rig.agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=rig.clock.now + 120)
    verdict, _ = settle(rig, 120)
    assert verdict.action == "break"
    assert rig.waits == [30]
    assert rig.agent.model == "backup-test-model"


def test_provider_reset_can_recover_when_optional_pool_metadata_cannot_be_read(rig, monkeypatch):
    exhaust_primary(rig, 120)

    def unreadable(*args, **kwargs):
        raise OSError("temporarily unavailable credential metadata")

    monkeypatch.setattr("agent.custom_entry_pool.load_runtime_pool", unreadable)
    verdict, _ = settle(rig, 300)
    assert verdict.action == "break"
    assert rig.waits == [120]
    assert rig.agent.model == "primary-test-model"
