"""Opt-in continuation of the pending request when every configured route hits its quota.

Cooldowns belong to backend deployments, not model-name literals. The ordinary fallback
walk gets first choice; this path only runs once that walk has no route left to try.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from agent.backend_identity import BackendIdentity, same_deployment
from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)
_QUOTA_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.upstream_rate_limit})


def _settings() -> tuple[bool, float]:
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly().get("fallback")
    if not isinstance(config, dict) or config.get("wait_for_reset") is not True:
        return False, 60.0
    try:
        poll = float(config.get("reset_poll_seconds", 60))
    except (TypeError, ValueError):
        poll = 60.0
    return True, poll if math.isfinite(poll) and poll >= 1 else 60.0


def _identity(entry: dict[str, Any]) -> BackendIdentity:
    from hermes_cli.model_normalize import normalize_model_for_provider

    provider = entry.get("provider") or ""
    return BackendIdentity.build(
        provider=provider, model=normalize_model_for_provider(entry.get("model") or "", provider),
        base_url=entry.get("base_url"))


def _active_entry(agent: Any) -> dict[str, Any]:
    return {"provider": agent.provider, "model": agent.model, "base_url": str(agent.base_url or "")}


def _record(agent: Any, identity: BackendIdentity, deadline: float | None) -> None:
    state = getattr(agent, "_quota_reset_deadlines", [])
    agent._quota_reset_deadlines = [
        (key, until) for key, until in state if not same_deployment(identity, key)
    ] + [(identity, deadline)]


def _recorded_deadline(agent: Any, entry: dict[str, Any]) -> float | None:
    identity = _identity(entry)
    return next((until for key, until in getattr(agent, "_quota_reset_deadlines", [])
                 if same_deployment(identity, key)), None)


def note_route_failure(agent: Any, reason: FailoverReason | None, reset_at=None) -> None:
    """Remember the route being left, including a non-quota failure superseding an old bench."""
    enabled, poll = _settings()
    if not enabled or reason is None:
        return
    deadline = None
    if reason in _QUOTA_REASONS:
        from agent.credential_pool import _parse_absolute_timestamp

        parsed = _parse_absolute_timestamp(reset_at)
        deadline = parsed if parsed is not None and math.isfinite(parsed) and parsed > time.time() else time.time() + poll
    _record(agent, _identity(_active_entry(agent)), deadline)


def quota_candidate_blocked(agent: Any, entry: dict[str, Any]) -> bool:
    """The normal fallback walk must not spend a request on an unexpired quota bench."""
    if not _settings()[0]:
        return False
    deadline = _recorded_deadline(agent, entry)
    return deadline is not None and deadline > time.time()


def _pool_deadline(entry: dict[str, Any]) -> float | None:
    from agent.custom_entry_pool import load_runtime_pool

    try:
        pool = load_runtime_pool(
            entry.get("provider"), entry.get("base_url"), entry.get("requested_provider"), entry)
        if pool is None:
            return None
        deadline = pool.next_available_at(model=_identity(entry).model or None)
    except Exception:
        # The provider's reset remains usable when optional on-disk auth metadata is unavailable.
        logger.debug("Could not read quota recovery pool metadata", exc_info=True)
        return None
    return deadline if deadline is not None and math.isfinite(deadline) and deadline > time.time() else None


def _routes(agent: Any) -> list[tuple[dict[str, Any], int | None, float]]:
    """Primary first; only routes known to have a temporary quota bench are retry candidates."""
    from agent.chat_completion_helpers import _fallback_entry_key
    from agent.fallback_cooldown import _is_entitlement_rejected

    primary = agent._primary_runtime
    entries = [(primary, None)] + [(entry, index) for index, entry in enumerate(agent._fallback_chain)]
    routes = []
    for entry, index in entries:
        if not entry or not entry.get("provider") or not entry.get("model"):
            continue
        if _is_entitlement_rejected(agent, entry["provider"], entry["model"]):
            continue
        if index is not None and _fallback_entry_key(entry) in getattr(agent, "_unavailable_fallback_keys", set()):
            continue
        deadline, pool_deadline = _recorded_deadline(agent, entry), _pool_deadline(entry)
        if deadline is None and pool_deadline is None:
            continue
        deadline = max(deadline or 0, pool_deadline or 0)
        if pool_deadline is not None:
            # Once the pool reopens it returns None, so retain its observed eligibility time.
            _record(agent, _identity(entry), deadline)
        if index is None:
            # Existing primary restoration also gates on the monotonic cooldown.
            cooldown = max(0, getattr(agent, "_rate_limited_until", 0) - time.monotonic())
            deadline = max(deadline, time.time() + cooldown)
        routes.append((entry, index, deadline))
    return routes


def _activate(agent: Any, entry: dict[str, Any], index: int | None) -> bool:
    if same_deployment(_identity(entry), _identity(_active_entry(agent))):
        agent._fallback_index = 0 if index is None else index + 1
        return True
    if index is None:
        return bool(agent._restore_primary_runtime())
    agent._fallback_index = index
    return bool(agent._try_activate_fallback())


def _wait(
    agent: Any, seconds: float, retry: Any, *, messages: Any,
    conversation_history: Any, api_call_count: int,
) -> dict[str, Any] | None:
    from agent.turn_recovery import interruptible_backoff_sleep

    primary = agent._primary_runtime
    notice = (
        f"⏳ Model quotas exhausted — waiting {math.ceil(seconds)}s for the next reset; "
        f"primary {primary['model']} via {primary['provider']} is preferred. "
        "Press Esc or send /stop to cancel."
    )
    agent._emit_diagnostic_status(notice)
    agent._emit_diagnostic_wait(notice)
    return interruptible_backoff_sleep(
        agent, seconds, retry, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count, abort_message="Quota reset wait cancelled.",
        interrupt_text="Operation interrupted while waiting for model usage to reset.",
        activity_label="waiting for model usage reset",
    )


def recover_exhausted_quota(
    agent: Any, api_error: Any, classified: Any, retry: Any, *, error_context: Any,
    messages: Any, api_messages: Any, active_system_prompt: Any,
    conversation_history: Any, api_call_count: int,
) -> dict[str, Any] | None:
    """Wait and restart this API request, or return None for normal terminal handling.

    This has no cycle cap: opting in keeps the current request pending until a route
    recovers or the user stops it. No completed tool call is re-executed by the wait.
    """
    if not _settings()[0] or classified.reason not in _QUOTA_REASONS:
        return None
    if agent._fallback_index < len(agent._fallback_chain) or not agent._primary_runtime:
        return None
    if agent._has_content_after_think_block(getattr(agent, "_current_streamed_assistant_text", "") or ""):
        return None
    context = {**(classified.error_context or {}), **(error_context or {})}
    if "reset_at" not in context:
        context.update(agent._extract_api_error_context(api_error))
    note_route_failure(agent, classified.reason, context.get("reset_at"))

    while routes := _routes(agent):
        now = time.time()
        ready = next((route for route in routes if route[2] <= now), None)
        if ready is None:
            interrupted = _wait(
                agent, min(route[2] for route in routes) - now, retry, messages=messages,
                conversation_history=conversation_history, api_call_count=api_call_count,
            )
            if interrupted is not None:
                return {"action": "return", "result": interrupted}
            if retry.restart_with_redirected_messages:
                return {"action": "break"}
            retry.quota_wait_completed = True
            continue
        entry, index, _ = ready
        if _activate(agent, entry, index):
            from agent.conversation_loop import _arm_fallback_restart

            prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, retry)
            agent._emit_diagnostic_status(f"↻ Usage reset retry: {agent.model} via {agent.provider}.")
            return {"action": "break", "active_system_prompt": prompt}
        logger.warning("Quota reset route could not be restored: %s/%s", entry["provider"], entry["model"])
        _record(agent, _identity(entry), None)
    return None
