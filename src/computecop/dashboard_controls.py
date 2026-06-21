"""Interactive queue controls and keyboard handling for the dashboard."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computecop.request_queue import AsyncRequestQueue

CONFIRMATION_TIMEOUT_SECONDS = 5.0
STATUS_MESSAGE_TIMEOUT_SECONDS = 3.0


@dataclass(slots=True)
class DashboardInteractionState:
    """Mutable dashboard UI state for interactive controls."""

    detail_mode: bool = False
    pending_action: str | None = None
    status_message: str | None = None
    status_set_at: float | None = None


@dataclass(slots=True)
class DashboardQueueController:
    """Bridge dashboard keyboard actions to the runtime request queue."""

    queue: AsyncRequestQueue
    drain_seconds: float
    _drain_task: asyncio.Task[bool] | None = field(default=None, init=False, repr=False)

    async def pause(self) -> None:
        await self.queue.pause()

    async def resume(self) -> None:
        await self.queue.resume()

    def start_drain(self) -> None:
        if self._drain_task is not None and not self._drain_task.done():
            return
        deadline = monotonic() + self.drain_seconds
        self._drain_task = asyncio.create_task(self.queue.drain(deadline))

    @property
    def draining(self) -> bool:
        return self._drain_task is not None and not self._drain_task.done()


class DashboardKeyHandler:
    """Translate dashboard key presses into queue actions and UI state updates."""

    def __init__(
        self,
        controller: DashboardQueueController,
        *,
        confirmation_timeout: float = CONFIRMATION_TIMEOUT_SECONDS,
        status_timeout: float = STATUS_MESSAGE_TIMEOUT_SECONDS,
    ) -> None:
        self._controller = controller
        self._confirmation_timeout = confirmation_timeout
        self._status_timeout = status_timeout
        self._quit_requested = False

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    async def handle(self, key: str, state: DashboardInteractionState) -> None:
        now = monotonic()
        self.expire_timers(state, now)
        normalized = key.lower()

        if state.pending_action == "drain":
            if normalized == "d":
                self._controller.start_drain()
                state.pending_action = None
                self._set_status(state, "Queue drain started", now)
                return
            if normalized in {"c", "\x1b", "q"}:
                state.pending_action = None
                self._set_status(state, "Drain cancelled", now)
                return
            return

        if normalized == "q":
            self._quit_requested = True
            return
        if normalized == "p":
            await self._controller.pause()
            self._set_status(state, "Queue paused", now)
            return
        if normalized == "r":
            await self._controller.resume()
            self._set_status(state, "Queue resumed", now)
            return
        if normalized == "d":
            state.pending_action = "drain"
            self._set_status(state, "Press D again to confirm drain (C to cancel)", now)
            return
        if normalized == "t":
            state.detail_mode = not state.detail_mode
            label = "enabled" if state.detail_mode else "disabled"
            self._set_status(state, f"Detail view {label}", now)

    def expire_timers(self, state: DashboardInteractionState, now: float | None = None) -> None:
        current = monotonic() if now is None else now
        if state.pending_action is not None and state.status_set_at is not None:
            if current - state.status_set_at > self._confirmation_timeout:
                state.pending_action = None
                self._set_status(state, "Confirmation expired", current)
                return
        if (
            state.pending_action is None
            and state.status_message is not None
            and state.status_set_at is not None
            and current - state.status_set_at > self._status_timeout
        ):
            state.status_message = None
            state.status_set_at = None

    def _set_status(
        self,
        state: DashboardInteractionState,
        message: str,
        now: float,
    ) -> None:
        state.status_message = message
        state.status_set_at = now
