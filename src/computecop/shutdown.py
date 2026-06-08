"""Centralized graceful shutdown helpers."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computecop.app import ComputeCopRuntime


class ShutdownCoordinator:
    """Coordinate idempotent runtime shutdown requests."""

    def __init__(self) -> None:
        self._shutdown_requested = False
        self._shutdown_complete = False

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    @property
    def shutdown_complete(self) -> bool:
        return self._shutdown_complete

    def request_shutdown(self) -> bool:
        """Record a shutdown request. Returns False on duplicate requests."""

        if self._shutdown_requested:
            return False
        self._shutdown_requested = True
        return True

    async def shutdown_runtime(
        self,
        runtime: ComputeCopRuntime,
        *,
        drain_timeout_seconds: float | None = None,
    ) -> None:
        """Stop runtime services once."""

        if self._shutdown_complete:
            return
        self.request_shutdown()
        try:
            await runtime.stop(drain_timeout_seconds=drain_timeout_seconds)
        finally:
            self._shutdown_complete = True


async def cancel_task(task: asyncio.Task[object]) -> None:
    """Cancel a task and wait without raising cancellation errors."""

    if task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
