"""Non-blocking terminal keyboard input for the interactive dashboard."""

from __future__ import annotations

import sys
import threading
from collections import deque


class DashboardKeyReader:
    """Poll single-key presses from the terminal without blocking the render loop."""

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="computecop-dashboard-keys")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def poll_keys(self) -> list[str]:
        with self._lock:
            keys = list(self._queue)
            self._queue.clear()
        return keys

    def _enqueue(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            self._queue.append(key.lower())

    def _reader_loop(self) -> None:
        if sys.platform == "win32":
            self._read_windows()
        else:
            self._read_posix()

    def _read_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                self._enqueue(key)
            else:
                self._stop.wait(0.05)

    def _read_posix(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)  # type: ignore[attr-defined]
        try:
            tty.setcbreak(fd)  # type: ignore[attr-defined]
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                key = sys.stdin.read(1)
                self._enqueue(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # type: ignore[attr-defined]
