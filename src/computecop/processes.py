"""Heavy developer process detection."""

from __future__ import annotations

from collections.abc import Iterable

import psutil

from computecop.models import ProcessSample

HEAVY_PROCESS_HINTS = {
    "code",
    "cursor",
    "devenv",
    "pycharm",
    "idea",
    "webstorm",
    "chrome",
    "msedge",
    "firefox",
    "node",
    "npm",
    "pnpm",
    "yarn",
    "python",
    "pytest",
    "ruff",
    "mypy",
    "docker",
    "docker desktop",
    "ollama",
    "llama",
    "llama-server",
    "main",
    "cmake",
    "cl",
    "clang",
    "gcc",
    "g++",
    "rustc",
    "cargo",
    "java",
}


class HeavyProcessDetector:
    """Find resource-heavy local developer processes."""

    def __init__(
        self,
        process_hints: Iterable[str] | None = None,
        min_rss_mb: float = 150.0,
        min_cpu_percent: float = 5.0,
        limit: int = 12,
    ) -> None:
        self.process_hints = {hint.casefold() for hint in (process_hints or HEAVY_PROCESS_HINTS)}
        self.min_rss_bytes = int(min_rss_mb * 1024 * 1024)
        self.min_cpu_percent = min_cpu_percent
        self.limit = limit

    def sample(self) -> tuple[ProcessSample, ...]:
        """Return the heaviest matching processes sorted by memory and CPU use."""

        samples: list[ProcessSample] = []
        attrs = ["pid", "name", "cpu_percent", "memory_info", "cmdline"]
        for process in psutil.process_iter(attrs=attrs):
            try:
                info = process.info
                name = str(info.get("name") or "")
                cpu_percent = float(info.get("cpu_percent") or 0.0)
                memory_info = info.get("memory_info")
                rss = int(getattr(memory_info, "rss", 0) or 0)
                cmdline = _join_cmdline(info.get("cmdline"))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
                continue

            haystack = f"{name} {cmdline}".casefold()
            is_known_heavy = any(hint in haystack for hint in self.process_hints)
            is_resource_heavy = rss >= self.min_rss_bytes or cpu_percent >= self.min_cpu_percent
            if not is_known_heavy and not is_resource_heavy:
                continue

            samples.append(
                ProcessSample(
                    pid=int(info.get("pid") or process.pid),
                    name=name or f"pid-{process.pid}",
                    cpu_percent=cpu_percent,
                    memory_rss_bytes=rss,
                    command=cmdline[:500],
                )
            )

        samples.sort(key=lambda item: (item.memory_rss_bytes, item.cpu_percent), reverse=True)
        return tuple(samples[: self.limit])


def _join_cmdline(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)
