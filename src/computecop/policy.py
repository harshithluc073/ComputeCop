"""Juice-level policy engine."""

from __future__ import annotations

from dataclasses import dataclass

from computecop.config import PolicyConfig
from computecop.models import JuiceBudget, RequestClass, SystemState, TelemetrySample, ThermalState
from computecop.platform import HostMemoryProfile


@dataclass(frozen=True, slots=True)
class PressureReport:
    """Policy evaluation output for a telemetry snapshot."""

    system_state: SystemState
    global_juice_level: int
    yield_active: bool
    yield_reason: str | None
    reasons: tuple[str, ...]
    dynamic_yield_percent: float
    dynamic_recover_percent: float
    memory_budget_scale: float
    total_ram_gb: float | None


class JuicePolicyEngine:
    """Map resource pressure to foreground and background inference budgets."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def evaluate(self, telemetry: TelemetrySample | None) -> PressureReport:
        """Evaluate global system pressure."""

        if telemetry is None:
            return PressureReport(
                system_state=SystemState.NORMAL,
                global_juice_level=self.config.background_base_juice_level,
                yield_active=False,
                yield_reason=None,
                reasons=("telemetry unavailable; using conservative defaults",),
                dynamic_yield_percent=self.config.ram_yield_percent,
                dynamic_recover_percent=self.config.ram_recover_percent,
                memory_budget_scale=1.0,
                total_ram_gb=None,
            )

        memory = HostMemoryProfile(
            total_bytes=telemetry.ram_total_bytes,
            minimum_supported_gb=self.config.minimum_supported_ram_gb,
        )
        dynamic_yield_percent = memory.dynamic_yield_percent(self.config.ram_yield_percent)
        dynamic_recover_percent = memory.dynamic_recover_percent(
            configured_recover_percent=self.config.ram_recover_percent,
            recover_gap_percent=self.config.ram_recover_gap_percent,
            configured_yield_percent=self.config.ram_yield_percent,
        )
        reasons: list[str] = []
        penalty = 0
        yield_reason: str | None = None

        if not memory.meets_minimum:
            reasons.append(
                f"RAM capacity {memory.total_gb:.1f} GiB is below "
                f"{self.config.minimum_supported_ram_gb:.1f} GiB baseline"
            )
            penalty += 25

        if telemetry.ram_used_percent >= dynamic_yield_percent:
            yield_reason = (
                f"RAM usage {telemetry.ram_used_percent:.1f}% exceeds "
                f"dynamic yield threshold {dynamic_yield_percent:.1f}%"
            )
            reasons.append(yield_reason)
            penalty += 55
        elif telemetry.ram_used_percent >= dynamic_recover_percent:
            reasons.append(f"RAM usage elevated at {telemetry.ram_used_percent:.1f}%")
            penalty += 25

        if telemetry.cpu_percent >= self.config.cpu_pressure_percent:
            reasons.append(f"CPU usage elevated at {telemetry.cpu_percent:.1f}%")
            penalty += 15

        if telemetry.swap_used_percent >= self.config.swap_pressure_percent:
            reasons.append(f"swap usage elevated at {telemetry.swap_used_percent:.1f}%")
            penalty += 20

        thermal_penalty = self._thermal_penalty(telemetry.thermal_state)
        if thermal_penalty:
            reasons.append(f"thermal state is {telemetry.thermal_state.value}")
            penalty += thermal_penalty

        if telemetry.heavy_processes:
            total_heavy_mb = sum(process.memory_rss_mb for process in telemetry.heavy_processes[:5])
            if total_heavy_mb >= memory.heavy_process_pressure_mb:
                reasons.append(f"heavy developer processes consume {total_heavy_mb:.0f} MiB")
                penalty += 10

        global_juice = max(
            self.config.minimum_background_juice_level,
            self.config.background_base_juice_level - penalty,
        )
        if yield_reason:
            system_state = SystemState.YIELDING
        elif penalty >= 30:
            system_state = SystemState.PRESSURED
        elif penalty > 0:
            system_state = SystemState.RECOVERING
        else:
            system_state = SystemState.NORMAL

        return PressureReport(
            system_state=system_state,
            global_juice_level=global_juice,
            yield_active=yield_reason is not None,
            yield_reason=yield_reason,
            reasons=tuple(reasons) or ("system pressure normal",),
            dynamic_yield_percent=dynamic_yield_percent,
            dynamic_recover_percent=dynamic_recover_percent,
            memory_budget_scale=memory.budget_scale,
            total_ram_gb=memory.total_gb,
        )

    def budget_for(self, request_class: RequestClass, report: PressureReport) -> JuiceBudget:
        """Return a compute budget for the request class."""

        if request_class == RequestClass.USER_PROMPT:
            return JuiceBudget(
                juice_level=self.config.foreground_juice_level,
                max_context_tokens=int(
                    self.config.base_context_tokens * report.memory_budget_scale
                ),
                max_output_tokens=int(self.config.base_output_tokens * report.memory_budget_scale),
                concurrency_limit=self.config.max_foreground_concurrency,
                reason="foreground prompt receives full budget",
            ).clamped()

        fraction = max(0.05, (report.global_juice_level / 100) * report.memory_budget_scale)
        return JuiceBudget(
            juice_level=report.global_juice_level,
            max_context_tokens=int(self.config.base_context_tokens * fraction),
            max_output_tokens=int(self.config.base_output_tokens * fraction),
            concurrency_limit=max(
                1, min(self.config.max_background_concurrency, round(3 * fraction))
            ),
            reason="; ".join(report.reasons),
        ).clamped()

    @staticmethod
    def _thermal_penalty(state: ThermalState) -> int:
        if state == ThermalState.CRITICAL:
            return 45
        if state == ThermalState.HOT:
            return 30
        if state == ThermalState.WARM:
            return 10
        return 0
