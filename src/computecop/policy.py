"""Juice-level policy engine."""

from __future__ import annotations

from dataclasses import dataclass, replace

from computecop.config import PolicyConfig
from computecop.models import (
    JuiceBudget,
    PolicyRuleEvent,
    PolicyRuleStatus,
    PolicyTrace,
    RequestClass,
    ResourcePressureBreakdown,
    SystemState,
    TelemetrySample,
    ThermalState,
)
from computecop.platform import HostMemoryProfile


@dataclass(frozen=True, slots=True)
class ConcurrencyLimits:
    """Recommended global and per-endpoint concurrency ceilings after pressure shaping."""

    max_foreground: int
    max_background: int
    max_endpoint_foreground: int
    max_endpoint_background: int
    reasons: tuple[str, ...]


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
    trace: PolicyTrace
    concurrency_limits: ConcurrencyLimits


class JuicePolicyEngine:
    """Map resource pressure to foreground and background inference budgets."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        telemetry: TelemetrySample | None,
        *,
        open_circuit_breaker_count: int = 0,
    ) -> PressureReport:
        """Evaluate global system pressure."""

        if telemetry is None:
            report = PressureReport(
                system_state=SystemState.NORMAL,
                global_juice_level=self.config.background_base_juice_level,
                yield_active=False,
                yield_reason=None,
                reasons=("telemetry unavailable; using conservative defaults",),
                dynamic_yield_percent=self.config.ram_yield_percent,
                dynamic_recover_percent=self.config.ram_recover_percent,
                memory_budget_scale=1.0,
                total_ram_gb=None,
                trace=PolicyTrace(
                    rules=(
                        PolicyRuleEvent(
                            name="telemetry",
                            status=PolicyRuleStatus.UNAVAILABLE,
                            observed=None,
                            threshold=None,
                            penalty=0,
                            detail="telemetry unavailable; using conservative defaults",
                        ),
                    ),
                    summary="telemetry unavailable; using conservative defaults",
                ),
                concurrency_limits=_default_concurrency_limits(self.config),
            )
            return _with_concurrency_limits(
                report,
                self.config,
                open_circuit_breaker_count=open_circuit_breaker_count,
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
        rules: list[PolicyRuleEvent] = []
        penalty = 0
        yield_reason: str | None = None
        heavy_process_rss_mb = sum(
            process.memory_rss_mb for process in telemetry.heavy_processes[:5]
        )

        if not memory.meets_minimum:
            detail = (
                f"RAM capacity {memory.total_gb:.1f} GiB is below "
                f"{self.config.minimum_supported_ram_gb:.1f} GiB baseline"
            )
            reasons.append(detail)
            penalty += 25
            rules.append(
                _rule(
                    name="memory_capacity",
                    triggered=True,
                    observed=round(memory.total_gb, 2),
                    threshold=self.config.minimum_supported_ram_gb,
                    penalty=25,
                    detail=detail,
                )
            )
        else:
            rules.append(
                _rule(
                    name="memory_capacity",
                    triggered=False,
                    observed=round(memory.total_gb, 2),
                    threshold=self.config.minimum_supported_ram_gb,
                    penalty=0,
                    detail="host RAM meets configured baseline",
                )
            )

        if telemetry.ram_used_percent >= dynamic_yield_percent:
            yield_reason = (
                f"RAM usage {telemetry.ram_used_percent:.1f}% exceeds "
                f"dynamic yield threshold {dynamic_yield_percent:.1f}%"
            )
            reasons.append(yield_reason)
            penalty += 55
            rules.append(
                _rule(
                    name="ram_yield",
                    triggered=True,
                    observed=round(telemetry.ram_used_percent, 2),
                    threshold=round(dynamic_yield_percent, 2),
                    penalty=55,
                    detail=yield_reason,
                )
            )
        elif telemetry.ram_used_percent >= dynamic_recover_percent:
            detail = f"RAM usage elevated at {telemetry.ram_used_percent:.1f}%"
            reasons.append(detail)
            penalty += 25
            rules.append(
                _rule(
                    name="ram_recover",
                    triggered=True,
                    observed=round(telemetry.ram_used_percent, 2),
                    threshold=round(dynamic_recover_percent, 2),
                    penalty=25,
                    detail=detail,
                )
            )
        else:
            rules.append(
                _rule(
                    name="ram_pressure",
                    triggered=False,
                    observed=round(telemetry.ram_used_percent, 2),
                    threshold=round(dynamic_recover_percent, 2),
                    penalty=0,
                    detail="RAM pressure below dynamic recovery threshold",
                )
            )

        if telemetry.cpu_percent >= self.config.cpu_pressure_percent:
            detail = f"CPU usage elevated at {telemetry.cpu_percent:.1f}%"
            reasons.append(detail)
            penalty += 15
            rules.append(
                _rule(
                    name="cpu_pressure",
                    triggered=True,
                    observed=round(telemetry.cpu_percent, 2),
                    threshold=self.config.cpu_pressure_percent,
                    penalty=15,
                    detail=detail,
                )
            )
        else:
            rules.append(
                _rule(
                    name="cpu_pressure",
                    triggered=False,
                    observed=round(telemetry.cpu_percent, 2),
                    threshold=self.config.cpu_pressure_percent,
                    penalty=0,
                    detail="CPU pressure below configured threshold",
                )
            )

        if telemetry.swap_used_percent >= self.config.swap_pressure_percent:
            detail = f"swap usage elevated at {telemetry.swap_used_percent:.1f}%"
            reasons.append(detail)
            penalty += 20
            rules.append(
                _rule(
                    name="swap_pressure",
                    triggered=True,
                    observed=round(telemetry.swap_used_percent, 2),
                    threshold=self.config.swap_pressure_percent,
                    penalty=20,
                    detail=detail,
                )
            )
        else:
            rules.append(
                _rule(
                    name="swap_pressure",
                    triggered=False,
                    observed=round(telemetry.swap_used_percent, 2),
                    threshold=self.config.swap_pressure_percent,
                    penalty=0,
                    detail="swap pressure below configured threshold",
                )
            )

        thermal_penalty = self._thermal_penalty(telemetry.thermal_state)
        if thermal_penalty:
            detail = f"thermal state is {telemetry.thermal_state.value}"
            reasons.append(detail)
            penalty += thermal_penalty
            rules.append(
                _rule(
                    name="thermal_pressure",
                    triggered=True,
                    observed=telemetry.thermal_state.value,
                    threshold="warm",
                    penalty=thermal_penalty,
                    detail=detail,
                )
            )
        else:
            rules.append(
                _rule(
                    name="thermal_pressure",
                    triggered=False,
                    observed=telemetry.thermal_state.value,
                    threshold="warm",
                    penalty=0,
                    detail="thermal pressure below policy threshold",
                )
            )

        if telemetry.heavy_processes:
            if heavy_process_rss_mb >= memory.heavy_process_pressure_mb:
                detail = f"heavy developer processes consume {heavy_process_rss_mb:.0f} MiB"
                reasons.append(detail)
                penalty += 10
                rules.append(
                    _rule(
                        name="heavy_process_pressure",
                        triggered=True,
                        observed=round(heavy_process_rss_mb, 2),
                        threshold=round(memory.heavy_process_pressure_mb, 2),
                        penalty=10,
                        detail=detail,
                    )
                )
            else:
                rules.append(
                    _rule(
                        name="heavy_process_pressure",
                        triggered=False,
                        observed=round(heavy_process_rss_mb, 2),
                        threshold=round(memory.heavy_process_pressure_mb, 2),
                        penalty=0,
                        detail="heavy process memory below host-relative threshold",
                    )
                )
        else:
            rules.append(
                _rule(
                    name="heavy_process_pressure",
                    triggered=False,
                    observed=0.0,
                    threshold=round(memory.heavy_process_pressure_mb, 2),
                    penalty=0,
                    detail="no heavy developer processes detected",
                )
            )

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

        summary = "; ".join(reasons) if reasons else "system pressure normal"
        trace = PolicyTrace(
            pressure=ResourcePressureBreakdown(
                ram_used_percent=telemetry.ram_used_percent,
                ram_total_gb=memory.total_gb,
                ram_available_gb=telemetry.ram_available_gb,
                dynamic_yield_percent=dynamic_yield_percent,
                dynamic_recover_percent=dynamic_recover_percent,
                cpu_percent=telemetry.cpu_percent,
                cpu_threshold_percent=self.config.cpu_pressure_percent,
                swap_used_percent=telemetry.swap_used_percent,
                swap_threshold_percent=self.config.swap_pressure_percent,
                thermal_state=telemetry.thermal_state,
                heavy_process_rss_mb=heavy_process_rss_mb,
                heavy_process_threshold_mb=memory.heavy_process_pressure_mb,
            ),
            rules=tuple(rules),
            system_state=system_state,
            global_juice_level=global_juice,
            yield_active=yield_reason is not None,
            summary=summary,
        )

        report = PressureReport(
            system_state=system_state,
            global_juice_level=global_juice,
            yield_active=yield_reason is not None,
            yield_reason=yield_reason,
            reasons=tuple(reasons) or ("system pressure normal",),
            dynamic_yield_percent=dynamic_yield_percent,
            dynamic_recover_percent=dynamic_recover_percent,
            memory_budget_scale=memory.budget_scale,
            total_ram_gb=memory.total_gb,
            trace=trace,
            concurrency_limits=_default_concurrency_limits(self.config),
        )
        return _with_concurrency_limits(
            report,
            self.config,
            open_circuit_breaker_count=open_circuit_breaker_count,
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


def _default_concurrency_limits(policy_config: PolicyConfig) -> ConcurrencyLimits:
    return ConcurrencyLimits(
        max_foreground=policy_config.max_foreground_concurrency,
        max_background=policy_config.max_background_concurrency,
        max_endpoint_foreground=policy_config.max_endpoint_foreground_concurrency,
        max_endpoint_background=policy_config.max_endpoint_background_concurrency,
        reasons=("configured concurrency ceilings",),
    )


def compute_concurrency_limits(
    report: PressureReport,
    policy_config: PolicyConfig,
    *,
    open_circuit_breaker_count: int = 0,
) -> ConcurrencyLimits:
    """Compute recommended concurrency limits from pressure and breaker state."""

    reasons: list[str] = []
    foreground = policy_config.max_foreground_concurrency
    background = policy_config.max_background_concurrency
    endpoint_foreground = policy_config.max_endpoint_foreground_concurrency
    endpoint_background = policy_config.max_endpoint_background_concurrency

    if report.yield_active:
        background = 0
        foreground = max(1, foreground // 2)
        endpoint_background = 0
        endpoint_foreground = max(1, endpoint_foreground // 2)
        reasons.append("RAM yield active; background concurrency disabled")
    elif report.system_state == SystemState.PRESSURED:
        background = max(1, background // 2)
        foreground = max(1, (foreground * 3) // 4)
        endpoint_background = max(1, endpoint_background // 2)
        endpoint_foreground = max(1, (endpoint_foreground * 3) // 4)
        reasons.append("host pressured; concurrency reduced")
    elif report.system_state == SystemState.RECOVERING:
        background = max(1, int(background * 0.75))
        endpoint_background = max(1, int(endpoint_background * 0.75))
        reasons.append("host recovering; background concurrency reduced")

    triggered_rules = {
        rule.name for rule in report.trace.rules if rule.status is PolicyRuleStatus.TRIGGERED
    }

    if "swap_pressure" in triggered_rules:
        background = max(0 if report.yield_active else 1, background // 2)
        endpoint_background = max(0 if report.yield_active else 1, endpoint_background // 2)
        reasons.append("swap pressure elevated; concurrency reduced")

    if "thermal_pressure" in triggered_rules:
        foreground = max(1, foreground - 1)
        background = max(0 if report.yield_active else 1, background - 1)
        endpoint_foreground = max(1, endpoint_foreground - 1)
        endpoint_background = max(0 if report.yield_active else 1, endpoint_background - 1)
        reasons.append("thermal pressure elevated; concurrency reduced")

    if open_circuit_breaker_count > 0:
        endpoint_foreground = max(1, endpoint_foreground - open_circuit_breaker_count)
        endpoint_background = max(
            0 if report.yield_active else 1,
            endpoint_background - open_circuit_breaker_count,
        )
        reasons.append(
            f"{open_circuit_breaker_count} endpoint circuit breaker(s) open; "
            "per-endpoint concurrency reduced"
        )

    if not reasons:
        reasons.append("configured concurrency ceilings")

    return ConcurrencyLimits(
        max_foreground=foreground,
        max_background=background,
        max_endpoint_foreground=endpoint_foreground,
        max_endpoint_background=endpoint_background,
        reasons=tuple(reasons),
    )


def _with_concurrency_limits(
    report: PressureReport,
    policy_config: PolicyConfig,
    *,
    open_circuit_breaker_count: int,
) -> PressureReport:
    limits = compute_concurrency_limits(
        report,
        policy_config,
        open_circuit_breaker_count=open_circuit_breaker_count,
    )
    return replace(report, concurrency_limits=limits)


def _rule(
    *,
    name: str,
    triggered: bool,
    observed: float | str | None,
    threshold: float | str | None,
    penalty: int,
    detail: str,
) -> PolicyRuleEvent:
    return PolicyRuleEvent(
        name=name,
        status=PolicyRuleStatus.TRIGGERED if triggered else PolicyRuleStatus.OBSERVED,
        observed=observed,
        threshold=threshold,
        penalty=penalty if triggered else 0,
        detail=detail,
    )
