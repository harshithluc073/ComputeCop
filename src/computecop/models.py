"""Core domain models shared by ComputeCop subsystems."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class RequestClass(str, Enum):
    """Semantic class of an incoming inference call."""

    USER_PROMPT = "user_prompt"
    BACKGROUND_REQUEST = "background_request"
    UNKNOWN = "unknown"


class RequestPriority(str, Enum):
    """Scheduling priority for work admitted through the proxy."""

    FOREGROUND = "foreground"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    BULK = "bulk"


class EndpointKind(str, Enum):
    """Supported upstream inference endpoint families."""

    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"


class ThermalState(str, Enum):
    """Thermal risk state derived from available sensor data."""

    UNKNOWN = "unknown"
    COOL = "cool"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"


class SystemState(str, Enum):
    """Global compute availability state."""

    NORMAL = "normal"
    PRESSURED = "pressured"
    YIELDING = "yielding"
    RECOVERING = "recovering"


class DecisionType(str, Enum):
    """Admission outcome for an incoming request."""

    ALLOW = "allow"
    THROTTLE = "throttle"
    QUEUE = "queue"
    REJECT = "reject"
    YIELD = "yield"


class QueueLifecycleState(str, Enum):
    """Lifecycle state for the background request queue."""

    ACCEPTING = "accepting"
    PAUSED = "paused"
    DRAINING = "draining"
    CLOSED = "closed"


class WorkerState(str, Enum):
    """Observed state for a queue worker task."""

    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class PolicyRuleStatus(str, Enum):
    """Whether a policy rule contributed pressure."""

    OBSERVED = "observed"
    TRIGGERED = "triggered"
    UNAVAILABLE = "unavailable"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(tz=UTC)


def new_correlation_id() -> str:
    """Return a short, URL-safe request correlation identifier."""

    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class ProcessSample:
    """Small process telemetry sample for a local heavy process."""

    pid: int
    name: str
    cpu_percent: float
    memory_rss_bytes: int
    command: str = ""

    @property
    def memory_rss_mb(self) -> float:
        return self.memory_rss_bytes / 1024 / 1024


@dataclass(frozen=True, slots=True)
class TemperatureSample:
    """Temperature reading from a hardware sensor."""

    label: str
    current_celsius: float
    high_celsius: float | None = None
    critical_celsius: float | None = None


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """Point-in-time system telemetry snapshot."""

    timestamp: datetime
    cpu_percent: float
    cpu_per_core_percent: tuple[float, ...]
    ram_total_bytes: int
    ram_available_bytes: int
    ram_used_percent: float
    swap_used_percent: float
    disk_read_bytes_per_sec: float
    disk_write_bytes_per_sec: float
    thermal_state: ThermalState
    temperatures: tuple[TemperatureSample, ...] = field(default_factory=tuple)
    heavy_processes: tuple[ProcessSample, ...] = field(default_factory=tuple)

    @property
    def ram_available_gb(self) -> float:
        return self.ram_available_bytes / 1024 / 1024 / 1024

    @property
    def ram_total_gb(self) -> float:
        return self.ram_total_bytes / 1024 / 1024 / 1024


@dataclass(frozen=True, slots=True)
class JuiceBudget:
    """Compute budget applied to inference work."""

    juice_level: int
    max_context_tokens: int
    max_output_tokens: int
    concurrency_limit: int
    reason: str

    def clamped(self) -> JuiceBudget:
        return JuiceBudget(
            juice_level=max(1, min(100, self.juice_level)),
            max_context_tokens=max(512, self.max_context_tokens),
            max_output_tokens=max(32, self.max_output_tokens),
            concurrency_limit=max(1, self.concurrency_limit),
            reason=self.reason,
        )


@dataclass(frozen=True, slots=True)
class PolicyRuleEvent:
    """One explainable policy rule evaluation."""

    name: str
    status: PolicyRuleStatus
    observed: float | str | None
    threshold: float | str | None
    penalty: int
    detail: str


@dataclass(frozen=True, slots=True)
class ResourcePressureBreakdown:
    """Telemetry values and dynamic thresholds used by policy."""

    ram_used_percent: float | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    dynamic_yield_percent: float
    dynamic_recover_percent: float
    cpu_percent: float | None
    cpu_threshold_percent: float
    swap_used_percent: float | None
    swap_threshold_percent: float
    thermal_state: ThermalState
    heavy_process_rss_mb: float
    heavy_process_threshold_mb: float


@dataclass(frozen=True, slots=True)
class PolicyTrace:
    """Structured explanation of a policy and admission decision."""

    trace_id: str = field(default_factory=new_correlation_id)
    created_at: datetime = field(default_factory=utc_now)
    pressure: ResourcePressureBreakdown | None = None
    rules: tuple[PolicyRuleEvent, ...] = field(default_factory=tuple)
    system_state: SystemState = SystemState.NORMAL
    global_juice_level: int = 100
    yield_active: bool = False
    request_class: RequestClass | None = None
    priority: RequestPriority | None = None
    decision: DecisionType | None = None
    endpoint_name: str | None = None
    path: str | None = None
    queue_size: int | None = None
    queue_position: int | None = None
    final_juice_level: int | None = None
    shaped_context_tokens: int | None = None
    shaped_output_tokens: int | None = None
    summary: str = "policy trace initialized"

    def with_admission(
        self,
        *,
        request_class: RequestClass,
        priority: RequestPriority,
        decision: DecisionType,
        endpoint_name: str | None,
        path: str,
        queue_size: int,
        queue_position: int | None,
        budget: JuiceBudget,
        summary: str,
    ) -> PolicyTrace:
        """Return a copy enriched with request admission context."""

        return PolicyTrace(
            trace_id=self.trace_id,
            created_at=self.created_at,
            pressure=self.pressure,
            rules=self.rules,
            system_state=self.system_state,
            global_juice_level=self.global_juice_level,
            yield_active=self.yield_active,
            request_class=request_class,
            priority=priority,
            decision=decision,
            endpoint_name=endpoint_name,
            path=path,
            queue_size=queue_size,
            queue_position=queue_position,
            final_juice_level=budget.juice_level,
            shaped_context_tokens=budget.max_context_tokens,
            shaped_output_tokens=budget.max_output_tokens,
            summary=summary,
        )


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Rich result detailing the request classification decision."""

    request_class: RequestClass
    priority: RequestPriority
    confidence_score: float
    matched_signals: tuple[str, ...] = field(default_factory=tuple)
    ambiguous_signals: tuple[str, ...] = field(default_factory=tuple)
    recommended_header_fixes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    """Normalized metadata used by policy and routing logic."""

    method: str
    path: str
    headers: dict[str, str]
    request_class: RequestClass
    priority: RequestPriority
    correlation_id: str = field(default_factory=new_correlation_id)
    client_host: str | None = None
    model: str | None = None
    endpoint_name: str | None = None
    received_at: datetime = field(default_factory=utc_now)
    classification: ClassificationResult | None = None

    def header(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name.lower(), default)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Decision returned by the admission controller."""

    decision: DecisionType
    request_class: RequestClass
    priority: RequestPriority
    budget: JuiceBudget
    reason: str
    correlation_id: str
    retry_after_seconds: float | None = None
    queue_position: int | None = None
    trace: PolicyTrace | None = None
    classification: ClassificationResult | None = None

    @property
    def allowed(self) -> bool:
        return self.decision in {DecisionType.ALLOW, DecisionType.THROTTLE}


@dataclass(frozen=True, slots=True)
class EndpointRoute:
    """Resolved upstream endpoint target."""

    name: str
    kind: EndpointKind
    base_url: str
    timeout_seconds: float
    health_path: str
    supports_streaming: bool = True


def to_jsonable(value: Any) -> Any:
    """Convert ComputeCop dataclasses and enums into JSON-safe values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
