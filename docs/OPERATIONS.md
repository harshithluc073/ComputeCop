# ComputeCop Operations

## Architecture

ComputeCop runs as a local-only FastAPI proxy in front of local inference
engines. The runtime graph has eight cooperating parts:

1. `PsutilTelemetrySampler` collects CPU, RAM, swap, disk, thermal, and heavy
   process signals.
2. `TelemetryLoop` smooths samples and publishes snapshots without blocking the
   asyncio event loop.
3. `JuicePolicyEngine` converts host pressure into a global `juice_level` and
   request-specific budgets.
4. `RequestClassifier` separates direct user prompts from automated background
   API requests.
5. `AdmissionController` decides whether requests are allowed, throttled,
   rejected, or held during RAM-yield pressure.
6. `AdaptiveScheduler` reserves foreground concurrency slots, governs background
   work against spare capacity, and applies queue aging to reduce starvation.
7. `AsyncRequestQueue` stores throttled background traffic with bounded priority
   ordering and live queue counters.
8. `UpstreamRouter` forwards accepted traffic to Ollama, llama.cpp, or
   OpenAI-compatible local endpoints.
9. `Dashboard` renders live host, scheduler capacity, and policy state in a Rich
   terminal interface.

## Prompt Versus Request Semantics

ComputeCop treats a prompt as foreground user intent and an API request as
background automation. Explicit headers always win:

- `x-computecop-class: prompt` or `x-computecop-priority: foreground` marks a
  direct prompt.
- `x-computecop-background: true`, `x-agent-request: true`, or
  `x-automation-request: true` marks background work.
- Payload metadata can also set `metadata.computecop_class`,
  `metadata.priority`, `metadata.interactive`, or `metadata.background`.

Ambiguous automation-looking traffic is classified as background to protect
foreground responsiveness.

## Decision Explainability

Each admission decision carries a structured policy trace. The trace records the
telemetry values, dynamic thresholds, triggered rules, penalties, request class,
queue position, and final shaped budget used for the decision.

Response headers include:

```text
x-computecop-correlation-id
x-computecop-trace-id
x-computecop-decision
x-computecop-juice-level
```

Recent decisions can be inspected through:

```text
GET /decisions/{correlation_id}
```

The Rich dashboard also includes a "Why" panel that shows the most recent policy
rules and penalties without exposing prompt or completion bodies.

## Juice Level

`juice_level` is a 1-100 compute budget. Foreground prompts receive the full
configured foreground budget. Background requests are reduced as pressure rises:

- RAM above the recovery threshold reduces context and output budgets.
- RAM above 85% activates yield mode by default.
- High CPU, swap pressure, hot thermal state, and heavy developer processes
  reduce the budget further.

For OpenAI-compatible calls, ComputeCop shapes `max_tokens` and attaches budget
metadata. For Ollama it shapes `options.num_ctx`, `options.num_predict`, and
`keep_alive`. For llama.cpp it shapes `n_ctx`, `n_predict`, and `cache_prompt`.

When the machine is pressured but not yielding, background requests are
throttled and executed through the bounded async queue. Foreground prompts still
skip that queue. If the queue is full or the queued request expires, ComputeCop
returns an explicit retryable error response with the original correlation ID.

## Queue Lifecycle

The background request queue exposes a lifecycle state through `GET /state`:

| State | Meaning |
| --- | --- | --- |
| `accepting` | Background work may enter the queue. |
| `paused` | New background work is rejected; queued work continues. |
| `draining` | New background work is rejected while existing queued work finishes. |
| `closed` | The queue is shut down and pending work is cancelled. |

`/state` also reports per-worker status for each background worker:

- worker ID
- worker state: `idle`, `running`, `failed`, `stopping`, or `stopped`
- active correlation ID while a worker is executing safe queue metadata

The Rich dashboard includes a Queue Workers panel with the same information.

## Adaptive Scheduler

Starting in v0.2.0, ComputeCop routes accepted proxy work through an adaptive
scheduler that sits above the async request queue:

- **Foreground reservation**: `policy.max_foreground_concurrency` slots are
  reserved for direct prompts and interactive traffic. Foreground work does not
  wait behind queued background batches.
- **Spare-capacity background execution**: background work may use only spare
  slots up to `policy.max_background_concurrency` and the current
  `effective_background_slots` value.
- **Pressure-aware shrinking**: when the host is recovering, pressured, or
  yielding, the scheduler lowers `effective_background_slots` before admitting
  more background execution.
- **Queue aging**: long-waiting queued background items gain scheduling priority
  on a configurable interval (`queue.aging_interval_seconds`) so bulk traffic
  cannot starve indefinitely.

`GET /state` exposes a `scheduler` object alongside queue counters:

| Field | Meaning |
| --- | --- |
| `reserved_foreground_slots` | Effective foreground slot ceiling after pressure shaping. |
| `effective_foreground_slots` | Same as `reserved_foreground_slots` for scheduler snapshots. |
| `max_background_slots` | Configured background concurrency ceiling. |
| `effective_background_slots` | Live background limit after pressure shaping. |
| `running_foreground` | Foreground executions currently holding slots. |
| `running_background` | Background executions currently holding slots. |
| `total_capacity` | Combined foreground and background slot budget. |
| `spare_slots` | Unused capacity across both classes. |

The Rich dashboard Policy panel mirrors these scheduler counters for live
operations.

## Concurrency Governor

Starting in v0.2.5, ComputeCop adds a second concurrency layer that limits
simultaneous upstream requests per endpoint and request class:

- **Per-endpoint semaphores**: `policy.max_endpoint_foreground_concurrency` and
  `policy.max_endpoint_background_concurrency` cap how many concurrent requests
  each configured endpoint may serve.
- **Acquire before forward**: the proxy acquires endpoint capacity after route
  selection and before upstream forwarding. Streaming responses release capacity
  when the stream completes, errors, or the client disconnects.
- **Dynamic concurrency limits**: policy computes recommended global and
  per-endpoint ceilings from RAM, swap, thermal, and open circuit-breaker
  pressure. The scheduler and endpoint governor both honor these live limits.

`GET /state` exposes a `concurrency` object when the governor is active:

| Field | Meaning |
| --- | --- |
| `limits.max_foreground` | Effective global foreground slot ceiling. |
| `limits.max_background` | Effective global background slot ceiling. |
| `limits.max_endpoint_foreground` | Per-endpoint foreground ceiling. |
| `limits.max_endpoint_background` | Per-endpoint background ceiling. |
| `limits.reasons` | Why limits were shaped. |
| `endpoints[].running_foreground` | Foreground requests currently using an endpoint. |
| `endpoints[].running_background` | Background requests currently using an endpoint. |

The scheduler `effective_foreground_slots` field now reflects live policy
shaping instead of only the configured maximum.

## Graceful Shutdown

ComputeCop drains the background queue before closing during normal shutdown.
The drain window is controlled by `queue.shutdown_drain_seconds` in TOML or the
built-in default of five seconds. During drain, the queue enters the `draining`
state, rejects new background submissions, and waits for in-flight queued work
to finish before closing workers and upstream HTTP clients.

Shutdown is idempotent: repeated stop requests, duplicate upstream closes, and
repeated Ctrl+C handling do not raise tracebacks or double-close clients. The
terminal dashboard uses the same shutdown coordinator as the proxy lifespan.

## Dynamic RAM Yield

ComputeCop derives RAM yield and recovery thresholds from total host memory.
`COMPUTECOP_RAM_YIELD_PERCENT` is an upper cap, not a fixed assumption. Smaller
machines reserve a larger share of memory for the operating system and
foreground applications, while larger machines can tolerate more absolute model
memory before yielding.

The supported baseline is 6GB RAM. Hosts below that baseline are still handled
gracefully, but policy applies additional pressure penalties and reduces token
budgets more aggressively.

When the dynamic yield threshold is crossed, ComputeCop:

- Continues to admit foreground prompts.
- Returns retryable yield responses for background requests.
- Calls best-effort model offload hooks for supported local engines.
- Recovers only after RAM usage drops to the lower hysteresis threshold.

The recovery threshold is also dynamic and stays below the yield threshold by
the configured hysteresis gap, preventing rapid state flapping.

## Configuration Precedence

ComputeCop resolves settings in this order:

1. Built-in defaults
2. TOML configuration file from `COMPUTECOP_CONFIG` or `--config`
3. Environment variables
4. CLI overrides on `computecop run`

Use `computecop config` to print the effective runtime configuration and
`computecop config explain` to see each value with its source. Add `--json` to
`config explain` for support-friendly machine output.

Example TOML file: `examples/computecop.toml`.

## Endpoint Configuration

Default endpoints are:

- Ollama: `http://127.0.0.1:11434`
- llama.cpp: `http://127.0.0.1:8080`

Override endpoints with `COMPUTECOP_ENDPOINTS`, a JSON list matching
`EndpointConfig` fields. Use `x-computecop-endpoint` to route a request to a
specific configured endpoint by name.

## Endpoint Capability Registry

Starting in v0.2.1, ComputeCop maintains a capability registry for every
configured upstream endpoint. The registry records API family, streaming support,
model-list support, offload support, default context/output hints, cached health
status, probe latency, failure streak, and a rolling failure rate.

Capability health probes are cached with a TTL controlled by
`endpoint_registry.capability_probe_ttl_seconds` (default 30 seconds). Pass
`?refresh=true` to `GET /endpoints` to force a fresh probe cycle.

```text
GET /endpoints
GET /endpoints?refresh=true
```

Each endpoint record includes:

| Section | Fields |
| --- | --- |
| `capabilities` | `api_family`, `supports_streaming`, `supports_model_list`, `supports_offload`, `default_context_tokens`, `default_output_tokens` |
| `health` | `healthy`, `status_code`, `latency_ms`, `failure_rate`, `failure_streak`, `last_success_at`, `checked_at`, `detail`, `stale` |
| `routing` | `is_default`, `explicit_header`, `compatible_api_families` |

When a request does not specify `x-computecop-endpoint`, ComputeCop selects a
compatible endpoint for the incoming API family. Healthy endpoints with lower
failure rates are preferred. Streaming requests require an endpoint with
`supports_streaming=true`. Endpoints with an open circuit breaker are excluded
from automatic selection.

## Endpoint Health Watcher

Starting in v0.2.2, ComputeCop runs a background health watcher that proactively
probes every configured endpoint on a fixed interval with randomized jitter. The
watcher keeps health snapshots warm so routing decisions can use recent probe
data instead of waiting for the next on-demand request.

| Setting | Default | Description |
| --- | --- | --- |
| `endpoint_registry.health_watcher_enabled` | `true` | Enable the background probe loop. |
| `endpoint_registry.health_watcher_interval_seconds` | `15` | Base seconds between probe cycles. |
| `endpoint_registry.health_watcher_jitter_fraction` | `0.1` | Random jitter fraction applied to the interval. |

Each probe records latency, failure streak, last success time, and a failure
status category when the endpoint is unhealthy. Disable the watcher only when an
external orchestrator is already probing the same endpoints.

## Endpoint Circuit Breakers

ComputeCop tracks per-endpoint circuit breaker state to avoid repeatedly routing
work to a failing upstream:

| State | Traffic allowed | Meaning |
| --- | --- | --- |
| `closed` | yes | Normal operation. |
| `open` | no | Too many consecutive probe or request failures. |
| `half_open` | yes | Cool-down elapsed; one successful probe closes the breaker. |

| Setting | Default | Description |
| --- | --- | --- |
| `endpoint_registry.circuit_breaker_failure_threshold` | `3` | Consecutive failures before opening. |
| `endpoint_registry.circuit_breaker_cooldown_seconds` | `30` | Seconds before entering `half_open`. |
| `endpoint_registry.circuit_breaker_half_open_successes` | `1` | Successes required to close from `half_open`. |

Both background probes and proxied upstream requests update breaker state. When
every compatible breaker is open, ComputeCop returns a retryable `503` with
remediation guidance instead of hammering a dead endpoint. Inspect breaker state
through `GET /endpoints` under each endpoint's `health.circuit_breaker` object.

## Upstream Failure Categories

When an upstream endpoint cannot serve a request, ComputeCop converts the
transport error into a typed failure instead of a generic 502. Each failure
carries a category, an HTTP status, a retryability flag, and a remediation hint.
The proxy returns the failure as a normalized error response with an
`error.type` of `computecop_upstream_<category>` and an `upstream_failure`
object, and records a prompt-free `upstream.failure` event.

| Category | Status | Retryable | Meaning |
| --- | --- | --- | --- |
| `unreachable` | 502 | yes | The endpoint refused the connection or could not be reached. |
| `timeout` | 504 | yes | The endpoint accepted the connection but did not respond in time. |
| `route_not_found` | 400 | no | The requested `x-computecop-endpoint` is not configured. |
| `status_error` | upstream status | only for 408/425/429/5xx | The endpoint returned an HTTP error status. |
| `stream_interrupted` | 502 | yes | A streaming response closed before completing. |
| `invalid_response` | 502 | varies | The endpoint returned a malformed or undecodable response. |
| `misconfigured_endpoint` | 502 | no | The configured `base_url` uses an invalid or unsupported scheme. |

Retryable failures include a `retry-after` header so clients can back off and
try again.

## Endpoint Probe Diagnostics

`computecop probe` reports practical health detail for each configured endpoint,
not just a yes/no result:

- measured probe latency in milliseconds
- consecutive failure streak since the last successful probe
- last successful probe time
- the failure category when a probe fails

This makes the command a first-line debugging tool: a high failure streak with
an `unreachable` category points at a stopped engine, while a `timeout` category
points at a slow or overloaded model.

## Event Store Reliability

ComputeCop persists prompt-free runtime events — `admission.decision`,
`policy.yield`, and `upstream.failure` — to a bounded JSONL log. The store is an
audit trail for recent throttling and yield behavior.

Persistence is hardened for unattended operation:

- **Durable appends.** Each event is appended and then flushed and `fsync`-ed
  before the write returns, so a recorded event survives an abrupt process exit.
- **Bounded retention.** The log is trimmed to its retention limit so it cannot
  grow without bound.
- **Graceful degradation.** If the configured event path cannot be created or
  written — for example a permission failure or a path that collides with an
  existing file — ComputeCop disables persistence, records the failure reason,
  and keeps serving traffic instead of crashing. Subsequent appends are silent
  no-ops until the process restarts.
- **Operator visibility.** When persistence is disabled, the dashboard renders a
  red warning panel showing the reason, and the runtime `/state` snapshot exposes
  an `event_persistence` object with `enabled` and `disabled_reason` fields.

Set the log location with `COMPUTECOP_EVENT_LOG`; it defaults to the per-user
cache directory.

## Event Query Commands

Diagnose recent runtime behavior without manually opening JSONL files:

```powershell
computecop events tail
computecop events tail --limit 50
computecop events find --correlation-id <correlation-or-trace-id>
computecop events stats
```

- `events tail` prints the most recent events (default 20, `--limit`/`-n` to
  adjust).
- `events find` returns every event that references the given correlation or
  trace ID, including IDs nested inside event payloads such as the
  `admission.decision` payload's `decision.correlation_id`.
- `events stats` aggregates event counts by kind and reports the earliest and
  latest recorded timestamps.

Every event command accepts `--json` for machine-readable output suitable for
piping into other tools.

## Commands

```powershell
computecop run
computecop dashboard
computecop config
computecop config explain
computecop config explain --json
computecop telemetry
computecop probe
computecop events tail
computecop events find --correlation-id <id>
computecop events stats
```

The proxy binds to `127.0.0.1:8765` by default. Binding to a non-local address
requires `COMPUTECOP_EXPOSE_REMOTE=true`.
