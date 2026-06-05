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
6. `AsyncRequestQueue` executes throttled background traffic with bounded
   priority ordering and live queue counters.
7. `UpstreamRouter` forwards accepted traffic to Ollama, llama.cpp, or
   OpenAI-compatible local endpoints.
8. `Dashboard` renders live host and policy state in a Rich terminal interface.

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

## Endpoint Configuration

Default endpoints are:

- Ollama: `http://127.0.0.1:11434`
- llama.cpp: `http://127.0.0.1:8080`

Override endpoints with `COMPUTECOP_ENDPOINTS`, a JSON list matching
`EndpointConfig` fields. Use `x-computecop-endpoint` to route a request to a
specific configured endpoint by name.

## Commands

```powershell
computecop run
computecop dashboard
computecop config
computecop telemetry
computecop probe
```

The proxy binds to `127.0.0.1:8765` by default. Binding to a non-local address
requires `COMPUTECOP_EXPOSE_REMOTE=true`.
