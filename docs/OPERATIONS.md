# ComputeCop Operations

## Architecture

ComputeCop runs as a local-only FastAPI proxy in front of local inference
engines. The runtime graph has seven cooperating parts:

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
6. `UpstreamRouter` forwards accepted traffic to Ollama, llama.cpp, or
   OpenAI-compatible local endpoints.
7. `Dashboard` renders live host and policy state in a Rich terminal interface.

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

## RAM Yield

The default RAM yield threshold is 85%. When crossed, ComputeCop:

- Continues to admit foreground prompts.
- Returns retryable yield responses for background requests.
- Calls best-effort model offload hooks for supported local engines.
- Recovers only after RAM usage drops to the lower hysteresis threshold.

The default recovery threshold is 78%, preventing rapid state flapping.

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
