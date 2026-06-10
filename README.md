# ComputeCop

[![Version](https://img.shields.io/badge/version-0.2.1-blue.svg)](https://github.com/harshithluc073/ComputeCop)
[![Python](https://img.shields.io/badge/python-3.11%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-informational.svg)](#platform-support)

ComputeCop is a local inference traffic controller for developer workstations.
It sits in front of local LLM endpoints such as Ollama, llama.cpp, and
OpenAI-compatible servers, watches live system pressure, and dynamically budgets
background inference work so interactive prompts stay responsive.

Use it when local agents, editors, scripts, and chat sessions are all competing
for the same CPU and memory. ComputeCop classifies foreground prompts separately
from background API requests, adjusts each request's compute budget, queues or
yields lower-priority traffic when the machine is under pressure, and exposes a
terminal dashboard for live visibility.

## Why ComputeCop?

- **Protect foreground prompts**: interactive user prompts keep priority over
  automated background inference calls.
- **Budget compute dynamically**: `juice_level`, context tokens, output tokens,
  and concurrency respond to live RAM, CPU, swap, thermal, and process pressure.
- **Prevent memory spirals**: dynamic RAM thresholds adapt to the host machine
  instead of assuming a fixed hardware profile.
- **Run locally**: the proxy binds to localhost by default and forwards to local
  inference engines you control.
- **Stay observable**: a Rich-powered terminal dashboard shows pressure,
  queueing, yield state, recent admission decisions, and the policy rules that
  explain those decisions.

## Architecture

```text
Client / Agent / IDE
        |
        v
ComputeCop Proxy
        |
        +--> Request Classifier
        |       foreground prompt or background request
        |
        +--> Telemetry Loop
        |       CPU, RAM, swap, disk, thermal, heavy processes
        |
        +--> Juice Policy Engine
        |       dynamic budget, yield state, queue guidance
        |
        +--> Admission Controller
        |       allow, throttle, queue, reject, yield
        |
        +--> Adaptive Scheduler
        |       foreground reservation, spare-capacity background execution
        |
        +--> Upstream Router
                Ollama, llama.cpp, OpenAI-compatible endpoint
```

## Platform Support

ComputeCop targets:

- Windows 10/11
- macOS
- Python 3.11 or newer
- 6GB RAM minimum recommended baseline

Telemetry is powered by `psutil`. Temperature sensors are optional because
sensor availability varies by operating system and hardware. When temperature
data is unavailable, ComputeCop falls back to CPU pressure heuristics and keeps
running.

## Installation

### Windows

```powershell
git clone https://github.com/harshithluc073/ComputeCop.git
cd ComputeCop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
computecop --help
```

### macOS

```bash
git clone https://github.com/harshithluc073/ComputeCop.git
cd ComputeCop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
computecop --help
```

## Quick Start

1. Start a local inference engine.

   Ollama example:

   ```bash
   ollama serve
   ```

   llama.cpp example:

   ```bash
   ./llama-server -m ./models/model.gguf --host 127.0.0.1 --port 8080
   ```

2. Probe configured endpoints.

   ```bash
   computecop probe
   ```

3. Run the proxy.

   ```bash
   computecop run
   ```

4. Send requests through ComputeCop.

   Foreground prompt:

   ```bash
   curl http://127.0.0.1:8765/v1/chat/completions \
     -H "content-type: application/json" \
     -H "x-computecop-class: prompt" \
     -d '{"model":"local","messages":[{"role":"user","content":"Explain this code."}]}'
   ```

   Background request:

   ```bash
   curl http://127.0.0.1:8765/api/chat \
     -H "content-type: application/json" \
     -H "x-computecop-background: true" \
     -d '{"model":"llama3.1","messages":[{"role":"user","content":"Summarize logs."}]}'
   ```

## Dashboard

Run the live terminal dashboard:

```bash
computecop dashboard
```

The dashboard displays:

- CPU and RAM pressure
- swap and disk activity
- thermal state when available
- global `juice_level`
- yield state
- queue counters and lifecycle state
- scheduler foreground reservation and effective background capacity
- per-worker queue status and active correlation IDs
- recent admission decisions
- policy explanation traces
- heavy local developer processes
- a warning panel when event persistence is disabled

## Decision Explainability

Every proxied request receives a correlation ID and a policy trace ID in
response headers:

```text
x-computecop-correlation-id: ...
x-computecop-trace-id: ...
```

Inspect a recent decision locally:

```bash
curl http://127.0.0.1:8765/decisions/<correlation-id>
```

The response includes the request class, final decision, shaped budget, dynamic
RAM thresholds, pressure rules, and penalties used by the policy engine.

## Event Inspection

ComputeCop records prompt-free runtime events — admission decisions, policy
yields, and upstream failures — to a bounded JSONL log. Persistence is best
effort: if the event path becomes unwritable, ComputeCop disables persistence
and surfaces a dashboard warning instead of crashing.

Inspect recent events without opening the JSONL file by hand:

```bash
# Show the most recent events (default 20).
computecop events tail
computecop events tail --limit 50

# Find every event tied to a correlation or trace ID.
computecop events find --correlation-id <correlation-or-trace-id>

# Summarize event counts by kind and the observed time range.
computecop events stats
```

Every command accepts `--json` for scripting:

```bash
computecop events tail --json
computecop events find --correlation-id <id> --json
computecop events stats --json
```

The event log location follows `COMPUTECOP_EVENT_LOG`, falling back to the
per-user cache directory.

## Diagnostics

Run `computecop doctor` to verify the host is ready before serving traffic. It
performs read-only checks and prints a single readiness report — handy when
filing a bug report or onboarding a new machine:

```bash
computecop doctor
```

The doctor runs seven checks:

| Check | What it verifies |
| --- | --- |
| `python` | The interpreter is Python 3.11 or newer. |
| `platform` | The host OS is a supported platform. |
| `ram` | Installed RAM meets the configured minimum. |
| `psutil` | Host memory and CPU telemetry are readable. |
| `endpoints` | Configured upstream inference endpoints are reachable. |
| `event_log` | The event log path is writable. |
| `config` | The effective configuration loads and validates. |

Each check reports `ok`, `warn`, or `fail`. The command exits non-zero only on a
hard failure such as an invalid configuration or unreadable telemetry. Degraded
but usable states — an offline inference engine, a sub-minimum RAM machine —
surface as warnings with a zero exit, so `doctor` is safe to run in CI.

```bash
# Emit the full report as JSON to attach to an issue.
computecop doctor --json

# Skip network probes when the inference engine is intentionally offline.
computecop doctor --skip-endpoints
```

Pass `--config <path>` (before the subcommand) to validate a specific TOML file.

## Configuration

ComputeCop loads settings in this order:

1. Built-in defaults
2. TOML configuration file
3. Environment variables
4. CLI flags such as `--host`, `--port`, and `--log-level`

Use a TOML file for durable local settings and keep environment variables for
temporary overrides in automation.

### TOML Configuration File

Point ComputeCop at a config file with `COMPUTECOP_CONFIG` or `--config`:

```bash
export COMPUTECOP_CONFIG=examples/computecop.toml
computecop run
```

```bash
computecop --config examples/computecop.toml run
```

Example file:

```toml
[server]
host = "127.0.0.1"
port = 8765

[policy]
minimum_supported_ram_gb = 6.0
ram_yield_percent = 85.0

[[endpoints]]
name = "ollama"
kind = "ollama"
base_url = "http://127.0.0.1:11434"
health_path = "/api/tags"
```

Inspect where each effective value came from:

```bash
computecop config explain
computecop config explain --json
```

### Common Settings

| Variable | Default | Description |
| --- | --- | --- |
| `COMPUTECOP_CONFIG` | unset | Path to a TOML configuration file. |
| `COMPUTECOP_HOST` | `127.0.0.1` | Proxy bind host. |
| `COMPUTECOP_PORT` | `8765` | Proxy bind port. |
| `COMPUTECOP_EXPOSE_REMOTE` | `false` | Required for non-local bind addresses. |
| `COMPUTECOP_ENDPOINTS` | built-in Ollama and llama.cpp defaults | JSON endpoint list. |
| `COMPUTECOP_MIN_RAM_GB` | `6.0` | Minimum supported RAM baseline used by policy. |
| `COMPUTECOP_RAM_YIELD_PERCENT` | `85.0` | Upper cap for dynamic RAM yield threshold. |
| `COMPUTECOP_RAM_RECOVER_PERCENT` | `78.0` | Upper cap for dynamic RAM recovery threshold. |
| `COMPUTECOP_RAM_RECOVER_GAP_PERCENT` | `7.0` | Hysteresis gap below dynamic yield threshold. |
| `COMPUTECOP_CPU_PRESSURE_PERCENT` | `88.0` | CPU pressure threshold. |
| `COMPUTECOP_QUEUE_MAX_SIZE` | `128` | Maximum queued background requests. |
| `queue.aging_interval_seconds` | `30.0` | Queue wait interval before a background item gains priority. |
| `policy.max_foreground_concurrency` | `4` | Reserved foreground execution slots. |
| `policy.max_background_concurrency` | `2` | Maximum background execution slots. |
| `COMPUTECOP_EVENT_LOG` | user cache directory | Optional JSONL event log path. |

### Endpoint Configuration

`COMPUTECOP_ENDPOINTS` is a JSON list:

```json
[
  {
    "name": "ollama",
    "kind": "ollama",
    "base_url": "http://127.0.0.1:11434",
    "health_path": "/api/tags",
    "timeout_seconds": 180,
    "supports_streaming": true
  },
  {
    "name": "llama-cpp",
    "kind": "llama_cpp",
    "base_url": "http://127.0.0.1:8080",
    "health_path": "/health",
    "timeout_seconds": 180,
    "supports_streaming": true
  }
]
```

Route a request to a specific endpoint with:

```text
x-computecop-endpoint: llama-cpp
```

Inspect configured endpoint capabilities, cached health, and routing metadata:

```bash
curl http://127.0.0.1:8765/endpoints
curl "http://127.0.0.1:8765/endpoints?refresh=true"
```

Each record reports API family, streaming/model-list/offload support, default
context hints, probe latency, failure rate, and whether the endpoint is the
default route for its family.

## Request Priority

ComputeCop distinguishes direct prompts from automated requests.

Foreground prompt headers:

```text
x-computecop-class: prompt
x-computecop-priority: foreground
```

Background request headers:

```text
x-computecop-background: true
x-agent-request: true
x-automation-request: true
```

Foreground prompts are admitted preferentially. Background requests may be
throttled, queued, or asked to retry when the host enters yield mode.

### Request Classification Guidance

ComputeCop automatically runs heuristics on incoming request payloads and user agents to infer their category. However, explicit headers guarantee correct scheduling and maximum throughput:

- **Explicit headers** (like `x-computecop-class` or `x-computecop-priority`) yield **high** classification confidence.
- **Payload/User-Agent heuristics** yield **medium** classification confidence.
- **Fallbacks** yield **low** classification confidence.

For low-confidence classifications, ComputeCop returns advisory headers in the HTTP response to help you align your integration:

- `x-computecop-classification-confidence`: `"high"`, `"medium"`, or `"low"`.
- `x-computecop-classification-hint`: returned only when confidence is `low` (e.g. `"add x-computecop-background: true for automated work"`).

## Development

Run the verification suite:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src/computecop
python -m pytest
python -m build
```

Windows helper:

```powershell
.\scripts\verify.ps1
```

Maintainers cutting a release should follow the
[release checklist](docs/RELEASE.md), which documents the version bump process,
verification gates, supported platforms, and the local-only artifact policy.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a focused feature branch.
3. Add tests for behavior changes.
4. Run formatting, linting, typing, tests, and build checks.
5. Open a pull request with a clear description and verification notes.

Good first areas include endpoint adapters, platform telemetry improvements,
dashboard refinements, and additional local inference engine examples.

## Security

ComputeCop is local-first and binds to `127.0.0.1` by default. Do not expose it
on a network interface unless you intentionally place it behind appropriate
authentication and network controls. See [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
