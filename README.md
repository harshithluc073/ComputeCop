# ComputeCop

ComputeCop is an asyncio-first Python traffic controller for local inference
endpoints. It sits in front of engines such as Ollama and llama.cpp, monitors
host pressure with `psutil`, and dynamically budgets background inference work
so foreground prompts stay responsive on constrained developer machines.

The reference target is an Intel i7 12th Gen workstation with 16GB RAM. The
agent emphasizes local-only operation, explicit request priority semantics,
RAM-pressure yielding, thermal awareness, and a Rich terminal dashboard.

## Current capabilities

- Local inference proxy scaffold for OpenAI-compatible, Ollama-compatible, and
  llama.cpp-compatible endpoints.
- Async telemetry, policy, queue, and dashboard architecture.
- Strict distinction between user prompts and automated background requests.
- Dynamic `juice_level` budgeting under RAM, CPU, swap, and thermal pressure.
- RAM-yield behavior when memory utilization exceeds 85%.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
computecop --help
```

See [COMPUTECOP_PLAN.md](COMPUTECOP_PLAN.md) for the complete implementation
plan.

