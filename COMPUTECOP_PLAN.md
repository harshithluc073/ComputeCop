# ComputeCop Development Plan

ComputeCop is a background Python agent that acts as a dynamic traffic controller for local inference endpoints on memory-constrained developer machines, with Intel i7 12th Gen and 16GB RAM as the reference target. The implementation will be asyncio-first, production-oriented, testable, and organized around a proxy, telemetry loop, policy engine, queue, and terminal dashboard.

## Atomic Execution Steps

1. Initialize the Git repository, configure the `origin` remote as `https://github.com/harshithluc073/ComputeCop.git`, and create the Python package scaffold with project metadata, license, ignore rules, package exports, and this plan committed as the first repository artifact.
2. Add typed core domain models for request priority, request class, telemetry samples, juice budgets, system states, admission decisions, and proxy routing metadata.
3. Implement configuration loading with defaults, environment-variable overrides, validation, endpoint definitions, thermal policy thresholds, and a serializable runtime config object.
4. Implement structured logging utilities with Rich-aware console output, JSON-safe event formatting, log-level configuration, and reusable logger factories.
5. Implement the telemetry sampling service using `psutil` for CPU, RAM, swap, process, disk, battery, and network-adjacent load signals.
6. Add CPU temperature and thermal-state detection with cross-platform fallbacks, Intel-friendly heuristics, and graceful degradation when sensors are unavailable.
7. Implement developer-process detection that identifies heavy local tools such as IDEs, browsers, compilers, Python workers, Ollama, llama.cpp, Docker, and Node build processes.
8. Build the asynchronous telemetry loop that periodically samples system metrics, smooths spikes, emits snapshots, and supports cooperative shutdown.
9. Implement the global state store with lock-safe telemetry snapshots, policy decisions, yield state, queue counters, and dashboard-facing read models.
10. Implement the juice-level policy engine that computes global and per-request compute budgets from RAM pressure, CPU pressure, thermal state, swap pressure, and foreground activity.
11. Implement strict request classification that distinguishes high-priority user prompts from automated background API requests using headers, metadata, payload fields, and source hints.
12. Implement the admission controller that allows prompts, throttles background requests, pauses low-priority requests during yield, and records explicit decision reasons.
13. Implement a durable async request queue with priority ordering, deadline handling, cancellation support, backpressure limits, and memory-aware admission.
14. Implement RAM pressure yielding, including the 85% RAM utilization threshold, automatic recovery hysteresis, queue suspension, and offload hook signaling.
15. Implement model-offload adapters for Ollama and llama.cpp compatible endpoints, using best-effort unload/keepalive semantics without relying on unavailable engine internals.
16. Implement HTTP client routing with `httpx`, streaming response support, timeout handling, retry-safe error mapping, and upstream endpoint health probing.
17. Implement the FastAPI proxy server skeleton with lifecycle management, dependency injection, health endpoints, state endpoints, and telemetry endpoints.
18. Implement OpenAI-compatible `/v1/chat/completions` interception with request classification, juice-budget mutation, queue admission, upstream forwarding, and streaming passthrough.
19. Implement Ollama-compatible `/api/generate`, `/api/chat`, and `/v1/*` passthrough handling with priority-aware routing and request-budget shaping.
20. Implement llama.cpp-compatible completion and chat completion passthrough with request-budget shaping and endpoint-specific parameter translation.
21. Implement response normalization and error contracts so proxy callers receive clear HTTP status codes, decision metadata, retry guidance, and correlation IDs.
22. Implement the Rich terminal dashboard with live CPU, RAM, thermal, queue, juice-level, endpoint, and decision panels.
23. Add a CLI using Typer with commands for running the proxy, running the dashboard, printing config, probing endpoints, and dumping one-shot telemetry.
24. Add graceful shutdown handling for the CLI, proxy, telemetry loop, queue workers, and dashboard refresh loop.
25. Add persistence for lightweight runtime events, policy changes, and recent decisions using a bounded JSONL store under the user cache directory.
26. Add comprehensive unit tests for configuration, request classification, juice policy, admission control, state store, and queue behavior.
27. Add telemetry tests with mocked `psutil` outputs for RAM pressure, CPU pressure, thermal fallbacks, process detection, and hysteresis behavior.
28. Add proxy tests using `httpx` ASGI transport for health routes, state routes, OpenAI-compatible requests, Ollama-compatible requests, queue decisions, and error contracts.
29. Add CLI smoke tests for config rendering, telemetry rendering, endpoint probing failure modes, and command import stability.
30. Add developer tooling with Ruff, MyPy-oriented typing configuration, pytest configuration, coverage settings, and repeatable local verification commands.
31. Add packaging metadata, console scripts, optional dependency groups, pinned minimum supported versions, and an installable wheel build path.
32. Add operational documentation covering architecture, request-priority semantics, juice-level behavior, endpoint configuration, RAM-yield behavior, and dashboard usage.
33. Add security and privacy documentation explaining local-only operation, header handling, logging redaction, and safe defaults for proxy exposure.
34. Add example configuration files for Ollama and llama.cpp, plus example client requests demonstrating prompt versus background request classification.
35. Run formatting, linting, type-oriented checks where available, and the full test suite; fix all discovered issues with focused commits.
36. Perform final repository verification, confirm the Git remote and branch state, push the final commit, and record the completion status.
