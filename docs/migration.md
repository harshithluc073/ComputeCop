# ComputeCop v0.1 to v0.2 Migration Guide

ComputeCop v0.2 introduces major changes to the scheduling, routing, and configuration architecture to support multi-engine deployments, capacity reservation, and robust queue management.

This guide outlines the key behavioral, API, and configuration differences between the v0.1 and v0.2 releases.

---

## 1. Scheduling and Capacity Control

In v0.1, ComputeCop utilized a simple priority queue where all requests competed directly for resources. In v0.2, this is replaced by a formal **Adaptive Scheduler** model with active capacity management.

### Behavioral Changes
- **Foreground Capacity Reservation**: Concurrency slots are explicitly reserved for foreground traffic (user prompts) to prevent background automation and agent tasks from starvation or performance degradation.
- **Dynamic Capacity Scaling**: Concurrency slots for background queue workers shrink automatically under RAM, swap, thermal, or process pressure, and drop to zero when yield mode is active.
- **Queue Aging**: The scheduler boosts the priority rank of long-waiting background tasks over time to guarantee they are eventually processed and avoid indefinite starvation.

---

## 2. Multi-Endpoint Routing and Capabilities

v0.1 expected a single upstream inference engine. v0.2 supports **Multi-Endpoint Routing** with active capability discovery.

### Capabilities and Probing
ComputeCop now automatically probes upstream endpoints (Ollama, llama.cpp, OpenAI-compatible) and caches their capabilities with a configured TTL:
- Supported API families.
- Available models.
- Streaming capabilities.
- Health status and latency.

### Routing Logic
- **Compatibility-Based Selection**: Requests are routed to the best available endpoint based on the requested model and endpoint capabilities.
- **Failover Routing**: For non-streaming requests, ComputeCop automatically fails over to other compatible, healthy endpoints if the primary endpoint fails.
- **Circuit Breakers**: Each endpoint has a circuit breaker. If an endpoint fails repeatedly, its breaker opens, temporarily excluding it from routing until a cooldown period passes and it recovers.

---

## 3. Configuration System

v0.2 updates the configuration system to support a cleaner TOML configuration workflow alongside environment variables.

### Precedence Order
When loading settings, ComputeCop uses the following deterministic precedence:
1. **CLI overrides** (highest priority).
2. **Environment variables**.
3. **TOML configuration file** (located at path set by `COMPUTECOP_CONFIG` or default directories).
4. **Built-in defaults** (lowest priority).

### Command Explain
Use the new command `computecop config explain` to display active configuration parameters along with their source (e.g., environment variable, file, or default).

---

## 4. API Endpoints Reference

v0.2 introduces several new REST APIs for endpoint monitoring and scheduler control.

### New API Endpoints

#### Endpoint Registry
- **`GET /endpoints`**: List all configured endpoints, their capabilities, health status, and circuit breaker states.

#### Queue Controls
- **`GET /queue/inspect`**: Returns a list of currently queued tasks (metadata, correlation ID, estimated tokens, and age). For privacy, prompt and completion bodies are excluded.
- **`POST /queue/pause`**: Pause the background queue. Accepts no new background submissions; running tasks continue.
- **`POST /queue/resume`**: Resume accepting background submissions.
- **`POST /queue/drain`**: Pause the queue and drain existing work. Rejects new submissions while waiting for queued tasks to finish up to a deadline.

---

## 5. Command-Line Interface (CLI) Updates

The CLI has been expanded with diagnostic and control tools:

- **`computecop queue pause` / `resume`**: Control the background worker queue from the command line.
- **`computecop config explain`**: Inspect active configuration details.
- **`computecop doctor`**: Check Python version, host platform, RAM baseline, event log paths, and endpoint reachability for rapid system diagnostics.
