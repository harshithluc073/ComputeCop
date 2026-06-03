# Security and Privacy

ComputeCop is designed for local developer workstations and local inference
engines. Its default security posture is intentionally conservative.

## Local-Only Defaults

The proxy binds to `127.0.0.1` by default. ComputeCop refuses remote bind
addresses unless `COMPUTECOP_EXPOSE_REMOTE=true` is explicitly set. This keeps
the proxy off the LAN unless the operator deliberately opts in.

## Request Content

ComputeCop forwards request bodies to configured local upstream endpoints. It
does not persist prompts, completions, request bodies, or response bodies in its
event log. Runtime events store only operational metadata such as request class,
decision, endpoint, model name, queue state, and correlation ID.

## Header Handling

The proxy forwards ordinary client headers except hop-by-hop transport headers
such as `host`, `connection`, `content-length`, and compression-related headers.
Structured logging redacts sensitive headers including:

- `authorization`
- `cookie`
- `set-cookie`
- `x-api-key`
- `api-key`

## Event Log

The bounded JSONL event log is stored under the user cache directory by default.
Set `COMPUTECOP_EVENT_LOG` to choose a different path. The log is capped to the
most recent events and is intended for diagnostics, not audit retention.

## Upstream Trust

Only configure upstream endpoints you control. ComputeCop assumes configured
Ollama, llama.cpp, or OpenAI-compatible endpoints are trusted local services.

## Safe Deployment Checklist

Before exposing ComputeCop beyond localhost:

1. Put it behind a trusted reverse proxy.
2. Require authentication at the network boundary.
3. Restrict upstream endpoints to local inference engines.
4. Confirm event log storage is appropriate for the machine.
5. Verify `COMPUTECOP_ENDPOINTS` contains only expected local targets.
