# Expose NATS Gateway

Use this when producers/consumers communicate through NATS instead of HTTP.

## Prerequisites

- NATS server is reachable.
- Gateway is started with `AGENTOS_NATS_URL`.
- Clients can provide a reply subject (NATS inbox or `replySubject`).

## Subjects

- `agentos.ag-ui.runs`
- `agentos.ai-sdk.runs`

## Steps

1. Start gateway (example).

```bash
AGENTOS_NATS_URL=nats://127.0.0.1:4222 cargo run --package tirea-agentos-server -- --config ./agentos.json
```

2. Publish AI SDK request with auto-reply inbox.

```bash
nats req agentos.ai-sdk.runs '{"agentId":"assistant","sessionId":"thread-1","input":"hello"}'
```

3. Publish AG-UI request with auto-reply inbox.

```bash
nats req agentos.ag-ui.runs '{"agentId":"assistant","request":{"threadId":"thread-2","runId":"run-1","messages":[{"role":"user","content":"hello"}],"tools":[]}}'
```

## Verify

- You receive protocol-encoded run events on reply subject.
- AG-UI stream contains `RUN_STARTED` and `RUN_FINISHED`.
- AI SDK stream ends with a finish event equivalent to HTTP `[DONE]` semantics.

## Common Errors

- Missing reply subject when using `publish` (gateway requires a reply target).
- Invalid JSON payload or missing required fields.
- Unknown `agentId` returns one error event on reply subject.

## Related Example

- No dedicated UI starter ships for NATS yet; use `crates/tirea-agentos-server/tests/nats_gateway.rs` as the end-to-end fixture

## Key Files

- `crates/tirea-agentos-server/src/protocol/ag_ui/nats.rs`
- `crates/tirea-agentos-server/src/protocol/ai_sdk_v6/nats.rs`
- `crates/tirea-agentos-server/src/transport/nats.rs`
- `crates/tirea-agentos-server/tests/nats_gateway.rs`

## Related

- [NATS Protocol](../reference/protocols/nats.md)
- [Expose HTTP SSE](./expose-http-sse.md)
