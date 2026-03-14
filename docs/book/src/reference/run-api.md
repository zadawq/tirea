# Run API

The Run API is the canonical HTTP surface for run/task management.

Unlike protocol adapters (`ag-ui`, `ai-sdk`, `a2a`), this API exposes a stable transport-neutral run model.

## Prerequisites

- Server must include `http::run_routes()`.

## Endpoints

- `GET /v1/runs`
- `GET /v1/runs/:id`
- `POST /v1/runs`
- `POST /v1/runs/:id/inputs`
- `POST /v1/runs/:id/cancel`

## Run Record Model

`GET /v1/runs/:id` returns `RunRecord`:

```json
{
  "run_id": "run-1",
  "thread_id": "thread-1",
  "agent_id": "assistant",
  "parent_run_id": "run-0",
  "parent_thread_id": "thread-0",
  "origin": "ag_ui",
  "status": "done",
  "termination_code": "input_required",
  "termination_detail": "approval needed",
  "created_at": 1760000000000,
  "updated_at": 1760000005000,
  "metadata": null
}
```

`origin`:

- `user`
- `subagent`
- `ag_ui`
- `ai_sdk`
- `a2a`
- `internal`

`status`:

- `running`
- `waiting`
- `done`

## List Runs

`GET /v1/runs` query params:

- `offset` (default `0`)
- `limit` (clamped `1..=200`, default `50`)
- `thread_id`
- `parent_run_id`
- `status`
- `origin`
- `termination_code`
- `created_at_from`, `created_at_to` (unix millis, inclusive)
- `updated_at_from`, `updated_at_to` (unix millis, inclusive)

Example:

```bash
curl 'http://127.0.0.1:8080/v1/runs?thread_id=t1&status=done&origin=ag_ui&limit=20'
```

## Start Run

`POST /v1/runs` starts a run and streams canonical SSE events.

Minimal payload:

```json
{
  "agentId": "assistant",
  "messages": [{ "role": "user", "content": "hello" }]
}
```

All accepted fields:

- `agentId` / `agent_id` (required)
- `threadId` / `thread_id` / `contextId` / `context_id` (optional — generated if absent)
- `runId` / `run_id` / `taskId` / `task_id` (optional — generated if absent)
- `parentRunId` / `parent_run_id` (optional — links this run to a parent run for lineage tracking)
- `parentThreadId` / `parent_thread_id` / `parentContextId` / `parent_context_id` (optional — links to a parent thread)
- `resourceId` / `resource_id` (optional — associates the run with a resource identifier)
- `state` (optional JSON object — initial state to merge into the thread before the run starts)
- `messages` (optional array — user messages to start the run with)
- `initialDecisions` / `initial_decisions` / `decisions` (optional array — pre-resolved tool call decisions)

Example:

```bash
curl -N \
  -H 'content-type: application/json' \
  -d '{"agentId":"assistant","threadId":"thread-1","messages":[{"role":"user","content":"hello"}]}' \
  http://127.0.0.1:8080/v1/runs
```

## Push Inputs

`POST /v1/runs/:id/inputs` has two modes.

1. Decision forwarding (active run):

```json
{
  "decisions": [
    {
      "target_id": "fc_perm_1",
      "decision_id": "d1",
      "action": "resume",
      "result": { "approved": true },
      "updated_at": 1760000000000
    }
  ]
}
```

Response (`202`):

```json
{
  "status": "decision_forwarded",
  "run_id": "run-1",
  "thread_id": "thread-1"
}
```

2. Continuation run (messages provided):

```json
{
  "agentId": "assistant",
  "messages": [{ "role": "user", "content": "continue" }],
  "decisions": []
}
```

Response (`202`):

```json
{
  "status": "continuation_started",
  "parent_run_id": "run-1",
  "thread_id": "thread-1"
}
```

Rules:

- `messages` and `decisions` cannot both be empty.
- If `messages` is present, `agentId` is required.
- Decision-only forwarding requires the target run to be active.

## Cancel Run

`POST /v1/runs/:id/cancel`

Response (`202`):

```json
{
  "status": "cancel_requested",
  "run_id": "run-1"
}
```

If run exists but is not active: `400` (`run is not active`).

## Errors

Common API errors:

- `404`: `run not found: <id>`
- `404`: `agent not found: <id>`
- `400`: input validation failures
- `500`: internal errors (including uninitialized run service)

Error body shape:

```json
{ "error": "bad request: messages and decisions cannot both be empty" }
```
