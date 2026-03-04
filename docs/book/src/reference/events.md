# Events

`AgentEvent` is the canonical run event stream.

## Lifecycle

- `RunStart`
- `StepStart`
- `InferenceComplete`
- `ToolCallStart` / `ToolCallDelta` / `ToolCallReady` / `ToolCallDone`
- `StepEnd`
- `RunFinish`

## State and UI Events

- `TextDelta`
- `ReasoningDelta` / `ReasoningEncryptedValue`
- `StateSnapshot` / `StateDelta`
- `MessagesSnapshot`
- `ActivitySnapshot` / `ActivityDelta`
- `ToolCallResumed`
- `Error`

## Tool Call Progress Payload

Tool-level progress is emitted through `ActivitySnapshot` / `ActivityDelta`
with:

- `activity_type = "tool-call-progress"` (canonical)
- `message_id = "tool_call:<tool_call_id>"` (node id)

Canonical `content` shape (`tool-call-progress.v1`):

```json
{
  "type": "tool-call-progress",
  "schema": "tool-call-progress.v1",
  "node_id": "tool_call:call_123",
  "parent_node_id": "tool_call:call_parent_1",
  "parent_call_id": "call_parent_1",
  "call_id": "call_123",
  "tool_name": "mcp.search",
  "status": "running",
  "progress": 0.42,
  "total": 100,
  "message": "fetching documents",
  "run_id": "run_abc",
  "parent_run_id": "run_root",
  "thread_id": "thread_1",
  "updated_at_ms": 1760000000000
}
```

`call_id` / `parent_call_id` are framework-maintained lineage fields.
Tools should not set or override them.

Compatibility:

- consumers should still accept legacy `activity_type = "progress"`
- consumers should ignore unknown fields for forward compatibility

## Terminal Semantics

`RunFinish.termination` indicates why the run ended and should be treated as authoritative.
