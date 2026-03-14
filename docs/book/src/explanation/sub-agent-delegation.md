# Sub-Agent Delegation

Sub-agent delegation is a built-in orchestration layer where one run can start/cancel/resume other agent runs.

## Runtime Model

Delegation is implemented through three agent-specific tools:

- `agent_run`: start or resume a child run
- `agent_stop`: cancel a running child run (descendants are cancelled automatically)
- `agent_output`: read child run output

System behaviors (`agent_tools`, `agent_recovery`) are wired during resolve and inject usage guidance/reminders.

## Ownership and Threads

- Parent run keeps ownership in its caller thread.
- Each child run executes on its own child thread (`sub-agent-<run_id>` pattern).
- Child run records carry lineage (`parent_run_id`, `parent_thread_id`).

This keeps parent and child state/history isolated while preserving ancestry.

## State and Handle Layers

Delegation state is tracked in two layers:

1. In-memory handle table (`SubAgentHandleTable`)
   - live `SubAgentHandle` per `run_id`, keyed by `run_id`
   - owner thread check (`owner_thread_id`)
   - epoch-based stale completion guard
   - cancellation token per handle

2. Persisted state in the owner thread:
   - **`SubAgentState`** at path `sub_agents` (scope: `Thread`)
   - `runs: HashMap<String, SubAgent>` — lightweight metadata per `run_id`
   - each `SubAgent` carries: agent id, execution ref (local `thread_id` or remote A2A ref), status, optional error

The in-memory table (`SubAgentHandleTable`) drives active control flow; `SubAgentState` persists metadata for recovery, output access, and cross-run lineage.

## Foreground vs Background

`agent_run(background=false)`:

- parent waits for child completion
- child progress can be forwarded to parent tool-call progress

`agent_run(background=true)`:

- child continues asynchronously
- parent gets immediate summary and may later call `agent_run` (resume/check), `agent_stop`, or `agent_output`

## Policy and Visibility

Target-agent visibility is filtered by scope policy:

- `RunPolicy.allowed_agents`
- `RunPolicy.excluded_agents`

`AgentDefinition::allowed_agents/excluded_agents` are projected into `RunPolicy` fields when absent (via `set_allowed_agents_if_absent`).

## Recovery Behavior

When stale running state is detected (for example after interruption), recovery behavior can transition records and enforce explicit resume/stop decisions before replay.

## Design Tradeoff

Delegation favors explicit tool-mediated orchestration over implicit nested runtime calls, so control flow remains observable, stoppable, and policy-filterable at each boundary.
