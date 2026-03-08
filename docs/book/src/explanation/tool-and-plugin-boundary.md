# Tool and Plugin Boundary

Tools and plugins solve different problems. Keep the boundary strict.

## Tool Responsibility

Tools implement domain actions:

- Read/write state through `ToolCallContext`
- Call external systems
- Return structured `ToolResult`
- Optionally emit `Action`s through `ToolExecutionEffect`

Tools should not orchestrate loop policy globally.

## Action Reducer Model

The runtime is not "tool returns JSON and stops there". Tools and plugins both participate in an action/reducer pipeline.

- A plain tool may return only `ToolResult`
- A richer tool may return `ToolExecutionEffect`
- `ToolExecutionEffect` contains:
  - a `ToolResult`
  - zero or more `Action`s applied during `AfterToolExecute`
- State changes are represented as `AnyStateAction` and reduced by the runtime into the execution patch
- Non-state actions can mutate step-local runtime structures such as queued user messages

Conceptually:

```text
Tool / Plugin -> Action(s) -> Phase validation -> StepContext apply -> Reducer / patch commit
```

This is why "tool execution" in Tirea can do more than return a payload. It can also update persisted state, inject messages, or alter later runtime behavior through actions.

## What Tools Can Change

From a tool, you can:

- write typed state through `ToolCallContext`
- return a `ToolResult`
- emit explicit `Action`s from `execute_effect`

Typical tool-emitted effects include:

- `AnyStateAction` to update reducer-backed state
- user-message insertion actions
- other custom `Action` implementations valid in `AfterToolExecute`

## What Plugins Can Change

Plugins implement cross-cutting policy:

- Inject context (`StepStart`, `BeforeInference`)
- Filter/allow/deny tools (`BeforeInference`, `BeforeToolExecute`)
- Add reminders or execution metadata (`AfterToolExecute`)

Plugins operate at phase boundaries and are the right place for rules that must apply uniformly across many tools.

## Concrete Examples

### Skill activation

The skill activation tool is a good example of a tool using `ToolExecutionEffect` instead of returning only `ToolResult`.

It does three things in one execution:

1. Returns a success `ToolResult`
2. Emits a state action to activate the skill in persisted state
3. Emits additional actions to:
   - append the skill instructions into user-visible message flow
   - widen allowed tool permissions for the activated skill

This is implemented in [`crates/tirea-extension-skills/src/tools.rs`](/home/chaizhenhua/Codes/uncarve/crates/tirea-extension-skills/src/tools.rs).

### Permission policy

Permission handling is a plugin concern because it is global execution policy, not domain work.

The permission plugin:

- checks state snapshot before tool execution
- blocks, allows, or suspends the call
- can emit `BeforeInferenceAction` to include/exclude tools
- can emit `BeforeToolExecuteAction` to deny or suspend execution

This is implemented in [`crates/tirea-extension-permission/src/plugin.rs`](/home/chaizhenhua/Codes/uncarve/crates/tirea-extension-permission/src/plugin.rs).

## Plugin Responsibility

Plugins should not own domain-side business operations.

## Rule of Thumb

- If it is business capability, build a tool.
- If it is execution policy or guardrail, build a plugin.
- If it is a domain tool that needs to return both a result and side effects, use `execute_effect`.
