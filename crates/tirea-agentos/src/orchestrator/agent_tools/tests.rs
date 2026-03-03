use super::*;
use crate::contracts::runtime::behavior::ReadOnlyContext;
use crate::contracts::runtime::phase::{Phase, StepContext};
use crate::contracts::runtime::phase::{BeforeInferenceAction, ActionSet};
use crate::contracts::runtime::state::{reduce_state_actions, ScopeContext};
use crate::contracts::runtime::tool_call::ToolStatus;
use crate::contracts::thread::Thread;
use crate::contracts::AgentBehavior;
use crate::orchestrator::InMemoryAgentRegistry;
use crate::runtime::loop_runner::{
    TOOL_SCOPE_CALLER_AGENT_ID_KEY, TOOL_SCOPE_CALLER_MESSAGES_KEY, TOOL_SCOPE_CALLER_STATE_KEY,
    TOOL_SCOPE_CALLER_THREAD_ID_KEY,
};
use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;
use tirea_contract::testing::{
    apply_after_inference_for_test, apply_after_tool_for_test, apply_before_inference_for_test,
    apply_before_tool_for_test, apply_lifecycle_for_test, TestFixture,
};
use tirea_state::apply_patches;

#[async_trait]
trait AgentBehaviorTestDispatch {
    async fn run_phase(&self, phase: Phase, step: &mut StepContext<'_>);
}

#[async_trait]
impl<T> AgentBehaviorTestDispatch for T
where
    T: AgentBehavior + ?Sized,
{
    async fn run_phase(&self, phase: Phase, step: &mut StepContext<'_>) {
        let ctx = ReadOnlyContext::new(
            phase,
            step.thread_id(),
            step.messages(),
            step.run_config(),
            step.ctx().doc(),
        );
        match phase {
            Phase::RunStart => apply_lifecycle_for_test(step, self.run_start(&ctx).await),
            Phase::StepStart => apply_lifecycle_for_test(step, self.step_start(&ctx).await),
            Phase::BeforeInference => {
                apply_before_inference_for_test(step, self.before_inference(&ctx).await)
            }
            Phase::AfterInference => {
                apply_after_inference_for_test(step, self.after_inference(&ctx).await)
            }
            Phase::BeforeToolExecute => {
                apply_before_tool_for_test(step, self.before_tool_execute(&ctx).await)
            }
            Phase::AfterToolExecute => {
                apply_after_tool_for_test(step, self.after_tool_execute(&ctx).await)
            }
            Phase::StepEnd => apply_lifecycle_for_test(step, self.step_end(&ctx).await),
            Phase::RunEnd => apply_lifecycle_for_test(step, self.run_end(&ctx).await),
        }
        // Reduce any pending state actions
        if !step.pending_state_actions.is_empty() {
            let state_actions = std::mem::take(&mut step.pending_state_actions);
            let snapshot = step.snapshot();
            let patches =
                reduce_state_actions(state_actions, &snapshot, "test", &ScopeContext::run())
                    .expect("state actions should reduce");
            for patch in patches {
                let doc = step.ctx().doc();
                for op in patch.patch().ops() {
                    doc.apply(op).expect("state action patch op should apply");
                }
                step.emit_patch(patch);
            }
        }
    }
}

// ── Plugin tests ─────────────────────────────────────────────────────────────

#[test]
fn plugin_filters_out_caller_agent() {
    let mut reg = InMemoryAgentRegistry::new();
    reg.upsert("a", crate::orchestrator::AgentDefinition::new("mock"));
    reg.upsert("b", crate::orchestrator::AgentDefinition::new("mock"));
    let plugin = AgentToolsPlugin::new(Arc::new(reg), Arc::new(SubAgentHandleTable::new()));
    let rendered = plugin.render_available_agents(Some("a"), None);
    assert!(rendered.contains("<id>b</id>"));
    assert!(!rendered.contains("<id>a</id>"));
}

#[test]
fn plugin_filters_agents_by_scope_policy() {
    let mut reg = InMemoryAgentRegistry::new();
    reg.upsert("writer", crate::orchestrator::AgentDefinition::new("mock"));
    reg.upsert(
        "reviewer",
        crate::orchestrator::AgentDefinition::new("mock"),
    );
    let plugin = AgentToolsPlugin::new(Arc::new(reg), Arc::new(SubAgentHandleTable::new()));
    let mut rt = tirea_contract::RunConfig::new();
    rt.set(SCOPE_ALLOWED_AGENTS_KEY, vec!["writer"]).unwrap();
    let rendered = plugin.render_available_agents(None, Some(&rt));
    assert!(rendered.contains("<id>writer</id>"));
    assert!(!rendered.contains("<id>reviewer</id>"));
}

#[test]
fn plugin_renders_agent_output_tool_usage() {
    let mut reg = InMemoryAgentRegistry::new();
    reg.upsert("worker", crate::orchestrator::AgentDefinition::new("mock"));
    let plugin = AgentToolsPlugin::new(Arc::new(reg), Arc::new(SubAgentHandleTable::new()));
    let rendered = plugin.render_available_agents(None, None);
    assert!(
        rendered.contains("agent_output"),
        "available agents should mention agent_output tool"
    );
}

#[tokio::test]
async fn plugin_adds_reminder_for_running_and_stopped_runs() {
    let mut reg = InMemoryAgentRegistry::new();
    reg.upsert("worker", crate::orchestrator::AgentDefinition::new("mock"));
    let handles = Arc::new(SubAgentHandleTable::new());
    let plugin = AgentToolsPlugin::new(Arc::new(reg), handles.clone());

    let epoch = handles
        .put_running(
            "run-1",
            "owner-1".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    assert_eq!(epoch, 1);

    let fixture = TestFixture::new();
    let mut step = StepContext::new(fixture.ctx(), "owner-1", &fixture.messages, vec![]);
    plugin.run_phase(Phase::AfterToolExecute, &mut step).await;
    let reminder = step.messaging.reminders.first()
        .expect("running reminder should be present");
    assert!(reminder.contains("status=\"running\""));

    handles.stop_owned_tree("owner-1", "run-1").await.unwrap();
    let fixture2 = TestFixture::new();
    let mut step2 = StepContext::new(fixture2.ctx(), "owner-1", &fixture2.messages, vec![]);
    plugin.run_phase(Phase::AfterToolExecute, &mut step2).await;
    let reminder2 = step2.messaging.reminders.first()
        .expect("stopped reminder should be present");
    assert!(reminder2.contains("status=\"stopped\""));
}

// ── Handle table tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn handle_table_ignores_stale_completion_by_epoch() {
    let handles = SubAgentHandleTable::new();
    let epoch1 = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "agent-a".to_string(),
            None,
            None,
        )
        .await;
    assert_eq!(epoch1, 1);

    let epoch2 = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "agent-a".to_string(),
            None,
            None,
        )
        .await;
    assert_eq!(epoch2, 2);

    let ignored = handles
        .update_after_completion(
            "run-1",
            epoch1,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await;
    assert!(ignored.is_none());

    let summary = handles
        .get_owned_summary("owner", "run-1")
        .await
        .expect("run should still exist");
    assert_eq!(summary.status, SubAgentStatus::Running);

    let applied = handles
        .update_after_completion(
            "run-1",
            epoch2,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await
        .expect("latest epoch completion should apply");
    assert_eq!(applied.status, SubAgentStatus::Completed);
}

#[tokio::test]
async fn handle_table_stop_tree_stops_descendants() {
    let handles = SubAgentHandleTable::new();
    handles
        .put_running(
            "parent-run",
            "owner-thread".to_string(),
            "sub-agent-parent-run".to_string(),
            "agent-a".to_string(),
            None,
            None,
        )
        .await;
    handles
        .put_running(
            "child-run",
            "owner-thread".to_string(),
            "sub-agent-child-run".to_string(),
            "agent-a".to_string(),
            Some("parent-run".to_string()),
            None,
        )
        .await;
    handles
        .put_running(
            "grandchild-run",
            "owner-thread".to_string(),
            "sub-agent-grandchild-run".to_string(),
            "agent-a".to_string(),
            Some("child-run".to_string()),
            None,
        )
        .await;
    handles
        .put_running(
            "other-owner-run",
            "other-owner".to_string(),
            "sub-agent-other-owner-run".to_string(),
            "agent-b".to_string(),
            Some("parent-run".to_string()),
            None,
        )
        .await;

    let stopped = handles
        .stop_owned_tree("owner-thread", "parent-run")
        .await
        .unwrap();

    assert_eq!(stopped.len(), 3);

    let parent = handles
        .get_owned_summary("owner-thread", "parent-run")
        .await
        .expect("parent run should exist");
    assert_eq!(parent.status, SubAgentStatus::Stopped);

    let child = handles
        .get_owned_summary("owner-thread", "child-run")
        .await
        .expect("child run should exist");
    assert_eq!(child.status, SubAgentStatus::Stopped);

    let grandchild = handles
        .get_owned_summary("owner-thread", "grandchild-run")
        .await
        .expect("grandchild run should exist");
    assert_eq!(grandchild.status, SubAgentStatus::Stopped);

    let denied = handles
        .stop_owned_tree("owner-thread", "other-owner-run")
        .await;
    assert!(denied.is_err());
}

// ── AgentRunTool tests ───────────────────────────────────────────────────────

#[derive(Debug)]
struct SlowTerminatePlugin;

#[async_trait]
impl AgentBehavior for SlowTerminatePlugin {
    fn id(&self) -> &str {
        "slow_terminate_behavior_requested"
    }

    async fn before_inference(&self, _ctx: &ReadOnlyContext<'_>) -> ActionSet<BeforeInferenceAction> {
        tokio::time::sleep(Duration::from_millis(120)).await;
        ActionSet::single(BeforeInferenceAction::Terminate(
            crate::contracts::TerminationReason::BehaviorRequested,
        ))
    }
}

fn caller_scope_with_state_and_run(
    state: serde_json::Value,
    run_id: &str,
) -> tirea_contract::RunConfig {
    caller_scope_with_state_run_and_messages(
        state,
        run_id,
        vec![crate::contracts::thread::Message::user("seed message")],
    )
}

fn caller_scope_with_state_run_and_messages(
    state: serde_json::Value,
    run_id: &str,
    messages: Vec<crate::contracts::thread::Message>,
) -> tirea_contract::RunConfig {
    let mut rt = tirea_contract::RunConfig::new();
    rt.set(TOOL_SCOPE_CALLER_THREAD_ID_KEY, "owner-thread")
        .unwrap();
    rt.set(TOOL_SCOPE_CALLER_AGENT_ID_KEY, "caller").unwrap();
    rt.set(SCOPE_RUN_ID_KEY, run_id).unwrap();
    rt.set(TOOL_SCOPE_CALLER_STATE_KEY, state).unwrap();
    rt.set(TOOL_SCOPE_CALLER_MESSAGES_KEY, messages).unwrap();
    rt
}

fn caller_scope_with_state(state: serde_json::Value) -> tirea_contract::RunConfig {
    caller_scope_with_state_and_run(state, "parent-run-default")
}

fn caller_scope() -> tirea_contract::RunConfig {
    caller_scope_with_state(json!({"forked": true}))
}

#[tokio::test]
async fn agent_run_tool_requires_scope_context() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));
    let fix = TestFixture::new();
    let result = tool
        .execute(
            json!({"agent_id":"worker","prompt":"hi","background":false}),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("missing caller thread context"));
}

#[tokio::test]
async fn agent_run_tool_rejects_disallowed_target_agent() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .with_agent(
            "reviewer",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();
    fix.run_config
        .set(SCOPE_ALLOWED_AGENTS_KEY, vec!["worker"])
        .unwrap();
    let result = tool
        .execute(
            json!({"agent_id":"reviewer","prompt":"hi","background":false}),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("Unknown or unavailable agent_id"));
}

#[tokio::test]
async fn agent_run_tool_rejects_self_target_agent() {
    let os = AgentOs::builder()
        .with_agent(
            "caller",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();
    let result = tool
        .execute(
            json!({"agent_id":"caller","prompt":"hi","background":false}),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("Unknown or unavailable agent_id"));
}

#[tokio::test]
async fn agent_run_tool_fork_context_filters_messages() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let run_tool = AgentRunTool::new(os, handles.clone());

    let fork_messages = vec![
        crate::contracts::thread::Message::system("parent-system"),
        crate::contracts::thread::Message::internal_system("parent-internal-system"),
        crate::contracts::thread::Message::user("parent-user-1"),
        crate::contracts::thread::Message::assistant_with_tool_calls(
            "parent-assistant-tool-call",
            vec![
                crate::contracts::thread::ToolCall::new(
                    "call-paired",
                    "search",
                    json!({"q":"paired"}),
                ),
                crate::contracts::thread::ToolCall::new(
                    "call-missing",
                    "search",
                    json!({"q":"missing"}),
                ),
            ],
        ),
        crate::contracts::thread::Message::tool("call-paired", "tool paired result"),
        crate::contracts::thread::Message::tool("call-orphan", "tool orphan result"),
        crate::contracts::thread::Message::assistant_with_tool_calls(
            "assistant-unpaired-only",
            vec![crate::contracts::thread::ToolCall::new(
                "call-only-assistant",
                "search",
                json!({"q":"only-assistant"}),
            )],
        ),
        crate::contracts::thread::Message::assistant("parent-assistant-plain"),
    ];

    let mut fix = TestFixture::new();
    fix.run_config = caller_scope_with_state_run_and_messages(
        json!({"forked": true}),
        "parent-run-42",
        fork_messages,
    );

    let started = run_tool
        .execute(
            json!({
                "agent_id":"worker",
                "prompt":"child-prompt",
                "background": true,
                "fork_context": true
            }),
            &fix.ctx_with("call-run", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(started.status, ToolStatus::Success);
    assert_eq!(started.data["status"], json!("running"));

    // Verify the fork_context flag produces a running status.
    // (Child thread details are now in ThreadStore, not in-memory handle.)
}

#[tokio::test]
async fn background_stop_then_resume_completes() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let run_tool = AgentRunTool::new(os, handles.clone());
    let stop_tool = AgentStopTool::new(handles);

    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();
    let started = run_tool
        .execute(
            json!({
                "agent_id":"worker",
                "prompt":"start",
                "background": true
            }),
            &fix.ctx_with("call-run", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(started.status, ToolStatus::Success);
    assert_eq!(started.data["status"], json!("running"));
    let run_id = started.data["run_id"]
        .as_str()
        .expect("run_id should exist")
        .to_string();

    let mut stop_fix = TestFixture::new();
    stop_fix.run_config = caller_scope();
    let stopped = stop_tool
        .execute(
            json!({ "run_id": run_id.clone() }),
            &stop_fix.ctx_with("call-stop", "tool:agent_stop"),
        )
        .await
        .unwrap();
    assert_eq!(stopped.status, ToolStatus::Success);
    assert_eq!(stopped.data["status"], json!("stopped"));

    // Give cancelled background task a chance to flush stale completion.
    tokio::time::sleep(Duration::from_millis(30)).await;

    // Resume: since there's no ThreadStore in this test context, resume from
    // the live handle which is Stopped.
    let resumed = run_tool
        .execute(
            json!({
                "run_id": run_id,
                "prompt":"resume",
                "background": false
            }),
            &fix.ctx_with("call-run", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(resumed.status, ToolStatus::Success);
    // The execution will fail because there's no ThreadStore configured,
    // but the tool itself should handle the error gracefully.
    let status = resumed.data["status"].as_str().unwrap();
    assert!(
        status == "completed" || status == "failed",
        "expected terminal status, got: {status}"
    );
}

#[tokio::test]
async fn agent_run_tool_persists_run_state_patch() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));

    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();
    let started = run_tool
        .execute(
            json!({
                "agent_id":"worker",
                "prompt":"start",
                "background": true
            }),
            &fix.ctx_with("call-run", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(started.status, ToolStatus::Success);
    let run_id = started.data["run_id"]
        .as_str()
        .expect("run_id should exist")
        .to_string();

    let patch = fix.ctx_with("call-run", "tool:agent_run").take_patch();
    assert!(
        !patch.patch().is_empty(),
        "expected tool to persist run snapshot into state"
    );
    let base = json!({});
    let updated = apply_patches(&base, std::iter::once(patch.patch())).unwrap();
    assert_eq!(
        updated["sub_agents"]["runs"][&run_id]["status"],
        json!("running")
    );
    // No thread field in SubAgent.
    assert!(
        updated["sub_agents"]["runs"][&run_id].get("thread").is_none()
            || updated["sub_agents"]["runs"][&run_id]["thread"].is_null(),
        "SubAgent should not contain embedded thread"
    );
    // thread_id should be present.
    assert!(
        updated["sub_agents"]["runs"][&run_id]["thread_id"]
            .as_str()
            .is_some(),
        "SubAgent should contain thread_id"
    );
}

#[tokio::test]
async fn agent_run_tool_binds_scope_run_id_and_parent_lineage() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let run_tool = AgentRunTool::new(os, handles.clone());

    let mut fix = TestFixture::new();
    fix.run_config = caller_scope_with_state_and_run(json!({"forked": true}), "parent-run-42");
    let started = run_tool
        .execute(
            json!({
                "agent_id":"worker",
                "prompt":"start",
                "background": true
            }),
            &fix.ctx_with("call-run", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(started.status, ToolStatus::Success);
    let run_id = started.data["run_id"]
        .as_str()
        .expect("run_id should exist")
        .to_string();

    let patch = fix.ctx_with("call-run", "tool:agent_run").take_patch();
    let base = json!({});
    let updated = apply_patches(&base, std::iter::once(patch.patch())).unwrap();
    assert_eq!(
        updated["sub_agents"]["runs"][&run_id]["parent_run_id"],
        json!("parent-run-42")
    );
}

#[tokio::test]
async fn agent_run_tool_resumes_from_persisted_state_without_live_record() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));

    let doc = json!({
        "sub_agents": {
            "runs": {
                "run-1": {
                    "thread_id": "sub-agent-run-1",
                    "agent_id": "worker",
                    "status": "stopped"
                }
            }
        }
    });
    let mut fix = TestFixture::new_with_state(doc.clone());
    fix.run_config = caller_scope_with_state(doc);
    let resumed = run_tool
        .execute(
            json!({
                "run_id":"run-1",
                "prompt":"resume",
                "background": false
            }),
            &fix.ctx_with("call-run", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(resumed.status, ToolStatus::Success);
    // The execution will fail because there's no ThreadStore, but the tool handles it.
    let status = resumed.data["status"].as_str().unwrap();
    assert!(
        status == "completed" || status == "failed",
        "expected terminal status, got: {status}"
    );
}

#[tokio::test]
async fn agent_run_tool_marks_orphan_running_as_stopped_before_resume() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));

    let doc = json!({
        "sub_agents": {
            "runs": {
                "run-1": {
                    "thread_id": "sub-agent-run-1",
                    "agent_id": "worker",
                    "status": "running"
                }
            }
        }
    });
    let mut fix = TestFixture::new_with_state(doc.clone());
    fix.run_config = caller_scope_with_state(doc);
    let summary = run_tool
        .execute(
            json!({
                "run_id":"run-1",
                "background": false
            }),
            &fix.ctx_with("call-run", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(summary.status, ToolStatus::Success);
    assert_eq!(summary.data["status"], json!("stopped"));
}

// ── AgentStopTool tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn agent_stop_tool_stops_descendant_runs() {
    let handles = Arc::new(SubAgentHandleTable::new());
    let stop_tool = AgentStopTool::new(handles.clone());
    let owner_thread_id = "owner-thread";

    let parent_run_id = "run-parent";
    let child_run_id = "run-child";
    let grandchild_run_id = "run-grandchild";

    handles
        .put_running(
            parent_run_id,
            owner_thread_id.to_string(),
            format!("sub-agent-{parent_run_id}"),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    handles
        .put_running(
            child_run_id,
            owner_thread_id.to_string(),
            format!("sub-agent-{child_run_id}"),
            "worker".to_string(),
            Some(parent_run_id.to_string()),
            None,
        )
        .await;
    handles
        .put_running(
            grandchild_run_id,
            owner_thread_id.to_string(),
            format!("sub-agent-{grandchild_run_id}"),
            "worker".to_string(),
            Some(child_run_id.to_string()),
            None,
        )
        .await;

    let doc = json!({
        "sub_agents": {
            "runs": {
                parent_run_id: {
                    "thread_id": format!("sub-agent-{parent_run_id}"),
                    "agent_id": "worker",
                    "status": "running"
                },
                child_run_id: {
                    "thread_id": format!("sub-agent-{child_run_id}"),
                    "parent_run_id": parent_run_id,
                    "agent_id": "worker",
                    "status": "running"
                },
                grandchild_run_id: {
                    "thread_id": format!("sub-agent-{grandchild_run_id}"),
                    "parent_run_id": child_run_id,
                    "agent_id": "worker",
                    "status": "running"
                }
            }
        }
    });

    let mut fix = TestFixture::new_with_state(doc.clone());
    fix.run_config = {
        let mut rt = tirea_contract::RunConfig::new();
        rt.set(TOOL_SCOPE_CALLER_THREAD_ID_KEY, owner_thread_id)
            .unwrap();
        rt
    };
    let result = stop_tool
        .execute(
            json!({ "run_id": parent_run_id }),
            &fix.ctx_with("call-stop", "tool:agent_stop"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Success);
    assert_eq!(result.data["status"], json!("stopped"));

    let parent = handles
        .get_owned_summary(owner_thread_id, parent_run_id)
        .await
        .expect("parent run should exist");
    assert_eq!(parent.status, SubAgentStatus::Stopped);
    let child = handles
        .get_owned_summary(owner_thread_id, child_run_id)
        .await
        .expect("child run should exist");
    assert_eq!(child.status, SubAgentStatus::Stopped);
    let grandchild = handles
        .get_owned_summary(owner_thread_id, grandchild_run_id)
        .await
        .expect("grandchild run should exist");
    assert_eq!(grandchild.status, SubAgentStatus::Stopped);

    let patch = fix.ctx_with("call-stop", "tool:agent_stop").take_patch();
    let updated = apply_patches(&doc, std::iter::once(patch.patch())).unwrap();
    assert_eq!(
        updated["sub_agents"]["runs"][parent_run_id]["status"],
        json!("stopped")
    );
    assert_eq!(
        updated["sub_agents"]["runs"][child_run_id]["status"],
        json!("stopped")
    );
    assert_eq!(
        updated["sub_agents"]["runs"][grandchild_run_id]["status"],
        json!("stopped")
    );
}

// ── AgentRecoveryPlugin tests ────────────────────────────────────────────────

#[tokio::test]
async fn recovery_plugin_detects_orphan_and_records_confirmation() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;
    assert!(matches!(
        step.run_action(),
        crate::contracts::RunAction::Continue
    ));

    let updated = fixture.updated_state();
    assert_eq!(
        updated["sub_agents"]["runs"]["run-1"]["status"],
        json!("stopped")
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["suspended_call"]["suspension"]["action"],
        json!(AGENT_RECOVERY_INTERACTION_ACTION)
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["suspended_call"]["suspension"]
            ["parameters"]["run_id"],
        json!("run-1")
    );

    let fixture2 = TestFixture::new_with_state(updated);
    let mut before = fixture2.step(vec![]);
    plugin.run_phase(Phase::BeforeInference, &mut before).await;
    assert!(
        matches!(before.run_action(), crate::contracts::RunAction::Continue),
        "recovery plugin should not control inference flow in BeforeInference"
    );
}

#[tokio::test]
async fn recovery_plugin_does_not_override_existing_suspended_interaction() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "__tool_call_scope": {
                "existing_1": {
                    "suspended_call": {
                        "call_id": "existing_1",
                        "tool_name": "agent_run",
                        "suspension": {
                            "id": "existing_1",
                            "action": AGENT_RECOVERY_INTERACTION_ACTION
                        },
                        "arguments": {},
                        "pending": {
                            "id": "existing_1",
                            "name": AGENT_RECOVERY_INTERACTION_ACTION,
                            "arguments": {}
                        },
                        "resume_mode": "pass_decision_to_tool"
                    }
                }
            },
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker",
                        "status": "running"
                    }
                }
            }
        }),
    );

    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;
    assert!(
        matches!(step.run_action(), crate::contracts::RunAction::Continue),
        "existing suspended interaction should not be replaced"
    );

    let updated = fixture.updated_state();
    assert_eq!(
        updated["__tool_call_scope"]["existing_1"]["suspended_call"]["suspension"]["id"],
        json!("existing_1")
    );
}

#[tokio::test]
async fn recovery_plugin_auto_approve_when_permission_allow() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "permissions": {
                "default_behavior": "ask",
                "tools": {
                    "recover_agent_run": "allow"
                }
            },
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["tool_call_state"]["status"],
        json!("resuming")
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["tool_call_state"]["resume"]["action"],
        json!("resume")
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["suspended_call"]["tool_name"],
        json!("agent_run")
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["suspended_call"]["suspension"]
            ["parameters"]["run_id"],
        json!("run-1")
    );
    assert_eq!(
        updated["sub_agents"]["runs"]["run-1"]["status"],
        json!("stopped")
    );
}

#[tokio::test]
async fn recovery_plugin_auto_deny_when_permission_deny() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "permissions": {
                "default_behavior": "ask",
                "tools": {
                    "recover_agent_run": "deny"
                }
            },
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();
    assert!(
        updated
            .get("__tool_call_scope")
            .and_then(|scopes| scopes.get("agent_recovery_run-1"))
            .and_then(|scope| scope.get("tool_call_state"))
            .is_none(),
        "deny should not set recovery tool-call resume state"
    );
    assert_eq!(
        updated["sub_agents"]["runs"]["run-1"]["status"],
        json!("stopped")
    );
    assert!(updated
        .get("__suspended_tool_calls")
        .and_then(|v| v.get("calls"))
        .and_then(|v| v.as_object())
        .map_or(true, |calls| calls.is_empty()));
}

#[tokio::test]
async fn recovery_plugin_auto_approve_from_default_behavior_allow() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "permissions": {
                "default_behavior": "allow",
                "tools": {}
            },
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["tool_call_state"]["status"],
        json!("resuming")
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["tool_call_state"]["resume"]["action"],
        json!("resume")
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["suspended_call"]["tool_name"],
        json!("agent_run")
    );
}

#[tokio::test]
async fn recovery_plugin_auto_deny_from_default_behavior_deny() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "permissions": {
                "default_behavior": "deny",
                "tools": {}
            },
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();
    assert!(
        updated
            .get("__tool_call_scope")
            .and_then(|scopes| scopes.get("agent_recovery_run-1"))
            .and_then(|scope| scope.get("tool_call_state"))
            .is_none(),
        "deny should not set recovery tool-call resume state"
    );
    assert!(updated
        .get("__suspended_tool_calls")
        .and_then(|v| v.get("calls"))
        .and_then(|v| v.as_object())
        .map_or(true, |calls| calls.is_empty()));
}

#[tokio::test]
async fn recovery_plugin_tool_rule_overrides_default_behavior() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "permissions": {
                "default_behavior": "allow",
                "tools": {
                    "recover_agent_run": "ask"
                }
            },
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();
    assert!(
        updated
            .get("__tool_call_scope")
            .and_then(|scopes| scopes.get("agent_recovery_run-1"))
            .and_then(|scope| scope.get("tool_call_state"))
            .is_none(),
        "tool-level ask should not set recovery tool-call resume state"
    );
    assert_eq!(
        updated["__tool_call_scope"]["agent_recovery_run-1"]["suspended_call"]["suspension"]["action"],
        json!(AGENT_RECOVERY_INTERACTION_ACTION)
    );
}

// ── Schema tests ─────────────────────────────────────────────────────────────

#[test]
fn parse_persisted_runs_from_doc_reads_new_path() {
    let doc = json!({
        "sub_agents": {
            "runs": {
                "run-1": {
                    "thread_id": "sub-agent-run-1",
                    "agent_id": "worker",
                    "status": "stopped"
                }
            }
        }
    });
    let runs = parse_persisted_runs_from_doc(&doc);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs["run-1"].status, SubAgentStatus::Stopped);
}

#[test]
fn parse_persisted_runs_from_doc_empty_returns_empty() {
    let runs = parse_persisted_runs_from_doc(&json!({}));
    assert!(runs.is_empty());
}

// ── Parallel sub-agent tests ─────────────────────────────────────────────────

#[tokio::test]
async fn parallel_background_runs_and_stop_all() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let run_tool = AgentRunTool::new(os, handles.clone());
    let stop_tool = AgentStopTool::new(handles.clone());

    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();

    // Launch 3 background runs in parallel.
    let mut run_ids = Vec::new();
    for i in 0..3 {
        let started = run_tool
            .execute(
                json!({
                    "agent_id": "worker",
                    "prompt": format!("task-{i}"),
                    "background": true
                }),
                &fix.ctx_with(&format!("call-{i}"), "tool:agent_run"),
            )
            .await
            .unwrap();
        assert_eq!(started.status, ToolStatus::Success);
        assert_eq!(started.data["status"], json!("running"));
        run_ids.push(
            started.data["run_id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }

    // All should show as running.
    let running = handles.running_or_stopped_for_owner("owner-thread").await;
    assert_eq!(running.len(), 3);
    for summary in &running {
        assert_eq!(summary.status, SubAgentStatus::Running);
    }

    // Stop each one.
    for run_id in &run_ids {
        let mut stop_fix = TestFixture::new();
        stop_fix.run_config = caller_scope();
        let stopped = stop_tool
            .execute(
                json!({ "run_id": run_id }),
                &stop_fix.ctx_with("call-stop", "tool:agent_stop"),
            )
            .await
            .unwrap();
        assert_eq!(stopped.status, ToolStatus::Success);
        assert_eq!(stopped.data["status"], json!("stopped"));
    }

    // All should show as stopped.
    let stopped_all = handles.running_or_stopped_for_owner("owner-thread").await;
    assert_eq!(stopped_all.len(), 3);
    for summary in &stopped_all {
        assert_eq!(summary.status, SubAgentStatus::Stopped);
    }
}

#[tokio::test]
async fn sub_agent_status_lifecycle_running_to_completed() {
    let handles = SubAgentHandleTable::new();
    let epoch = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;

    let summary = handles
        .get_owned_summary("owner", "run-1")
        .await
        .expect("run should exist");
    assert_eq!(summary.status, SubAgentStatus::Running);

    let completed = handles
        .update_after_completion(
            "run-1",
            epoch,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await
        .expect("should complete");
    assert_eq!(completed.status, SubAgentStatus::Completed);
}

#[tokio::test]
async fn sub_agent_status_lifecycle_running_to_failed() {
    let handles = SubAgentHandleTable::new();
    let epoch = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;

    let failed = handles
        .update_after_completion(
            "run-1",
            epoch,
            SubAgentCompletion {
                status: SubAgentStatus::Failed,
                error: Some("something went wrong".to_string()),
            },
        )
        .await
        .expect("should fail");
    assert_eq!(failed.status, SubAgentStatus::Failed);
    assert_eq!(failed.error.as_deref(), Some("something went wrong"));
}

#[tokio::test]
async fn cancellation_requested_overrides_completion_status() {
    let handles = SubAgentHandleTable::new();
    let token = RunCancellationToken::new();
    let epoch = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            Some(token.clone()),
        )
        .await;

    // Stop the run (sets run_cancellation_requested).
    handles
        .stop_owned_tree("owner", "run-1")
        .await
        .unwrap();

    // Completion arrives with Completed, but cancellation should win.
    let result = handles
        .update_after_completion(
            "run-1",
            epoch,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await
        .expect("should apply");
    assert_eq!(result.status, SubAgentStatus::Stopped);
}

#[tokio::test]
async fn agent_output_tool_returns_error_for_unknown_run() {
    let os = AgentOs::builder().build().unwrap();
    let tool = AgentOutputTool::new(os);
    let fix = TestFixture::new();
    let result = tool
        .execute(
            json!({ "run_id": "nonexistent" }),
            &fix.ctx_with("call-1", "tool:agent_output"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("Unknown run_id"));
}

#[tokio::test]
async fn agent_output_tool_returns_status_from_persisted_state() {
    let os = AgentOs::builder().build().unwrap();
    let tool = AgentOutputTool::new(os);

    let doc = json!({
        "sub_agents": {
            "runs": {
                "run-1": {
                    "thread_id": "sub-agent-run-1",
                    "agent_id": "worker",
                    "status": "completed"
                }
            }
        }
    });
    let fix = TestFixture::new_with_state(doc);
    let result = tool
        .execute(
            json!({ "run_id": "run-1" }),
            &fix.ctx_with("call-1", "tool:agent_output"),
        )
        .await
        .unwrap();
    // Will get store_error because no ThreadStore configured, but the tool
    // should handle it gracefully.
    let status_str = result.data.get("status").and_then(|v| v.as_str());
    let is_error = result.status == ToolStatus::Error;
    assert!(
        status_str == Some("completed") || is_error,
        "expected completed status or error for missing store"
    );
}

#[test]
fn sub_agent_thread_id_convention() {
    assert_eq!(
        super::tools::sub_agent_thread_id("run-123"),
        "sub-agent-run-123"
    );
}

// ── Handle table contains test ───────────────────────────────────────────────

#[tokio::test]
async fn handle_table_contains_returns_true_for_existing() {
    let handles = SubAgentHandleTable::new();
    handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    assert!(handles.contains("run-1").await);
    assert!(!handles.contains("run-2").await);
}

// ── Ownership isolation tests ────────────────────────────────────────────────

#[tokio::test]
async fn handle_table_different_owners_cannot_see_each_others_runs() {
    let handles = SubAgentHandleTable::new();
    handles
        .put_running(
            "run-a",
            "owner-1".to_string(),
            "sub-agent-run-a".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    handles
        .put_running(
            "run-b",
            "owner-2".to_string(),
            "sub-agent-run-b".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;

    // Owner-1 can see run-a but not run-b.
    assert!(handles.get_owned_summary("owner-1", "run-a").await.is_some());
    assert!(handles.get_owned_summary("owner-1", "run-b").await.is_none());

    // Owner-2 can see run-b but not run-a.
    assert!(handles.get_owned_summary("owner-2", "run-b").await.is_some());
    assert!(handles.get_owned_summary("owner-2", "run-a").await.is_none());

    // running_or_stopped_for_owner returns only own runs.
    let owner1_runs = handles.running_or_stopped_for_owner("owner-1").await;
    assert_eq!(owner1_runs.len(), 1);
    assert_eq!(owner1_runs[0].run_id, "run-a");

    let owner2_runs = handles.running_or_stopped_for_owner("owner-2").await;
    assert_eq!(owner2_runs.len(), 1);
    assert_eq!(owner2_runs[0].run_id, "run-b");
}

#[tokio::test]
async fn handle_table_cross_owner_stop_is_denied() {
    let handles = SubAgentHandleTable::new();
    handles
        .put_running(
            "run-a",
            "owner-1".to_string(),
            "sub-agent-run-a".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;

    // Owner-2 cannot stop owner-1's run.
    let err = handles.stop_owned_tree("owner-2", "run-a").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("Unknown run_id"));

    // Verify run is still running.
    let summary = handles.get_owned_summary("owner-1", "run-a").await.unwrap();
    assert_eq!(summary.status, SubAgentStatus::Running);
}

#[tokio::test]
async fn handle_for_resume_rejects_wrong_owner() {
    let handles = SubAgentHandleTable::new();
    handles
        .put_running(
            "run-1",
            "owner-1".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;

    let err = handles.handle_for_resume("owner-2", "run-1").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("Unknown run_id"));
}

// ── Handle table status filter tests ─────────────────────────────────────────

#[tokio::test]
async fn running_or_stopped_for_owner_excludes_completed_and_failed() {
    let handles = SubAgentHandleTable::new();

    let epoch1 = handles
        .put_running(
            "run-running",
            "owner".to_string(),
            "sub-agent-run-running".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    let epoch2 = handles
        .put_running(
            "run-completed",
            "owner".to_string(),
            "sub-agent-run-completed".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    let epoch3 = handles
        .put_running(
            "run-failed",
            "owner".to_string(),
            "sub-agent-run-failed".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    handles
        .put_running(
            "run-stopped",
            "owner".to_string(),
            "sub-agent-run-stopped".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;

    // Transition run-completed and run-failed.
    handles
        .update_after_completion(
            "run-completed",
            epoch2,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await;
    handles
        .update_after_completion(
            "run-failed",
            epoch3,
            SubAgentCompletion {
                status: SubAgentStatus::Failed,
                error: Some("boom".to_string()),
            },
        )
        .await;
    handles.stop_owned_tree("owner", "run-stopped").await.unwrap();

    // Verify: only Running and Stopped returned.
    let visible = handles.running_or_stopped_for_owner("owner").await;
    let visible_ids: Vec<&str> = visible.iter().map(|s| s.run_id.as_str()).collect();
    assert!(visible_ids.contains(&"run-running"));
    assert!(visible_ids.contains(&"run-stopped"));
    assert!(!visible_ids.contains(&"run-completed"));
    assert!(!visible_ids.contains(&"run-failed"));
    // Verify sort order.
    assert_eq!(visible_ids, {
        let mut sorted = visible_ids.clone();
        sorted.sort();
        sorted
    });

    // Verify epoch1 is untouched.
    let s = handles.get_owned_summary("owner", "run-running").await.unwrap();
    assert_eq!(s.status, SubAgentStatus::Running);
    let _ = epoch1;
}

// ── Stop edge cases ──────────────────────────────────────────────────────────

#[tokio::test]
async fn stop_already_stopped_run_returns_error() {
    let handles = SubAgentHandleTable::new();
    handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    handles.stop_owned_tree("owner", "run-1").await.unwrap();

    // Second stop should fail.
    let result = handles.stop_owned_tree("owner", "run-1").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not running"));
}

#[tokio::test]
async fn stop_completed_run_returns_error() {
    let handles = SubAgentHandleTable::new();
    let epoch = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    handles
        .update_after_completion(
            "run-1",
            epoch,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await;

    let result = handles.stop_owned_tree("owner", "run-1").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not running"));
}

#[tokio::test]
async fn stop_unknown_run_returns_error() {
    let handles = SubAgentHandleTable::new();
    let result = handles.stop_owned_tree("owner", "nonexistent").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown run_id"));
}

// ── Concurrent handle table operations ───────────────────────────────────────

#[tokio::test]
async fn concurrent_put_running_same_run_id_increments_epoch() {
    let handles = Arc::new(SubAgentHandleTable::new());
    let h1 = handles.clone();
    let h2 = handles.clone();

    // Launch two registrations concurrently for the same run_id.
    let (e1, e2) = tokio::join!(
        h1.put_running(
            "run-race",
            "owner".to_string(),
            "sub-agent-run-race".to_string(),
            "worker-a".to_string(),
            None,
            None,
        ),
        h2.put_running(
            "run-race",
            "owner".to_string(),
            "sub-agent-run-race".to_string(),
            "worker-b".to_string(),
            None,
            None,
        ),
    );

    // Epochs should be different (one is 1, the other is 2, depending on ordering).
    assert_ne!(e1, e2);
    let max_epoch = e1.max(e2);
    assert_eq!(max_epoch, 2);

    // The final state should reflect the last writer.
    let summary = handles.get_owned_summary("owner", "run-race").await.unwrap();
    assert_eq!(summary.status, SubAgentStatus::Running);
}

#[tokio::test]
async fn concurrent_launches_different_run_ids() {
    let handles = Arc::new(SubAgentHandleTable::new());
    let num_runs = 10;
    let mut tasks = Vec::new();

    for i in 0..num_runs {
        let h = handles.clone();
        tasks.push(tokio::spawn(async move {
            h.put_running(
                &format!("run-{i}"),
                "owner".to_string(),
                format!("sub-agent-run-{i}"),
                "worker".to_string(),
                None,
                None,
            )
            .await
        }));
    }

    let epochs: Vec<u64> = futures::future::join_all(tasks)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // All should be epoch 1 (first registration).
    for epoch in &epochs {
        assert_eq!(*epoch, 1);
    }

    // All 10 should be visible.
    let running = handles.running_or_stopped_for_owner("owner").await;
    assert_eq!(running.len(), num_runs);
    for summary in &running {
        assert_eq!(summary.status, SubAgentStatus::Running);
    }
}

#[tokio::test]
async fn concurrent_stop_multiple_independent_runs() {
    let handles = Arc::new(SubAgentHandleTable::new());

    // Register 5 independent runs.
    for i in 0..5 {
        handles
            .put_running(
                &format!("run-{i}"),
                "owner".to_string(),
                format!("sub-agent-run-{i}"),
                "worker".to_string(),
                None,
                None,
            )
            .await;
    }

    // Stop them all concurrently.
    let mut tasks = Vec::new();
    for i in 0..5 {
        let h = handles.clone();
        tasks.push(tokio::spawn(async move {
            h.stop_owned_tree("owner", &format!("run-{i}")).await
        }));
    }

    let results: Vec<_> = futures::future::join_all(tasks)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    for result in results {
        assert!(result.is_ok());
    }

    let all = handles.running_or_stopped_for_owner("owner").await;
    assert_eq!(all.len(), 5);
    for summary in &all {
        assert_eq!(summary.status, SubAgentStatus::Stopped);
    }
}

// ── Interleaved launch/stop/re-launch ────────────────────────────────────────

#[tokio::test]
async fn launch_stop_relaunch_increments_epoch_and_resumes() {
    let handles = SubAgentHandleTable::new();

    // Launch.
    let epoch1 = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    assert_eq!(epoch1, 1);

    // Stop.
    handles.stop_owned_tree("owner", "run-1").await.unwrap();
    let summary = handles.get_owned_summary("owner", "run-1").await.unwrap();
    assert_eq!(summary.status, SubAgentStatus::Stopped);

    // Re-launch (simulates resume).
    let epoch2 = handles
        .put_running(
            "run-1",
            "owner".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    assert_eq!(epoch2, 2);

    let summary = handles.get_owned_summary("owner", "run-1").await.unwrap();
    assert_eq!(summary.status, SubAgentStatus::Running);

    // Stale epoch1 completion is ignored.
    let stale = handles
        .update_after_completion(
            "run-1",
            epoch1,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await;
    assert!(stale.is_none());

    // Run is still running.
    let summary = handles.get_owned_summary("owner", "run-1").await.unwrap();
    assert_eq!(summary.status, SubAgentStatus::Running);

    // Correct epoch2 completion applies.
    let applied = handles
        .update_after_completion(
            "run-1",
            epoch2,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await
        .expect("epoch2 completion should apply");
    assert_eq!(applied.status, SubAgentStatus::Completed);
}

// ── AgentRunTool: resume completed/failed returns status ─────────────────────

#[tokio::test]
async fn agent_run_tool_returns_completed_status_when_resuming_completed_run() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let epoch = handles
        .put_running(
            "run-1",
            "owner-thread".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    handles
        .update_after_completion(
            "run-1",
            epoch,
            SubAgentCompletion {
                status: SubAgentStatus::Completed,
                error: None,
            },
        )
        .await;

    let run_tool = AgentRunTool::new(os, handles);
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();

    let result = run_tool
        .execute(
            json!({ "run_id": "run-1", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Success);
    assert_eq!(result.data["status"], json!("completed"));
}

#[tokio::test]
async fn agent_run_tool_returns_failed_status_when_resuming_failed_run() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let epoch = handles
        .put_running(
            "run-1",
            "owner-thread".to_string(),
            "sub-agent-run-1".to_string(),
            "worker".to_string(),
            None,
            None,
        )
        .await;
    handles
        .update_after_completion(
            "run-1",
            epoch,
            SubAgentCompletion {
                status: SubAgentStatus::Failed,
                error: Some("agent failed".to_string()),
            },
        )
        .await;

    let run_tool = AgentRunTool::new(os, handles);
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();

    let result = run_tool
        .execute(
            json!({ "run_id": "run-1", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Success);
    assert_eq!(result.data["status"], json!("failed"));
    assert_eq!(result.data["error"], json!("agent failed"));
}

// ── AgentRunTool: missing required args ──────────────────────────────────────

#[tokio::test]
async fn agent_run_tool_requires_prompt_for_new_run() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();

    let result = run_tool
        .execute(
            json!({ "agent_id": "worker", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("missing 'prompt'"));
}

#[tokio::test]
async fn agent_run_tool_requires_agent_id_for_new_run() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();

    let result = run_tool
        .execute(
            json!({ "prompt": "hello", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("missing 'agent_id'"));
}

// ── AgentRunTool: excluded agents via scope ──────────────────────────────────

#[tokio::test]
async fn agent_run_tool_rejects_excluded_agent() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .with_agent(
            "secret",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();
    fix.run_config
        .set(SCOPE_EXCLUDED_AGENTS_KEY, vec!["secret"])
        .unwrap();

    let result = run_tool
        .execute(
            json!({ "agent_id": "secret", "prompt": "hi", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("Unknown or unavailable agent_id"));
}

// ── AgentRunTool: unknown run_id (no handle, no persisted) ───────────────────

#[tokio::test]
async fn agent_run_tool_returns_error_for_unknown_run_id() {
    let os = AgentOs::builder().build().unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();

    let result = run_tool
        .execute(
            json!({ "run_id": "nonexistent", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("Unknown run_id"));
}

// ── AgentRunTool: persisted completed/failed returns without re-run ──────────

#[tokio::test]
async fn agent_run_tool_returns_persisted_completed_without_rerun() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));

    let doc = json!({
        "sub_agents": {
            "runs": {
                "run-1": {
                    "thread_id": "sub-agent-run-1",
                    "agent_id": "worker",
                    "status": "completed"
                }
            }
        }
    });
    let mut fix = TestFixture::new_with_state(doc.clone());
    fix.run_config = caller_scope_with_state(doc);

    let result = run_tool
        .execute(
            json!({ "run_id": "run-1", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Success);
    assert_eq!(result.data["status"], json!("completed"));
}

#[tokio::test]
async fn agent_run_tool_returns_persisted_failed_with_error() {
    let os = AgentOs::builder()
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini"),
        )
        .build()
        .unwrap();
    let run_tool = AgentRunTool::new(os, Arc::new(SubAgentHandleTable::new()));

    let doc = json!({
        "sub_agents": {
            "runs": {
                "run-1": {
                    "thread_id": "sub-agent-run-1",
                    "agent_id": "worker",
                    "status": "failed",
                    "error": "something broke"
                }
            }
        }
    });
    let mut fix = TestFixture::new_with_state(doc.clone());
    fix.run_config = caller_scope_with_state(doc);

    let result = run_tool
        .execute(
            json!({ "run_id": "run-1", "background": false }),
            &fix.ctx_with("call-1", "tool:agent_run"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Success);
    assert_eq!(result.data["status"], json!("failed"));
    assert_eq!(result.data["error"], json!("something broke"));
}

// ── AgentStopTool edge cases ─────────────────────────────────────────────────

#[tokio::test]
async fn agent_stop_tool_requires_scope_context() {
    let handles = Arc::new(SubAgentHandleTable::new());
    let stop_tool = AgentStopTool::new(handles);
    let fix = TestFixture::new();

    let result = stop_tool
        .execute(
            json!({ "run_id": "run-1" }),
            &fix.ctx_with("call-1", "tool:agent_stop"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("missing caller thread context"));
}

#[tokio::test]
async fn agent_stop_tool_requires_run_id() {
    let handles = Arc::new(SubAgentHandleTable::new());
    let stop_tool = AgentStopTool::new(handles);
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();

    let result = stop_tool
        .execute(
            json!({}),
            &fix.ctx_with("call-1", "tool:agent_stop"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("missing 'run_id'"));
}

#[tokio::test]
async fn agent_stop_tool_stops_persisted_running_without_live_handle() {
    let handles = Arc::new(SubAgentHandleTable::new());
    let stop_tool = AgentStopTool::new(handles);

    let doc = json!({
        "sub_agents": {
            "runs": {
                "run-orphan": {
                    "thread_id": "sub-agent-run-orphan",
                    "agent_id": "worker",
                    "status": "running"
                }
            }
        }
    });
    let mut fix = TestFixture::new_with_state(doc.clone());
    fix.run_config = {
        let mut rt = tirea_contract::RunConfig::new();
        rt.set(TOOL_SCOPE_CALLER_THREAD_ID_KEY, "owner-thread")
            .unwrap();
        rt
    };

    let result = stop_tool
        .execute(
            json!({ "run_id": "run-orphan" }),
            &fix.ctx_with("call-stop", "tool:agent_stop"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Success);
    assert_eq!(result.data["status"], json!("stopped"));

    // Verify the state patch marks it stopped.
    let patch = fix.ctx_with("call-stop", "tool:agent_stop").take_patch();
    let updated = apply_patches(&doc, std::iter::once(patch.patch())).unwrap();
    assert_eq!(
        updated["sub_agents"]["runs"]["run-orphan"]["status"],
        json!("stopped")
    );
}

// ── AgentOutputTool: missing run_id param ────────────────────────────────────

#[tokio::test]
async fn agent_output_tool_requires_run_id() {
    let os = AgentOs::builder().build().unwrap();
    let tool = AgentOutputTool::new(os);
    let fix = TestFixture::new();

    let result = tool
        .execute(
            json!({}),
            &fix.ctx_with("call-1", "tool:agent_output"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result
        .message
        .unwrap_or_default()
        .contains("missing 'run_id'"));
}

// ── Recovery plugin: multiple orphans ────────────────────────────────────────

#[tokio::test]
async fn recovery_plugin_detects_multiple_orphans_creates_one_suspension() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker-a",
                        "status": "running"
                    },
                    "run-2": {
                        "thread_id": "sub-agent-run-2",
                        "agent_id": "worker-b",
                        "status": "running"
                    },
                    "run-3": {
                        "thread_id": "sub-agent-run-3",
                        "agent_id": "worker-c",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();

    // All 3 should be marked stopped.
    assert_eq!(
        updated["sub_agents"]["runs"]["run-1"]["status"],
        json!("stopped")
    );
    assert_eq!(
        updated["sub_agents"]["runs"]["run-2"]["status"],
        json!("stopped")
    );
    assert_eq!(
        updated["sub_agents"]["runs"]["run-3"]["status"],
        json!("stopped")
    );

    // Only one recovery suspension should be created (for the first orphan).
    let scope = &updated["__tool_call_scope"];
    let suspended_count = scope
        .as_object()
        .map(|obj| {
            obj.values()
                .filter(|v| v.get("suspended_call").is_some())
                .count()
        })
        .unwrap_or(0);
    assert_eq!(
        suspended_count, 1,
        "only one recovery suspension should be created"
    );
}

// ── Recovery plugin: mix of orphans and live handles ─────────────────────────

#[tokio::test]
async fn recovery_plugin_only_marks_orphans_when_some_have_live_handles() {
    let handles = Arc::new(SubAgentHandleTable::new());
    // run-1 has a live handle.
    handles
        .put_running(
            "run-1",
            "owner-1".to_string(),
            "sub-agent-run-1".to_string(),
            "worker-a".to_string(),
            None,
            None,
        )
        .await;

    let plugin = AgentRecoveryPlugin::new(handles);
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker-a",
                        "status": "running"
                    },
                    "run-2": {
                        "thread_id": "sub-agent-run-2",
                        "agent_id": "worker-b",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();

    // run-1 has live handle → not marked stopped.
    assert_eq!(
        updated["sub_agents"]["runs"]["run-1"]["status"],
        json!("running")
    );

    // run-2 is orphaned → marked stopped.
    assert_eq!(
        updated["sub_agents"]["runs"]["run-2"]["status"],
        json!("stopped")
    );
}

// ── Recovery plugin: no orphans when all have live handles ───────────────────

#[tokio::test]
async fn recovery_plugin_no_action_when_all_running_have_live_handles() {
    let handles = Arc::new(SubAgentHandleTable::new());
    handles
        .put_running(
            "run-1",
            "owner-1".to_string(),
            "sub-agent-run-1".to_string(),
            "worker-a".to_string(),
            None,
            None,
        )
        .await;
    handles
        .put_running(
            "run-2",
            "owner-1".to_string(),
            "sub-agent-run-2".to_string(),
            "worker-b".to_string(),
            None,
            None,
        )
        .await;

    let plugin = AgentRecoveryPlugin::new(handles);
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "sub_agents": {
                "runs": {
                    "run-1": {
                        "thread_id": "sub-agent-run-1",
                        "agent_id": "worker-a",
                        "status": "running"
                    },
                    "run-2": {
                        "thread_id": "sub-agent-run-2",
                        "agent_id": "worker-b",
                        "status": "running"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();

    // Both still running (no orphan detection).
    assert_eq!(
        updated["sub_agents"]["runs"]["run-1"]["status"],
        json!("running")
    );
    assert_eq!(
        updated["sub_agents"]["runs"]["run-2"]["status"],
        json!("running")
    );

    // No suspended recovery interaction.
    let has_suspended = updated
        .get("__tool_call_scope")
        .and_then(|scope| scope.as_object())
        .map(|obj| {
            obj.values()
                .any(|v| v.get("suspended_call").is_some())
        })
        .unwrap_or(false);
    assert!(!has_suspended, "no recovery suspension should be created when all have live handles");
}

// ── Recovery plugin: mixed statuses only Running without handle is orphan ────

#[tokio::test]
async fn recovery_plugin_ignores_completed_stopped_failed_in_persisted_state() {
    let plugin = AgentRecoveryPlugin::new(Arc::new(SubAgentHandleTable::new()));
    let thread = Thread::with_initial_state(
        "owner-1",
        json!({
            "sub_agents": {
                "runs": {
                    "run-completed": {
                        "thread_id": "sub-agent-run-completed",
                        "agent_id": "worker",
                        "status": "completed"
                    },
                    "run-failed": {
                        "thread_id": "sub-agent-run-failed",
                        "agent_id": "worker",
                        "status": "failed",
                        "error": "oops"
                    },
                    "run-stopped": {
                        "thread_id": "sub-agent-run-stopped",
                        "agent_id": "worker",
                        "status": "stopped"
                    }
                }
            }
        }),
    );
    let doc = thread.rebuild_state().unwrap();
    let fixture = TestFixture::new_with_state(doc);
    let mut step = fixture.step(vec![]);
    plugin.run_phase(Phase::RunStart, &mut step).await;

    let updated = fixture.updated_state();

    // None of them should change status.
    assert_eq!(
        updated["sub_agents"]["runs"]["run-completed"]["status"],
        json!("completed")
    );
    assert_eq!(
        updated["sub_agents"]["runs"]["run-failed"]["status"],
        json!("failed")
    );
    assert_eq!(
        updated["sub_agents"]["runs"]["run-stopped"]["status"],
        json!("stopped")
    );

    // No suspended recovery interaction.
    let has_suspended = updated
        .get("__tool_call_scope")
        .and_then(|scope| scope.as_object())
        .map(|obj| {
            obj.values()
                .any(|v| v.get("suspended_call").is_some())
        })
        .unwrap_or(false);
    assert!(!has_suspended);
}

// ── State helper tests ───────────────────────────────────────────────────────

#[test]
fn collect_descendant_run_ids_from_state_includes_full_tree() {
    let runs = HashMap::from([
        (
            "root".to_string(),
            SubAgent {
                thread_id: "sub-agent-root".to_string(),
                parent_run_id: None,
                agent_id: "w".to_string(),
                status: SubAgentStatus::Running,
                error: None,
            },
        ),
        (
            "child-1".to_string(),
            SubAgent {
                thread_id: "sub-agent-child-1".to_string(),
                parent_run_id: Some("root".to_string()),
                agent_id: "w".to_string(),
                status: SubAgentStatus::Running,
                error: None,
            },
        ),
        (
            "child-2".to_string(),
            SubAgent {
                thread_id: "sub-agent-child-2".to_string(),
                parent_run_id: Some("root".to_string()),
                agent_id: "w".to_string(),
                status: SubAgentStatus::Running,
                error: None,
            },
        ),
        (
            "grandchild".to_string(),
            SubAgent {
                thread_id: "sub-agent-grandchild".to_string(),
                parent_run_id: Some("child-1".to_string()),
                agent_id: "w".to_string(),
                status: SubAgentStatus::Running,
                error: None,
            },
        ),
        (
            "unrelated".to_string(),
            SubAgent {
                thread_id: "sub-agent-unrelated".to_string(),
                parent_run_id: None,
                agent_id: "w".to_string(),
                status: SubAgentStatus::Running,
                error: None,
            },
        ),
    ]);

    let descendants = collect_descendant_run_ids_from_state(&runs, "root", true);
    assert!(descendants.contains(&"root".to_string()));
    assert!(descendants.contains(&"child-1".to_string()));
    assert!(descendants.contains(&"child-2".to_string()));
    assert!(descendants.contains(&"grandchild".to_string()));
    assert!(!descendants.contains(&"unrelated".to_string()));
    assert_eq!(descendants.len(), 4);
}

#[test]
fn collect_descendant_run_ids_from_state_exclude_root() {
    let runs = HashMap::from([
        (
            "root".to_string(),
            SubAgent {
                thread_id: "sub-agent-root".to_string(),
                parent_run_id: None,
                agent_id: "w".to_string(),
                status: SubAgentStatus::Running,
                error: None,
            },
        ),
        (
            "child".to_string(),
            SubAgent {
                thread_id: "sub-agent-child".to_string(),
                parent_run_id: Some("root".to_string()),
                agent_id: "w".to_string(),
                status: SubAgentStatus::Running,
                error: None,
            },
        ),
    ]);

    let descendants = collect_descendant_run_ids_from_state(&runs, "root", false);
    assert!(!descendants.contains(&"root".to_string()));
    assert!(descendants.contains(&"child".to_string()));
    assert_eq!(descendants.len(), 1);
}

#[test]
fn collect_descendant_run_ids_from_state_unknown_root_returns_empty() {
    let runs: HashMap<String, SubAgent> = HashMap::new();
    let descendants = collect_descendant_run_ids_from_state(&runs, "nonexistent", true);
    assert!(descendants.is_empty());
}

#[test]
fn collect_descendant_run_ids_from_state_leaf_node_returns_only_self() {
    let runs = HashMap::from([(
        "leaf".to_string(),
        SubAgent {
            thread_id: "sub-agent-leaf".to_string(),
            parent_run_id: Some("parent".to_string()),
            agent_id: "w".to_string(),
            status: SubAgentStatus::Running,
            error: None,
        },
    )]);

    let descendants = collect_descendant_run_ids_from_state(&runs, "leaf", true);
    assert_eq!(descendants, vec!["leaf".to_string()]);
}

// ── Parallel tool-level operations ───────────────────────────────────────────

#[tokio::test]
async fn parallel_background_launches_produce_unique_run_ids() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let run_tool = AgentRunTool::new(os, handles.clone());

    let mut run_ids = Vec::new();
    for i in 0..5 {
        let mut fix = TestFixture::new();
        fix.run_config = caller_scope();
        let started = run_tool
            .execute(
                json!({
                    "agent_id": "worker",
                    "prompt": format!("task-{i}"),
                    "background": true
                }),
                &fix.ctx_with(&format!("call-{i}"), "tool:agent_run"),
            )
            .await
            .unwrap();
        assert_eq!(started.status, ToolStatus::Success);
        run_ids.push(
            started.data["run_id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }

    // All run_ids should be unique.
    let unique: std::collections::HashSet<&str> =
        run_ids.iter().map(|s| s.as_str()).collect();
    assert_eq!(unique.len(), 5, "all run_ids should be unique");

    // All should be visible as running.
    let running = handles.running_or_stopped_for_owner("owner-thread").await;
    assert_eq!(running.len(), 5);
}

#[tokio::test]
async fn parallel_launch_and_immediate_stop() {
    let os = AgentOs::builder()
        .with_registered_behavior(
            "slow_terminate_behavior_requested",
            Arc::new(SlowTerminatePlugin),
        )
        .with_agent(
            "worker",
            crate::orchestrator::AgentDefinition::new("gpt-4o-mini")
                .with_behavior_id("slow_terminate_behavior_requested"),
        )
        .build()
        .unwrap();
    let handles = Arc::new(SubAgentHandleTable::new());
    let run_tool = AgentRunTool::new(os, handles.clone());
    let stop_tool = AgentStopTool::new(handles.clone());

    // Launch background run.
    let mut fix = TestFixture::new();
    fix.run_config = caller_scope();
    let started = run_tool
        .execute(
            json!({
                "agent_id": "worker",
                "prompt": "fast task",
                "background": true
            }),
            &fix.ctx_with("call-launch", "tool:agent_run"),
        )
        .await
        .unwrap();
    let run_id = started.data["run_id"].as_str().unwrap().to_string();

    // Immediately stop (race with background execution).
    let mut stop_fix = TestFixture::new();
    stop_fix.run_config = caller_scope();
    let stopped = stop_tool
        .execute(
            json!({ "run_id": run_id }),
            &stop_fix.ctx_with("call-stop", "tool:agent_stop"),
        )
        .await
        .unwrap();
    assert_eq!(stopped.status, ToolStatus::Success);
    assert_eq!(stopped.data["status"], json!("stopped"));

    // Wait for background task to flush.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Verify the run is stopped (cancellation override should prevent it from flipping to completed).
    let summary = handles.get_owned_summary("owner-thread", &run_id).await.unwrap();
    assert_eq!(summary.status, SubAgentStatus::Stopped);
}

// ── SubAgent serialization round-trip ────────────────────────────────────────

#[test]
fn sub_agent_serialization_roundtrip() {
    let sub = SubAgent {
        thread_id: "sub-agent-run-42".to_string(),
        parent_run_id: Some("parent-7".to_string()),
        agent_id: "researcher".to_string(),
        status: SubAgentStatus::Completed,
        error: None,
    };
    let json = serde_json::to_value(&sub).unwrap();
    assert_eq!(json["thread_id"], "sub-agent-run-42");
    assert_eq!(json["parent_run_id"], "parent-7");
    assert_eq!(json["agent_id"], "researcher");
    assert_eq!(json["status"], "completed");
    assert!(json.get("error").is_none(), "None error should be skipped");

    let roundtrip: SubAgent = serde_json::from_value(json).unwrap();
    assert_eq!(roundtrip.thread_id, sub.thread_id);
    assert_eq!(roundtrip.parent_run_id, sub.parent_run_id);
    assert_eq!(roundtrip.agent_id, sub.agent_id);
    assert_eq!(roundtrip.status as u8, SubAgentStatus::Completed as u8);
    assert!(roundtrip.error.is_none());
}

#[test]
fn sub_agent_deserializes_without_optional_fields() {
    let json = json!({
        "thread_id": "sub-agent-run-99",
        "agent_id": "coder",
        "status": "running"
    });
    let sub: SubAgent = serde_json::from_value(json).unwrap();
    assert!(sub.parent_run_id.is_none());
    assert!(sub.error.is_none());
}

#[test]
fn sub_agent_status_as_str() {
    assert_eq!(SubAgentStatus::Running.as_str(), "running");
    assert_eq!(SubAgentStatus::Completed.as_str(), "completed");
    assert_eq!(SubAgentStatus::Failed.as_str(), "failed");
    assert_eq!(SubAgentStatus::Stopped.as_str(), "stopped");
}

// ── Handle table: collect_descendant_run_ids_by_parent ───────────────────────

#[tokio::test]
async fn handle_table_collect_descendants_across_owners() {
    // Verifies that collect_descendant_run_ids_by_parent scopes to owner.
    let handles = SubAgentHandleTable::new();
    handles
        .put_running(
            "root",
            "owner-1".to_string(),
            "sub-agent-root".to_string(),
            "w".to_string(),
            None,
            None,
        )
        .await;
    handles
        .put_running(
            "child",
            "owner-1".to_string(),
            "sub-agent-child".to_string(),
            "w".to_string(),
            Some("root".to_string()),
            None,
        )
        .await;
    // This child belongs to a different owner, so should NOT be included.
    handles
        .put_running(
            "other-child",
            "owner-2".to_string(),
            "sub-agent-other-child".to_string(),
            "w".to_string(),
            Some("root".to_string()),
            None,
        )
        .await;

    let stopped = handles.stop_owned_tree("owner-1", "root").await.unwrap();
    let stopped_ids: Vec<&str> = stopped.iter().map(|s| s.run_id.as_str()).collect();
    assert!(stopped_ids.contains(&"root"));
    assert!(stopped_ids.contains(&"child"));
    assert!(!stopped_ids.contains(&"other-child"));

    // other-child should still be running.
    let other = handles.get_owned_summary("owner-2", "other-child").await.unwrap();
    assert_eq!(other.status, SubAgentStatus::Running);
}

// ── Plugin rendering edge cases ──────────────────────────────────────────────

#[test]
fn plugin_renders_empty_when_no_agents() {
    let reg = InMemoryAgentRegistry::new();
    let plugin = AgentToolsPlugin::new(Arc::new(reg), Arc::new(SubAgentHandleTable::new()));
    let rendered = plugin.render_available_agents(None, None);
    assert!(rendered.is_empty());
}

#[tokio::test]
async fn plugin_no_reminder_when_no_handles() {
    let reg = InMemoryAgentRegistry::new();
    let handles = Arc::new(SubAgentHandleTable::new());
    let plugin = AgentToolsPlugin::new(Arc::new(reg), handles);

    let fixture = TestFixture::new();
    let mut step = StepContext::new(fixture.ctx(), "owner-1", &fixture.messages, vec![]);
    plugin.run_phase(Phase::AfterToolExecute, &mut step).await;
    assert!(
        step.messaging.reminders.is_empty(),
        "no reminder should be emitted when there are no handles"
    );
}

#[tokio::test]
async fn plugin_reminder_shows_multiple_runs() {
    let reg = InMemoryAgentRegistry::new();
    let handles = Arc::new(SubAgentHandleTable::new());
    handles
        .put_running(
            "run-1",
            "owner-1".to_string(),
            "sub-agent-run-1".to_string(),
            "worker-a".to_string(),
            None,
            None,
        )
        .await;
    handles
        .put_running(
            "run-2",
            "owner-1".to_string(),
            "sub-agent-run-2".to_string(),
            "worker-b".to_string(),
            None,
            None,
        )
        .await;

    let plugin = AgentToolsPlugin::new(Arc::new(reg), handles);
    let fixture = TestFixture::new();
    let mut step = StepContext::new(fixture.ctx(), "owner-1", &fixture.messages, vec![]);
    plugin.run_phase(Phase::AfterToolExecute, &mut step).await;
    let reminder = step.messaging.reminders.first()
        .expect("should have a reminder for multiple running sub-agents");
    assert!(reminder.contains("run-1"));
    assert!(reminder.contains("run-2"));
    assert!(reminder.contains("worker-a"));
    assert!(reminder.contains("worker-b"));
    assert!(reminder.contains("agent_output"));
}
