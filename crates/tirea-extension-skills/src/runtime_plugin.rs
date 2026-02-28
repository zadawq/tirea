use crate::SKILLS_RUNTIME_PLUGIN_ID;
use async_trait::async_trait;
use serde_json::Value;
use tirea_contract::runtime::plugin::agent::{AgentBehavior, ReadOnlyContext};
use tirea_contract::runtime::plugin::phase::effect::PhaseOutput;
use tirea_contract::runtime::plugin::phase::state_spec::{reduce_state_actions, AnyStateAction};
use tirea_contract::runtime::plugin::phase::AnyPluginAction;
use tirea_state::TrackedPatch;

/// Placeholder plugin for activated skill state.
///
/// Skill instructions are injected via `append_user_messages` (single injection path)
/// and tool results for references/scripts/assets are already visible in conversation
/// history. This plugin no longer injects system context to avoid token waste from
/// duplicate injection.
#[derive(Debug, Default, Clone)]
pub struct SkillRuntimePlugin;

impl SkillRuntimePlugin {
    pub fn new() -> Self {
        Self
    }
}

pub(crate) enum SkillRuntimeAction {
    ApplyPatch { patch: TrackedPatch },
}

impl From<SkillRuntimeAction> for AnyPluginAction {
    fn from(action: SkillRuntimeAction) -> Self {
        AnyPluginAction::new(SKILLS_RUNTIME_PLUGIN_ID, action)
    }
}

#[async_trait]
impl AgentBehavior for SkillRuntimePlugin {
    fn id(&self) -> &str {
        SKILLS_RUNTIME_PLUGIN_ID
    }

    async fn before_inference(&self, _ctx: &ReadOnlyContext<'_>) -> PhaseOutput {
        // No-op: skill content is delivered via append_user_messages and tool results.
        PhaseOutput::default()
    }

    fn reduce_plugin_actions(
        &self,
        actions: Vec<AnyPluginAction>,
        base_snapshot: &Value,
    ) -> Result<Vec<TrackedPatch>, String> {
        let mut state_actions = Vec::new();
        for action in actions {
            if action.plugin_id() != SKILLS_RUNTIME_PLUGIN_ID {
                return Err(format!(
                    "skills runtime plugin received action for unexpected plugin '{}'",
                    action.plugin_id()
                ));
            }
            let action = action.downcast::<SkillRuntimeAction>().map_err(|other| {
                format!(
                    "skills runtime plugin failed to downcast action '{}'",
                    other.action_type_name()
                )
            })?;
            match action {
                SkillRuntimeAction::ApplyPatch { patch } => {
                    state_actions.push(AnyStateAction::Patch(patch));
                }
            }
        }

        reduce_state_actions(state_actions, base_snapshot, "plugin:skills_runtime")
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tirea_contract::runtime::plugin::phase::Phase;
    use tirea_contract::RunConfig;
    use tirea_state::DocCell;

    #[tokio::test]
    async fn plugin_does_not_inject_system_context() {
        let state = json!({
            "skills": {
                "active": ["a"],
                "instructions": {"a": "Do X"},
                "references": {},
                "scripts": {}
            }
        });
        let p = SkillRuntimePlugin::new();
        let config = RunConfig::new();
        let doc = DocCell::new(state);
        let ctx = ReadOnlyContext::new(Phase::BeforeInference, "t1", &[], &config, &doc);
        let output = AgentBehavior::before_inference(&p, &ctx).await;
        assert!(
            output.is_empty(),
            "runtime plugin should not inject system context"
        );
    }
}
