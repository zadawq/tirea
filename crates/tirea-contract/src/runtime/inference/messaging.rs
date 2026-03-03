use crate::runtime::action::Action;
use crate::runtime::phase::step::StepContext;
use crate::runtime::phase::Phase;

/// Post-tool messaging extension: reminders and user messages.
///
/// Populated by `AddSystemReminder`, `AddUserMessage` actions
/// during `AfterToolExecute`.
#[derive(Debug, Default, Clone)]
pub struct MessagingContext {
    /// System reminders injected after tool results.
    pub reminders: Vec<String>,
    /// User messages to append after tool execution.
    pub user_messages: Vec<String>,
}

/// Append a user-role message after tool execution.
///
/// Only valid in `AfterToolExecute`. The message is written into
/// [`MessagingContext::user_messages`] and converted to a thread
/// `Message::user` by the loop after all actions are applied.
pub struct AddUserMessage(pub String);

impl Action for AddUserMessage {
    fn label(&self) -> &'static str {
        "add_user_message"
    }

    fn validate(&self, phase: Phase) -> Result<(), String> {
        if phase == Phase::AfterToolExecute {
            Ok(())
        } else {
            Err(format!(
                "AddUserMessage requires AfterToolExecute, got {phase:?}"
            ))
        }
    }

    fn apply(self: Box<Self>, step: &mut StepContext<'_>) {
        step.extensions
            .get_or_default::<MessagingContext>()
            .user_messages
            .push(self.0);
    }
}
