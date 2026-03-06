//! Truncation recovery logic for the agent loop.
//!
//! When the LLM stops due to `MaxTokens` without emitting complete tool
//! calls, this module provides helpers to inject a continuation prompt and
//! re-enter inference.

use super::run_state::LoopRunState;
use crate::contracts::runtime::inference::{StopReason, StreamResult};
use crate::contracts::thread::{Message, Visibility};

/// Maximum number of truncation recovery retries per run.
const MAX_RETRIES: usize = 3;

/// Continuation prompt sent to the model after truncation.
const CONTINUATION_PROMPT: &str =
    "Your response was cut off because it exceeded the output token limit. \
     Please break your work into smaller pieces. Continue from where you left off.";

/// Check if truncation recovery should retry inference.
///
/// Returns `true` (and increments the retry counter) when all three
/// conditions are met:
/// 1. Model stopped due to `MaxTokens`
/// 2. No complete tool calls in the response
/// 3. Haven't exceeded max retries
pub(super) fn should_retry(result: &StreamResult, run_state: &mut LoopRunState) -> bool {
    if result.stop_reason == Some(StopReason::MaxTokens)
        && result.tool_calls.is_empty()
        && run_state.truncation_retries < MAX_RETRIES
    {
        run_state.truncation_retries += 1;
        true
    } else {
        false
    }
}

/// Build the continuation prompt message (Internal visibility).
pub(super) fn continuation_message() -> Message {
    let mut msg = Message::user(CONTINUATION_PROMPT);
    msg.visibility = Visibility::Internal;
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::runtime::inference::TokenUsage;

    fn max_tokens_result() -> StreamResult {
        StreamResult {
            text: "partial output...".into(),
            tool_calls: vec![],
            usage: Some(TokenUsage {
                completion_tokens: Some(4096),
                ..Default::default()
            }),
            stop_reason: Some(StopReason::MaxTokens),
        }
    }

    fn end_turn_result() -> StreamResult {
        StreamResult {
            text: "done".into(),
            tool_calls: vec![],
            usage: None,
            stop_reason: Some(StopReason::EndTurn),
        }
    }

    fn max_tokens_with_tools() -> StreamResult {
        use crate::contracts::thread::ToolCall;
        use serde_json::json;
        StreamResult {
            text: "Using tool".into(),
            tool_calls: vec![ToolCall::new("c1", "search", json!({"q": "test"}))],
            usage: None,
            stop_reason: Some(StopReason::MaxTokens),
        }
    }

    #[test]
    fn triggers_retry_on_max_tokens_without_tools() {
        let mut state = LoopRunState::new();
        assert!(should_retry(&max_tokens_result(), &mut state));
        assert_eq!(state.truncation_retries, 1);
    }

    #[test]
    fn no_retry_on_end_turn() {
        let mut state = LoopRunState::new();
        assert!(!should_retry(&end_turn_result(), &mut state));
        assert_eq!(state.truncation_retries, 0);
    }

    #[test]
    fn no_retry_when_tool_calls_present() {
        let mut state = LoopRunState::new();
        assert!(!should_retry(&max_tokens_with_tools(), &mut state));
    }

    #[test]
    fn respects_max_retries() {
        let mut state = LoopRunState::new();
        for _ in 0..MAX_RETRIES {
            assert!(should_retry(&max_tokens_result(), &mut state));
        }
        assert!(!should_retry(&max_tokens_result(), &mut state));
        assert_eq!(state.truncation_retries, MAX_RETRIES);
    }

    #[test]
    fn continuation_message_is_internal() {
        let msg = continuation_message();
        assert_eq!(msg.visibility, Visibility::Internal);
        assert_eq!(msg.role, crate::contracts::thread::Role::User);
        assert!(!msg.content.is_empty());
    }
}
