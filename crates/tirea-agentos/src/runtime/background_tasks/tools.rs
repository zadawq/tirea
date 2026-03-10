//! Built-in tools for querying and managing background tasks.
//!
//! These tools provide a unified interface for the LLM to check status,
//! read output, and cancel any background task regardless of type.

use super::manager::BackgroundTaskManager;
use crate::contracts::runtime::tool_call::{
    Tool, ToolCallContext, ToolDescriptor, ToolError, ToolResult,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

pub const TASK_STATUS_TOOL_ID: &str = "task_status";
pub const TASK_CANCEL_TOOL_ID: &str = "task_cancel";

fn required_string<'a>(args: &'a Value, key: &str) -> Result<&'a str, ToolResult> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            ToolResult::error(
                TASK_STATUS_TOOL_ID,
                format!("Missing required parameter: {key}"),
            )
        })
}

fn owner_thread_id(ctx: &ToolCallContext<'_>) -> Option<String> {
    ctx.run_config()
        .value("__agent_tool_caller_thread_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// task_status
// ---------------------------------------------------------------------------

/// Query background task status and result.
///
/// Supports querying a single task by `task_id` or listing all tasks.
#[derive(Debug, Clone)]
pub struct TaskStatusTool {
    manager: Arc<BackgroundTaskManager>,
}

impl TaskStatusTool {
    pub fn new(manager: Arc<BackgroundTaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for TaskStatusTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            TASK_STATUS_TOOL_ID,
            "Task Status",
            "Check the status and result of background tasks. \
             Provide task_id to query a specific task, or omit to list all tasks.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to query. Omit to list all tasks."
                }
            }
        }))
    }

    async fn execute(
        &self,
        args: Value,
        ctx: &ToolCallContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let Some(thread_id) = owner_thread_id(ctx) else {
            return Ok(ToolResult::error(
                TASK_STATUS_TOOL_ID,
                "Missing caller thread context",
            ));
        };

        let task_id = args.get("task_id").and_then(Value::as_str);

        if let Some(task_id) = task_id {
            // Single task query.
            match self.manager.get(&thread_id, task_id).await {
                Some(summary) => Ok(ToolResult::success(
                    TASK_STATUS_TOOL_ID,
                    serde_json::to_value(&summary).unwrap_or(Value::Null),
                )),
                None => Ok(ToolResult::error(
                    TASK_STATUS_TOOL_ID,
                    format!("Unknown task_id: {task_id}"),
                )),
            }
        } else {
            // List all tasks.
            let tasks = self.manager.list(&thread_id, None).await;
            Ok(ToolResult::success(
                TASK_STATUS_TOOL_ID,
                json!({
                    "tasks": serde_json::to_value(&tasks).unwrap_or(Value::Null),
                    "total": tasks.len(),
                }),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// task_cancel
// ---------------------------------------------------------------------------

/// Cancel a running background task.
#[derive(Debug, Clone)]
pub struct TaskCancelTool {
    manager: Arc<BackgroundTaskManager>,
}

impl TaskCancelTool {
    pub fn new(manager: Arc<BackgroundTaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for TaskCancelTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            TASK_CANCEL_TOOL_ID,
            "Task Cancel",
            "Cancel a running background task by task_id.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to cancel"
                }
            },
            "required": ["task_id"]
        }))
    }

    async fn execute(
        &self,
        args: Value,
        ctx: &ToolCallContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let task_id = match required_string(&args, "task_id") {
            Ok(v) => v,
            Err(err) => return Ok(err),
        };

        let Some(thread_id) = owner_thread_id(ctx) else {
            return Ok(ToolResult::error(
                TASK_CANCEL_TOOL_ID,
                "Missing caller thread context",
            ));
        };

        match self.manager.cancel(&thread_id, task_id).await {
            Ok(()) => Ok(ToolResult::success(
                TASK_CANCEL_TOOL_ID,
                json!({
                    "task_id": task_id,
                    "cancelled": true,
                }),
            )),
            Err(e) => Ok(ToolResult::error(TASK_CANCEL_TOOL_ID, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::RunConfig;
    use tirea_contract::testing::TestFixture;

    const THREAD_ID_KEY: &str = "__agent_tool_caller_thread_id";

    fn fixture_with_thread(thread_id: &str) -> TestFixture {
        let mut fix = TestFixture::new();
        fix.run_config = {
            let mut rc = RunConfig::new();
            rc.set(THREAD_ID_KEY, thread_id.to_string()).unwrap();
            rc
        };
        fix
    }

    #[test]
    fn task_status_descriptor_has_optional_task_id() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskStatusTool::new(mgr);
        let desc = tool.descriptor();
        assert_eq!(desc.id, TASK_STATUS_TOOL_ID);
        // task_id is not in "required".
        let required = desc.parameters.get("required");
        assert!(required.is_none() || required.unwrap().as_array().unwrap().is_empty());
    }

    #[test]
    fn task_cancel_descriptor_requires_task_id() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskCancelTool::new(mgr);
        let desc = tool.descriptor();
        assert_eq!(desc.id, TASK_CANCEL_TOOL_ID);
        let required = desc.parameters["required"].as_array().unwrap();
        assert!(required.contains(&json!("task_id")));
    }

    // -----------------------------------------------------------------------
    // TaskStatusTool execute() tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn status_tool_missing_thread_context_returns_error() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskStatusTool::new(mgr);
        let fix = TestFixture::new(); // no __agent_tool_caller_thread_id
        let result = tool.execute(json!({}), &fix.ctx()).await.unwrap();
        assert!(!result.is_success());
        assert!(result.message.as_deref().unwrap_or("").contains("Missing caller thread context"));
    }

    #[tokio::test]
    async fn status_tool_list_all_when_no_tasks() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskStatusTool::new(mgr);
        let fix = fixture_with_thread("thread-1");
        let result = tool.execute(json!({}), &fix.ctx()).await.unwrap();
        assert!(result.is_success());
        let content: Value = result.data.clone();
        assert_eq!(content["total"], 0);
        assert!(content["tasks"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn status_tool_query_single_task() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tid = mgr
            .spawn("thread-1", "shell", "echo hi", |_cancel| async {
                super::super::types::TaskResult::Success(json!({"exit": 0}))
            })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let tool = TaskStatusTool::new(mgr);
        let fix = fixture_with_thread("thread-1");
        let result = tool
            .execute(json!({"task_id": tid}), &fix.ctx())
            .await
            .unwrap();
        assert!(result.is_success());
        let content: Value = result.data.clone();
        assert_eq!(content["status"], "completed");
        assert_eq!(content["result"]["exit"], 0);
    }

    #[tokio::test]
    async fn status_tool_query_unknown_task_returns_error() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskStatusTool::new(mgr);
        let fix = fixture_with_thread("thread-1");
        let result = tool
            .execute(json!({"task_id": "bogus"}), &fix.ctx())
            .await
            .unwrap();
        assert!(!result.is_success());
        assert!(result.message.as_deref().unwrap_or("").contains("Unknown task_id"));
    }

    #[tokio::test]
    async fn status_tool_list_shows_running_and_completed() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        // Running task.
        let _running = mgr
            .spawn("thread-1", "shell", "long", |cancel| async move {
                cancel.cancelled().await;
                super::super::types::TaskResult::Cancelled
            })
            .await;
        // Completed task.
        mgr.spawn("thread-1", "http", "fetch", |_| async {
            super::super::types::TaskResult::Success(Value::Null)
        })
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let tool = TaskStatusTool::new(mgr);
        let fix = fixture_with_thread("thread-1");
        let result = tool.execute(json!({}), &fix.ctx()).await.unwrap();
        assert!(result.is_success());
        let content: Value = result.data.clone();
        assert_eq!(content["total"], 2);
    }

    #[tokio::test]
    async fn status_tool_thread_isolation() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tid = mgr
            .spawn("thread-A", "shell", "private", |_| async {
                super::super::types::TaskResult::Success(Value::Null)
            })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let tool = TaskStatusTool::new(mgr);

        // Thread-B cannot see thread-A's task.
        let fix_b = fixture_with_thread("thread-B");
        let result = tool
            .execute(json!({"task_id": tid}), &fix_b.ctx())
            .await
            .unwrap();
        assert!(!result.is_success());

        // Thread-A can see it.
        let fix_a = fixture_with_thread("thread-A");
        let result = tool
            .execute(json!({"task_id": tid}), &fix_a.ctx())
            .await
            .unwrap();
        assert!(result.is_success());
    }

    // -----------------------------------------------------------------------
    // TaskCancelTool execute() tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn cancel_tool_missing_thread_context_returns_error() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskCancelTool::new(mgr);
        let fix = TestFixture::new();
        let result = tool
            .execute(json!({"task_id": "some"}), &fix.ctx())
            .await
            .unwrap();
        assert!(!result.is_success());
        assert!(result.message.as_deref().unwrap_or("").contains("Missing caller thread context"));
    }

    #[tokio::test]
    async fn cancel_tool_missing_task_id_param() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskCancelTool::new(mgr);
        let fix = fixture_with_thread("thread-1");
        let result = tool.execute(json!({}), &fix.ctx()).await.unwrap();
        assert!(!result.is_success());
        assert!(result.message.as_deref().unwrap_or("").contains("Missing required parameter"));
    }

    #[tokio::test]
    async fn cancel_tool_cancels_running_task() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tid = mgr
            .spawn("thread-1", "shell", "long", |cancel| async move {
                cancel.cancelled().await;
                super::super::types::TaskResult::Cancelled
            })
            .await;

        let tool = TaskCancelTool::new(mgr.clone());
        let fix = fixture_with_thread("thread-1");
        let result = tool
            .execute(json!({"task_id": tid}), &fix.ctx())
            .await
            .unwrap();
        assert!(result.is_success());
        let content: Value = result.data.clone();
        assert!(content["cancelled"].as_bool().unwrap());

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let summary = mgr.get("thread-1", &tid).await.unwrap();
        assert_eq!(summary.status, super::super::types::TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn cancel_tool_unknown_task_returns_error() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tool = TaskCancelTool::new(mgr);
        let fix = fixture_with_thread("thread-1");
        let result = tool
            .execute(json!({"task_id": "nope"}), &fix.ctx())
            .await
            .unwrap();
        assert!(!result.is_success());
        assert!(result.message.as_deref().unwrap_or("").contains("Unknown task_id"));
    }

    #[tokio::test]
    async fn cancel_tool_already_completed_returns_error() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tid = mgr
            .spawn("thread-1", "shell", "done", |_| async {
                super::super::types::TaskResult::Success(Value::Null)
            })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let tool = TaskCancelTool::new(mgr);
        let fix = fixture_with_thread("thread-1");
        let result = tool
            .execute(json!({"task_id": tid}), &fix.ctx())
            .await
            .unwrap();
        assert!(!result.is_success());
        assert!(result.message.as_deref().unwrap_or("").contains("not running"));
    }

    #[tokio::test]
    async fn cancel_tool_wrong_owner_rejected() {
        let mgr = Arc::new(BackgroundTaskManager::new());
        let tid = mgr
            .spawn("thread-1", "shell", "private", |cancel| async move {
                cancel.cancelled().await;
                super::super::types::TaskResult::Cancelled
            })
            .await;

        let tool = TaskCancelTool::new(mgr);
        let fix = fixture_with_thread("thread-2");
        let result = tool
            .execute(json!({"task_id": tid}), &fix.ctx())
            .await
            .unwrap();
        assert!(!result.is_success());
    }
}
