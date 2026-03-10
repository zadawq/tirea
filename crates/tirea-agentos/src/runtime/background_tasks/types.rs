//! Core types for the background task system.
//!
//! These types model the lifecycle of background tasks spawned by tools.
//! Tasks are thread-scoped and outlive individual runs.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tirea_state::State;

/// Unique identifier for a background task.
pub type TaskId = String;

/// Status of a background task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }
}

/// Result produced by a background task on completion.
#[derive(Debug, Clone)]
pub enum TaskResult {
    /// Task completed successfully with a result value.
    Success(Value),
    /// Task failed with an error message.
    Failed(String),
    /// Task was cancelled.
    Cancelled,
}

impl TaskResult {
    pub fn status(&self) -> TaskStatus {
        match self {
            Self::Success(_) => TaskStatus::Completed,
            Self::Failed(_) => TaskStatus::Failed,
            Self::Cancelled => TaskStatus::Cancelled,
        }
    }
}

/// Summary of a background task visible to tools and plugins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSummary {
    pub task_id: TaskId,
    pub task_type: String,
    pub description: String,
    pub status: TaskStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<u64>,
}

/// Lightweight persisted metadata for a background task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTask {
    pub task_type: String,
    pub description: String,
    pub status: TaskStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<u64>,
}

/// Persisted background task state at `state["background_tasks"]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, State)]
#[tirea(path = "background_tasks", action = "BackgroundTaskAction", scope = "thread")]
pub struct BackgroundTaskState {
    /// Background tasks keyed by `task_id`.
    #[tirea(default = "HashMap::new()")]
    pub tasks: HashMap<TaskId, BackgroundTask>,
}

/// Reducer actions for `BackgroundTaskState`.
#[derive(Serialize, Deserialize)]
pub enum BackgroundTaskAction {
    /// Register a new running task.
    Register {
        task_id: TaskId,
        task: BackgroundTask,
    },
    /// Update status of an existing task.
    SetStatus {
        task_id: TaskId,
        status: TaskStatus,
        error: Option<String>,
        completed_at_ms: Option<u64>,
    },
}

impl BackgroundTaskState {
    fn reduce(&mut self, action: BackgroundTaskAction) {
        match action {
            BackgroundTaskAction::Register { task_id, task } => {
                self.tasks.insert(task_id, task);
            }
            BackgroundTaskAction::SetStatus {
                task_id,
                status,
                error,
                completed_at_ms,
            } => {
                if let Some(task) = self.tasks.get_mut(&task_id) {
                    task.status = status;
                    task.error = error;
                    if let Some(ts) = completed_at_ms {
                        task.completed_at_ms = Some(ts);
                    }
                }
            }
        }
    }
}
