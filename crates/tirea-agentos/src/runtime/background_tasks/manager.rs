//! In-memory background task handle table and spawner.
//!
//! Manages the lifecycle of background tasks: spawn, track, cancel, query.
//! Epoch-based stale-completion guards prevent double updates.

use super::types::*;
use crate::loop_runtime::loop_runner::RunCancellationToken;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

fn gen_task_id() -> TaskId {
    format!("bg_{}", uuid::Uuid::now_v7().simple())
}

/// In-memory runtime handle for a single background task.
#[derive(Debug)]
struct TaskHandle {
    epoch: u64,
    owner_thread_id: String,
    task_type: String,
    description: String,
    status: TaskStatus,
    error: Option<String>,
    result: Option<Value>,
    cancel_token: RunCancellationToken,
    cancellation_requested: bool,
    created_at_ms: u64,
    completed_at_ms: Option<u64>,
}

/// Callback invoked after a background task completes.
///
/// Implementations deliver the completion notification (e.g. via mailbox).
/// The trait is object-safe so different notification strategies can be plugged in.
#[async_trait::async_trait]
pub trait TaskCompletionNotifier: Send + Sync {
    async fn notify(
        &self,
        thread_id: &str,
        task_id: &str,
        task_type: &str,
        description: &str,
        result: &TaskResult,
    );
}

/// No-op notifier used when no notification channel is configured.
pub(super) struct NoopNotifier;

#[async_trait::async_trait]
impl TaskCompletionNotifier for NoopNotifier {
    async fn notify(&self, _: &str, _: &str, _: &str, _: &str, _: &TaskResult) {}
}

/// Thread-scoped background task manager.
///
/// Manages the full lifecycle: spawn → track → cancel → query.
/// Tasks outlive individual runs but are scoped to a thread.
#[derive(Clone)]
pub struct BackgroundTaskManager {
    handles: Arc<Mutex<HashMap<TaskId, TaskHandle>>>,
    notifier: Arc<dyn TaskCompletionNotifier>,
}

impl std::fmt::Debug for BackgroundTaskManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundTaskManager")
            .field("task_count", &"<locked>")
            .finish()
    }
}

impl Default for BackgroundTaskManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundTaskManager {
    pub fn new() -> Self {
        Self {
            handles: Arc::new(Mutex::new(HashMap::new())),
            notifier: Arc::new(NoopNotifier),
        }
    }

    pub fn with_notifier(notifier: Arc<dyn TaskCompletionNotifier>) -> Self {
        Self {
            handles: Arc::new(Mutex::new(HashMap::new())),
            notifier,
        }
    }

    /// Spawn a background task with the given closure.
    ///
    /// Returns the generated `TaskId` immediately. The closure receives a
    /// `CancellationToken` for cooperative cancellation.
    pub async fn spawn<F, Fut>(
        &self,
        owner_thread_id: &str,
        task_type: &str,
        description: &str,
        task: F,
    ) -> TaskId
    where
        F: FnOnce(RunCancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = TaskResult> + Send,
    {
        let task_id = gen_task_id();
        let cancel_token = RunCancellationToken::new();
        let now = now_ms();

        let epoch = {
            let mut handles = self.handles.lock().await;
            let epoch = handles
                .get(&task_id)
                .map(|h| h.epoch + 1)
                .unwrap_or(1);
            handles.insert(
                task_id.clone(),
                TaskHandle {
                    epoch,
                    owner_thread_id: owner_thread_id.to_string(),
                    task_type: task_type.to_string(),
                    description: description.to_string(),
                    status: TaskStatus::Running,
                    error: None,
                    result: None,
                    cancel_token: cancel_token.clone(),
                    cancellation_requested: false,
                    created_at_ms: now,
                    completed_at_ms: None,
                },
            );
            epoch
        };

        let handles = self.handles.clone();
        let notifier = self.notifier.clone();
        let tid = task_id.clone();
        let ttype = task_type.to_string();
        let desc = description.to_string();
        let thread_id = owner_thread_id.to_string();

        tokio::spawn(async move {
            let result = task(cancel_token).await;

            // Update in-memory handle.
            let completed_at = now_ms();
            {
                let mut map = handles.lock().await;
                if let Some(handle) = map.get_mut(&tid) {
                    if handle.epoch != epoch {
                        return; // Stale completion.
                    }
                    let status = if handle.cancellation_requested {
                        TaskStatus::Cancelled
                    } else {
                        result.status()
                    };
                    handle.status = status;
                    handle.error = match &result {
                        TaskResult::Failed(e) => Some(e.clone()),
                        _ => None,
                    };
                    handle.result = match &result {
                        TaskResult::Success(v) => Some(v.clone()),
                        _ => None,
                    };
                    handle.completed_at_ms = Some(completed_at);
                }
            }

            // Deliver completion notification.
            notifier
                .notify(&thread_id, &tid, &ttype, &desc, &result)
                .await;
        });

        task_id
    }

    /// Cancel a task owned by the given thread. Returns `true` if cancelled.
    pub async fn cancel(&self, owner_thread_id: &str, task_id: &str) -> Result<(), String> {
        let mut handles = self.handles.lock().await;
        let Some(handle) = handles.get_mut(task_id) else {
            return Err(format!("Unknown task_id: {task_id}"));
        };
        if handle.owner_thread_id != owner_thread_id {
            return Err(format!("Unknown task_id: {task_id}"));
        }
        if handle.status != TaskStatus::Running {
            return Err(format!(
                "Task '{task_id}' is not running (current status: {})",
                handle.status.as_str()
            ));
        }
        handle.cancellation_requested = true;
        handle.cancel_token.cancel();
        Ok(())
    }

    /// Get a summary of a single task.
    pub async fn get(
        &self,
        owner_thread_id: &str,
        task_id: &str,
    ) -> Option<TaskSummary> {
        let handles = self.handles.lock().await;
        let handle = handles.get(task_id)?;
        if handle.owner_thread_id != owner_thread_id {
            return None;
        }
        Some(summary_from_handle(task_id, handle))
    }

    /// List all tasks for a thread, optionally filtered by status.
    pub async fn list(
        &self,
        owner_thread_id: &str,
        status_filter: Option<TaskStatus>,
    ) -> Vec<TaskSummary> {
        let handles = self.handles.lock().await;
        let mut out: Vec<TaskSummary> = handles
            .iter()
            .filter(|(_, h)| h.owner_thread_id == owner_thread_id)
            .filter(|(_, h)| status_filter.is_none_or(|s| s == h.status))
            .map(|(id, h)| summary_from_handle(id, h))
            .collect();
        out.sort_by(|a, b| a.created_at_ms.cmp(&b.created_at_ms));
        out
    }

    /// Check if there are running tasks for a thread.
    pub async fn has_running_tasks(&self, owner_thread_id: &str) -> bool {
        let handles = self.handles.lock().await;
        handles.values().any(|h| {
            h.owner_thread_id == owner_thread_id && h.status == TaskStatus::Running
        })
    }

    /// Remove completed/terminal task entries from the in-memory table.
    pub async fn gc_terminal(&self, owner_thread_id: &str) -> usize {
        let mut handles = self.handles.lock().await;
        let before = handles.len();
        handles.retain(|_, h| {
            h.owner_thread_id != owner_thread_id || !h.status.is_terminal()
        });
        before - handles.len()
    }
}

fn summary_from_handle(task_id: &str, handle: &TaskHandle) -> TaskSummary {
    TaskSummary {
        task_id: task_id.to_string(),
        task_type: handle.task_type.clone(),
        description: handle.description.clone(),
        status: handle.status,
        error: handle.error.clone(),
        result: handle.result.clone(),
        created_at_ms: handle.created_at_ms,
        completed_at_ms: handle.completed_at_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test notifier that counts invocations.
    struct CountingNotifier {
        count: AtomicUsize,
    }

    impl CountingNotifier {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                count: AtomicUsize::new(0),
            })
        }

        fn count(&self) -> usize {
            self.count.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl TaskCompletionNotifier for CountingNotifier {
        async fn notify(&self, _: &str, _: &str, _: &str, _: &str, _: &TaskResult) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn spawn_and_complete_success() {
        let notifier = CountingNotifier::new();
        let mgr = BackgroundTaskManager::with_notifier(notifier.clone());
        let tid = mgr
            .spawn("thread-1", "shell", "echo hello", |_cancel| async {
                TaskResult::Success(serde_json::json!({ "exit_code": 0 }))
            })
            .await;

        // Wait for the spawned task to complete.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let summary = mgr.get("thread-1", &tid).await.expect("task should exist");
        assert_eq!(summary.status, TaskStatus::Completed);
        assert!(summary.result.is_some());
        assert!(summary.error.is_none());
        assert!(summary.completed_at_ms.is_some());
        assert_eq!(notifier.count(), 1);
    }

    #[tokio::test]
    async fn spawn_and_complete_failure() {
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "bad cmd", |_cancel| async {
                TaskResult::Failed("command not found".into())
            })
            .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let summary = mgr.get("thread-1", &tid).await.unwrap();
        assert_eq!(summary.status, TaskStatus::Failed);
        assert_eq!(summary.error.as_deref(), Some("command not found"));
    }

    #[tokio::test]
    async fn cancel_running_task() {
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "long running", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;

        // Task should be running.
        let summary = mgr.get("thread-1", &tid).await.unwrap();
        assert_eq!(summary.status, TaskStatus::Running);

        // Cancel it.
        mgr.cancel("thread-1", &tid).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let summary = mgr.get("thread-1", &tid).await.unwrap();
        assert_eq!(summary.status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn cancel_wrong_owner_rejected() {
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "task", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;

        let result = mgr.cancel("thread-other", &tid).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn list_filters_by_owner_and_status() {
        let mgr = BackgroundTaskManager::new();

        // Running task for thread-1.
        let _t1 = mgr
            .spawn("thread-1", "shell", "task-a", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;

        // Completed task for thread-1.
        mgr.spawn("thread-1", "http", "task-b", |_cancel| async {
            TaskResult::Success(Value::Null)
        })
        .await;

        // Task for thread-2 (should not appear).
        mgr.spawn("thread-2", "shell", "task-c", |cancel| async move {
            cancel.cancelled().await;
            TaskResult::Cancelled
        })
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let all = mgr.list("thread-1", None).await;
        assert_eq!(all.len(), 2);

        let running = mgr.list("thread-1", Some(TaskStatus::Running)).await;
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].task_type, "shell");

        let completed = mgr.list("thread-1", Some(TaskStatus::Completed)).await;
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].task_type, "http");
    }

    #[tokio::test]
    async fn has_running_tasks_reflects_state() {
        let mgr = BackgroundTaskManager::new();

        assert!(!mgr.has_running_tasks("thread-1").await);

        let tid = mgr
            .spawn("thread-1", "shell", "task", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;

        assert!(mgr.has_running_tasks("thread-1").await);

        mgr.cancel("thread-1", &tid).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        assert!(!mgr.has_running_tasks("thread-1").await);
    }

    #[tokio::test]
    async fn gc_terminal_removes_completed_tasks() {
        let mgr = BackgroundTaskManager::new();

        mgr.spawn("thread-1", "shell", "done", |_| async {
            TaskResult::Success(Value::Null)
        })
        .await;

        let _running = mgr
            .spawn("thread-1", "shell", "still going", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let removed = mgr.gc_terminal("thread-1").await;
        assert_eq!(removed, 1);

        let all = mgr.list("thread-1", None).await;
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].status, TaskStatus::Running);
    }

    #[tokio::test]
    async fn get_returns_none_for_wrong_owner() {
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "task", |_| async {
                TaskResult::Success(Value::Null)
            })
            .await;

        assert!(mgr.get("thread-other", &tid).await.is_none());
        assert!(mgr.get("thread-1", &tid).await.is_some());
    }

    #[tokio::test]
    async fn cancel_already_completed_returns_error() {
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "fast", |_| async {
                TaskResult::Success(Value::Null)
            })
            .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let err = mgr.cancel("thread-1", &tid).await.unwrap_err();
        assert!(err.contains("not running"));
    }

    // -----------------------------------------------------------------------
    // Concurrency & edge-case tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn many_concurrent_spawns_all_tracked() {
        let mgr = BackgroundTaskManager::new();
        let mut ids = Vec::new();

        for i in 0..20 {
            let desc = format!("task-{i}");
            let tid = mgr
                .spawn("thread-1", "shell", &desc, |cancel| async move {
                    cancel.cancelled().await;
                    TaskResult::Cancelled
                })
                .await;
            ids.push(tid);
        }

        let all = mgr.list("thread-1", None).await;
        assert_eq!(all.len(), 20);

        // All tasks should be running.
        let running = mgr.list("thread-1", Some(TaskStatus::Running)).await;
        assert_eq!(running.len(), 20);

        // Cancel all.
        for tid in &ids {
            mgr.cancel("thread-1", tid).await.unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let cancelled = mgr.list("thread-1", Some(TaskStatus::Cancelled)).await;
        assert_eq!(cancelled.len(), 20);
    }

    #[tokio::test]
    async fn concurrent_cancel_and_complete_race() {
        // Task completes very quickly; cancel may arrive after completion.
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "fast", |_| async {
                TaskResult::Success(Value::Null)
            })
            .await;

        // Race: try to cancel immediately (may or may not succeed).
        let cancel_result = mgr.cancel("thread-1", &tid).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let summary = mgr.get("thread-1", &tid).await.unwrap();
        // Either cancelled (if we beat the completion) or completed (if not).
        assert!(
            summary.status == TaskStatus::Cancelled || summary.status == TaskStatus::Completed,
            "unexpected status: {:?}",
            summary.status
        );

        // If cancel succeeded, status should be Cancelled.
        if cancel_result.is_ok() {
            assert_eq!(summary.status, TaskStatus::Cancelled);
        }
    }

    #[tokio::test]
    async fn task_with_tokio_select_respects_cancellation() {
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "select-based", |cancel| async move {
                tokio::select! {
                    _ = cancel.cancelled() => TaskResult::Cancelled,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {
                        TaskResult::Success(Value::Null)
                    }
                }
            })
            .await;

        assert!(mgr.has_running_tasks("thread-1").await);

        mgr.cancel("thread-1", &tid).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let summary = mgr.get("thread-1", &tid).await.unwrap();
        assert_eq!(summary.status, TaskStatus::Cancelled);
        assert!(!mgr.has_running_tasks("thread-1").await);
    }

    #[tokio::test]
    async fn task_failure_with_panic_safety() {
        // A task that returns a failure result (not a panic).
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "http", "timeout", |_| async {
                TaskResult::Failed("connection timed out".into())
            })
            .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let summary = mgr.get("thread-1", &tid).await.unwrap();
        assert_eq!(summary.status, TaskStatus::Failed);
        assert_eq!(summary.error.as_deref(), Some("connection timed out"));
        assert!(summary.completed_at_ms.is_some());
    }

    #[tokio::test]
    async fn gc_only_affects_specified_thread() {
        let mgr = BackgroundTaskManager::new();

        // Completed task on thread-1.
        mgr.spawn("thread-1", "shell", "done-1", |_| async {
            TaskResult::Success(Value::Null)
        })
        .await;
        // Completed task on thread-2.
        mgr.spawn("thread-2", "shell", "done-2", |_| async {
            TaskResult::Success(Value::Null)
        })
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // GC only thread-1.
        let removed = mgr.gc_terminal("thread-1").await;
        assert_eq!(removed, 1);

        // Thread-2's task should still be there.
        let t2_tasks = mgr.list("thread-2", None).await;
        assert_eq!(t2_tasks.len(), 1);
    }

    #[tokio::test]
    async fn notifier_called_for_every_completion() {
        let notifier = CountingNotifier::new();
        let mgr = BackgroundTaskManager::with_notifier(notifier.clone());

        for _ in 0..5 {
            mgr.spawn("thread-1", "shell", "task", |_| async {
                TaskResult::Success(Value::Null)
            })
            .await;
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(notifier.count(), 5);
    }

    #[tokio::test]
    async fn notifier_called_on_cancel() {
        let notifier = CountingNotifier::new();
        let mgr = BackgroundTaskManager::with_notifier(notifier.clone());

        let tid = mgr
            .spawn("thread-1", "shell", "long", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;

        mgr.cancel("thread-1", &tid).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        assert_eq!(notifier.count(), 1);
    }

    #[tokio::test]
    async fn task_summary_has_timing_info() {
        let mgr = BackgroundTaskManager::new();
        let tid = mgr
            .spawn("thread-1", "shell", "timed", |_| async {
                TaskResult::Success(Value::Null)
            })
            .await;

        // While running, should have created_at but no completed_at.
        let running = mgr.get("thread-1", &tid).await.unwrap();
        assert!(running.created_at_ms > 0);
        // Note: task may have already completed since it's instant.

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let completed = mgr.get("thread-1", &tid).await.unwrap();
        assert!(completed.created_at_ms > 0);
        assert!(completed.completed_at_ms.is_some());
        assert!(completed.completed_at_ms.unwrap() >= completed.created_at_ms);
    }

    #[tokio::test]
    async fn list_returns_sorted_by_creation_time() {
        let mgr = BackgroundTaskManager::new();

        let t1 = mgr
            .spawn("thread-1", "shell", "first", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;
        // Small delays to guarantee distinct timestamps.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let t2 = mgr
            .spawn("thread-1", "shell", "second", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let t3 = mgr
            .spawn("thread-1", "shell", "third", |cancel| async move {
                cancel.cancelled().await;
                TaskResult::Cancelled
            })
            .await;

        let tasks = mgr.list("thread-1", None).await;
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].task_id, t1);
        assert_eq!(tasks[1].task_id, t2);
        assert_eq!(tasks[2].task_id, t3);
    }

    #[tokio::test]
    async fn default_impl_is_noop_notifier() {
        // BackgroundTaskManager::new() uses NoopNotifier.
        let mgr = BackgroundTaskManager::new();
        mgr.spawn("thread-1", "shell", "task", |_| async {
            TaskResult::Success(Value::Null)
        })
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // No panic, no error — just verify it works.
        let tasks = mgr.list("thread-1", None).await;
        assert_eq!(tasks.len(), 1);
    }
}
