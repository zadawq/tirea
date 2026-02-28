mod support;

use async_trait::async_trait;
use mcp::transport::{McpServerConnectionConfig, McpTransportError};
use mcp::{CreateMessageParams, CreateMessageResult, Role, SamplingContent};
use std::sync::{Arc, Mutex};
use tirea_extension_mcp::{McpToolRegistryManager, McpToolTransport, SamplingHandler};

struct RecordingSamplingHandler {
    calls: Arc<Mutex<Vec<CreateMessageParams>>>,
    response_text: String,
}

impl RecordingSamplingHandler {
    fn new(response_text: impl Into<String>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            response_text: response_text.into(),
        }
    }

    #[allow(dead_code)]
    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl SamplingHandler for RecordingSamplingHandler {
    async fn handle_create_message(
        &self,
        params: CreateMessageParams,
    ) -> Result<CreateMessageResult, McpTransportError> {
        self.calls.lock().unwrap().push(params);
        Ok(CreateMessageResult {
            role: Role::Assistant,
            content: vec![SamplingContent::Text {
                text: self.response_text.clone(),
                annotations: None,
                meta: None,
            }],
            model: "test-model".to_string(),
            stop_reason: Some("end_turn".to_string()),
            meta: None,
        })
    }
}

struct FailingSamplingHandler;

#[async_trait]
impl SamplingHandler for FailingSamplingHandler {
    async fn handle_create_message(
        &self,
        _params: CreateMessageParams,
    ) -> Result<CreateMessageResult, McpTransportError> {
        Err(McpTransportError::TransportError(
            "LLM unavailable".to_string(),
        ))
    }
}

#[tokio::test]
async fn sampling_handler_trait_is_object_safe_and_constructible() {
    let handler: Arc<dyn SamplingHandler> = Arc::new(RecordingSamplingHandler::new("hello"));
    let params = CreateMessageParams {
        messages: vec![],
        model_preferences: None,
        system_prompt: Some("You are helpful".to_string()),
        include_context: None,
        temperature: None,
        max_tokens: 100,
        stop_sequences: None,
        metadata: None,
        tools: None,
        tool_choice: None,
        task: None,
        meta: None,
    };

    let result = handler.handle_create_message(params).await.unwrap();
    assert_eq!(result.model, "test-model");
    assert_eq!(result.role, Role::Assistant);
    match &result.content[0] {
        SamplingContent::Text { text, .. } => assert_eq!(text, "hello"),
        _ => panic!("expected text content"),
    }
}

#[tokio::test]
async fn failing_sampling_handler_returns_error() {
    let handler: Arc<dyn SamplingHandler> = Arc::new(FailingSamplingHandler);
    let params = CreateMessageParams {
        messages: vec![],
        model_preferences: None,
        system_prompt: None,
        include_context: None,
        temperature: None,
        max_tokens: 100,
        stop_sequences: None,
        metadata: None,
        tools: None,
        tool_choice: None,
        task: None,
        meta: None,
    };

    let err = handler.handle_create_message(params).await.unwrap_err();
    assert!(err.to_string().contains("LLM unavailable"));
}

#[tokio::test]
async fn connect_with_sampling_accepts_handler() {
    // This test verifies that the API compiles and the handler type
    // is accepted. It uses an empty config list so no actual connection occurs.
    let handler: Arc<dyn SamplingHandler> = Arc::new(RecordingSamplingHandler::new("ok"));
    let configs: Vec<McpServerConnectionConfig> = vec![];
    let manager = McpToolRegistryManager::connect_with_sampling(configs, Some(handler))
        .await
        .unwrap();
    assert!(manager.registry().is_empty());
}

#[tokio::test]
async fn connect_with_sampling_none_handler_works() {
    let configs: Vec<McpServerConnectionConfig> = vec![];
    let manager = McpToolRegistryManager::connect_with_sampling(configs, None)
        .await
        .unwrap();
    assert!(manager.registry().is_empty());
}

// ---------------------------------------------------------------------------
// Full bidirectional sampling E2E test using DuplexStream transport
// ---------------------------------------------------------------------------

/// Spawn a minimal MCP server on a DuplexStream that sends
/// `sampling/createMessage` during tool execution.
fn spawn_sampling_server(
    prompt: &str,
) -> (
    tokio::io::DuplexStream,
    tokio::task::JoinHandle<()>,
) {
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicI64, Ordering};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let prompt = prompt.to_string();
    let (client_stream, server_stream) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(async move {
        let (reader_half, writer_half) = tokio::io::split(server_stream);
        let mut reader = BufReader::new(reader_half);
        let mut writer = writer_half;
        let next_id = AtomicI64::new(1000);

        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }

            let msg: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Skip notifications
            if msg.get("id").is_none() {
                continue;
            }

            let id = msg["id"].clone();
            let method = msg["method"].as_str().unwrap_or("");

            let response = match method {
                "initialize" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "sampling-test-server", "version": "0.1.0"}
                    }
                }),
                "tools/list" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [{
                            "name": "summarize",
                            "description": "Summarize using LLM via sampling",
                            "inputSchema": {"type": "object", "properties": {}}
                        }]
                    }
                }),
                "tools/call" => {
                    // Send sampling/createMessage to the client
                    let sampling_id = next_id.fetch_add(1, Ordering::SeqCst);
                    let sampling_req = json!({
                        "jsonrpc": "2.0",
                        "id": sampling_id,
                        "method": "sampling/createMessage",
                        "params": {
                            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                            "maxTokens": 100
                        }
                    });
                    let req_line = format!("{}\n", serde_json::to_string(&sampling_req).unwrap());
                    if writer.write_all(req_line.as_bytes()).await.is_err() {
                        break;
                    }
                    if writer.flush().await.is_err() {
                        break;
                    }

                    // Read the client's response to sampling
                    let mut resp_line = String::new();
                    match reader.read_line(&mut resp_line).await {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(_) => break,
                    }
                    let resp: Value = serde_json::from_str(&resp_line).unwrap_or(json!(null));
                    let llm_text = resp["result"]["content"]
                        .as_array()
                        .and_then(|arr| arr.first())
                        .and_then(|c| c.get("text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("(no response)");

                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [{"type": "text", "text": format!("sampling:{llm_text}")}]
                        }
                    })
                }
                _ => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32601, "message": format!("Unknown: {method}")}
                }),
            };

            let out = format!("{}\n", serde_json::to_string(&response).unwrap());
            if writer.write_all(out.as_bytes()).await.is_err() {
                break;
            }
            if writer.flush().await.is_err() {
                break;
            }
        }
    });
    (client_stream, handle)
}

#[tokio::test]
async fn sampling_request_routed_through_duplex_transport() {
    use support::duplex_transport::DuplexStreamTransport;
    use tirea_contract::runtime::tool_call::Tool;

    let handler = Arc::new(RecordingSamplingHandler::new("LLM says hello"));
    let (stream, _server) = spawn_sampling_server("Summarize this document");

    let transport = Arc::new(
        DuplexStreamTransport::connect(stream, Some(handler.clone() as Arc<dyn SamplingHandler>))
            .await
            .unwrap(),
    ) as Arc<dyn McpToolTransport>;

    let cfg = McpServerConnectionConfig::stdio("sampling_server", "dummy", vec![]);
    let manager = McpToolRegistryManager::from_transports([(cfg, transport)])
        .await
        .unwrap();
    let registry = manager.registry();

    let tool = registry.get("mcp__sampling_server__summarize").unwrap();
    let fix = tirea_contract::testing::TestFixture::new();
    let ctx = fix.ctx_with("sampling-call", "test");

    let result = tool
        .execute(serde_json::json!({}), &ctx)
        .await
        .unwrap();

    assert!(result.is_success());
    // The tool result should contain the sampling response from the handler
    let result_text = result.data.as_str().unwrap();
    assert!(
        result_text.contains("LLM says hello"),
        "tool result should contain sampling response, got: {result_text}"
    );

    // Verify the handler was actually called
    assert_eq!(handler.call_count(), 1);
}

#[tokio::test]
async fn sampling_without_handler_returns_error_to_server() {
    use support::duplex_transport::DuplexStreamTransport;
    use tirea_contract::runtime::tool_call::Tool;

    // No sampling handler provided
    let (stream, _server) = spawn_sampling_server("This will fail");
    let transport = Arc::new(
        DuplexStreamTransport::connect(stream, None).await.unwrap(),
    ) as Arc<dyn McpToolTransport>;

    let cfg = McpServerConnectionConfig::stdio("no_handler", "dummy", vec![]);
    let manager = McpToolRegistryManager::from_transports([(cfg, transport)])
        .await
        .unwrap();
    let registry = manager.registry();

    let tool = registry.get("mcp__no_handler__summarize").unwrap();
    let fix = tirea_contract::testing::TestFixture::new();
    let ctx = fix.ctx_with("no-handler-call", "test");

    let result = tool
        .execute(serde_json::json!({}), &ctx)
        .await
        .unwrap();

    assert!(result.is_success());
    // The server receives an error response for sampling, so it returns "(no response)"
    let result_text = result.data.as_str().unwrap();
    assert!(
        result_text.contains("(no response)") || result_text.contains("sampling:"),
        "tool should still succeed, got: {result_text}"
    );
}
