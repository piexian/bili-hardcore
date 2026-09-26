use crate::config::LlmConfig;
use crate::llm::prompt::{build_chat_prompt, build_split_prompt};
use crate::llm::protocol::{Protocol, endpoint};
use crate::llm::request::QuizRequest;
use crate::llm::shared::auth::apply_auth;
use crate::llm::shared::errors::extract_api_error;
use crate::llm::shared::http::{build_http_client, format_reqwest_error, safe_preview, send_error};
use crate::llm::shared::stream::{
    BodyStream, format_event_stream_error, open_body_stream, read_error_message,
};
use crate::llm::LlmChunk;
use futures::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const LABEL: &str = "Claude";
const MAX_JSON_RESPONSE_SIZE: usize = 4 * 1024 * 1024;

pub struct ClaudeClient {
    http: Client,
    endpoint: String,
    model: String,
    api_key: String,
    enable_thinking: bool,
    reasoning_effort: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ParsedPayload {
    reasoning: Option<String>,
    content: Option<String>,
}

/// Claude 的思考档位走 output_config.effort，支持 low/medium/high/xhigh/max，
/// 本项目的低/高/最大可以直接透传。关闭思考时降到 low：新一代模型是自适应思考，
/// 发 `thinking: {"type":"disabled"}` 会被拒（Opus 5.5 上任何 effort 都 400）。
fn output_effort(thinking: bool, saved_effort: &str) -> &'static str {
    if !thinking {
        return "low";
    }
    match saved_effort.trim().to_ascii_lowercase().as_str() {
        "low" => "low",
        "medium" => "medium",
        "max" => "max",
        "xhigh" => "xhigh",
        // 配置文件可能被手动编辑；未知值回退到兼容性最好的 high。
        _ => "high",
    }
}

/// max_tokens 是"思考 + 正文"的总量上限，档位越高思考越贵，
/// 沿用其他协议的 128 会让高 effort 在思考阶段就被截断、拿不到答案。
fn max_tokens(thinking: bool, saved_effort: &str) -> u64 {
    if !thinking {
        return 4096;
    }
    match saved_effort.trim().to_ascii_lowercase().as_str() {
        "low" => 8192,
        "medium" => 16384,
        "xhigh" | "max" => 65536,
        _ => 16384,
    }
}

fn build_request_body(
    model: &str,
    system: &str,
    question: &str,
    thinking: bool,
    saved_effort: &str,
) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "max_tokens": max_tokens(thinking, saved_effort),
        "stream": true,
        "system": system,
        "messages": [{"role": "user", "content": question}],
        "output_config": {"effort": output_effort(thinking, saved_effort)}
    })
}

fn non_empty(text: Option<&str>) -> Option<String> {
    text.filter(|value| !value.is_empty()).map(str::to_owned)
}

fn parse_event(value: &serde_json::Value) -> Result<ParsedPayload, String> {
    if let Some(message) = extract_api_error(value) {
        return Err(format!("{LABEL} API 返回错误: {message}"));
    }

    Ok(match value.get("type").and_then(serde_json::Value::as_str) {
        Some("content_block_delta") => match value
            .pointer("/delta/type")
            .and_then(serde_json::Value::as_str)
        {
            Some("text_delta") => ParsedPayload {
                content: non_empty(value.pointer("/delta/text").and_then(serde_json::Value::as_str)),
                ..Default::default()
            },
            Some("thinking_delta") => ParsedPayload {
                reasoning: non_empty(
                    value
                        .pointer("/delta/thinking")
                        .and_then(serde_json::Value::as_str),
                ),
                ..Default::default()
            },
            // signature_delta 等与答案无关。
            _ => ParsedPayload::default(),
        },
        // 首个分块可能自带完整文本。
        Some("content_block_start") => ParsedPayload {
            content: non_empty(
                value
                    .pointer("/content_block/text")
                    .and_then(serde_json::Value::as_str),
            ),
            reasoning: non_empty(
                value
                    .pointer("/content_block/thinking")
                    .and_then(serde_json::Value::as_str),
            ),
        },
        _ => ParsedPayload::default(),
    })
}

/// 非流式兜底：整条 message 的 content 块。
fn text_from_message(value: &serde_json::Value) -> Option<String> {
    let mut text = String::new();
    for block in value.get("content")?.as_array()? {
        if block.get("type").and_then(serde_json::Value::as_str) == Some("text")
            && let Some(value) = block.get("text").and_then(serde_json::Value::as_str)
        {
            text.push_str(value);
        }
    }
    (!text.is_empty()).then_some(text)
}

fn emit_payload(
    payload: ParsedPayload,
    full_content: &mut String,
    tx: &mpsc::UnboundedSender<LlmChunk>,
) {
    if let Some(reasoning) = payload.reasoning {
        let _ = tx.send(LlmChunk::Thinking(reasoning));
    }
    if let Some(content) = payload.content {
        full_content.push_str(&content);
        let _ = tx.send(LlmChunk::Content(content));
    }
}

impl ClaudeClient {
    pub fn new(config: &LlmConfig) -> Self {
        let reasoning_effort = if config.reasoning_effort.trim().is_empty() {
            "high".to_string()
        } else {
            config.reasoning_effort.clone()
        };
        Self {
            http: build_http_client(),
            endpoint: endpoint(Protocol::Claude, &config.base_url, &config.model),
            model: config.model.clone(),
            api_key: config.api_key.clone(),
            enable_thinking: config.enable_thinking,
            reasoning_effort,
        }
    }

    pub fn ask(
        &self,
        request: &QuizRequest,
        tx: mpsc::UnboundedSender<LlmChunk>,
        token: CancellationToken,
    ) {
        let question = build_chat_prompt(request.question, request.options);
        let (system, question) =
            build_split_prompt(request.categories, &question, self.enable_thinking);
        tracing::info!("{LABEL} prompt:\n{system}\n{question}");
        let body = build_request_body(
            &self.model,
            &system,
            &question,
            self.enable_thinking,
            &self.reasoning_effort,
        );

        let url = self.endpoint.clone();
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let request =
            apply_auth(http.post(&url).header("Content-Type", "application/json"), Protocol::Claude, &api_key)
                .json(&body);

        tokio::spawn(async move {
            if token.is_cancelled() {
                return;
            }
            let resp = tokio::select! {
                biased;
                _ = token.cancelled() => return,
                result = request.send() => match result {
                        Ok(response) => response,
                        Err(error) => {
                            send_error(
                                &tx,
                                &api_key,
                                format_reqwest_error(&format!("{LABEL} 请求失败"), &error, &api_key),
                            );
                            return;
                        }
                    }
            };

            if !resp.status().is_success() {
                if let Some(message) = read_error_message(resp, LABEL, &api_key, &token).await {
                    send_error(&tx, &api_key, message);
                }
                return;
            }

            let body = match open_body_stream(resp, LABEL, &api_key, MAX_JSON_RESPONSE_SIZE, &token).await {
                Ok(Some(body)) => body,
                Ok(None) => return,
                Err(error) => {
                    send_error(&tx, &api_key, error);
                    return;
                }
            };

            let mut full_content = String::new();
            match body {
                BodyStream::Json(bytes) => {
                    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
                        Ok(value) => value,
                        Err(error) => {
                            let preview =
                                safe_preview(&String::from_utf8_lossy(&bytes), &api_key, 200);
                            send_error(
                                &tx,
                                &api_key,
                                format!("解析 {LABEL} JSON 响应失败: {error}; 响应摘要: {preview}"),
                            );
                            return;
                        }
                    };
                    if let Some(message) = extract_api_error(&value) {
                        send_error(&tx, &api_key, format!("{LABEL} API 返回错误: {message}"));
                        return;
                    }
                    match text_from_message(&value) {
                        Some(text) => {
                            full_content = text;
                            let _ = tx.send(LlmChunk::Content(full_content.clone()));
                        }
                        None => {
                            send_error(&tx, &api_key, format!("{LABEL} JSON 响应不包含文本内容"));
                            return;
                        }
                    }
                }
                BodyStream::Sse(mut events) => loop {
                    let event = tokio::select! {
                        biased;
                        _ = token.cancelled() => return,
                        event = events.next() => event,
                    };
                    let Some(event) = event else {
                        break;
                    };
                    let event = match event {
                        Ok(event) => event,
                        Err(error) => {
                            send_error(
                                &tx,
                                &api_key,
                                format_event_stream_error(&error, LABEL, &api_key),
                            );
                            return;
                        }
                    };
                    let value: serde_json::Value = match serde_json::from_str(&event.data) {
                        Ok(value) => value,
                        Err(error) => {
                            send_error(
                                &tx,
                                &api_key,
                                format!("解析 {LABEL} SSE 事件 JSON 失败: {error}"),
                            );
                            return;
                        }
                    };
                    match parse_event(&value) {
                        Ok(payload) => emit_payload(payload, &mut full_content, &tx),
                        Err(error) => {
                            send_error(&tx, &api_key, error);
                            return;
                        }
                    }
                },
            }

            if token.is_cancelled() {
                return;
            }
            if full_content.trim().is_empty() {
                send_error(&tx, &api_key, format!("{LABEL} 响应结束但未包含可用正文"));
            } else {
                let _ = tx.send(LlmChunk::Done(full_content));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_uses_messages_system_and_output_config() {
        let body = build_request_body("claude-sonnet-5", "人设", "题目", true, "high");
        assert_eq!(body["model"], "claude-sonnet-5");
        assert_eq!(body["system"], "人设");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "题目");
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["stream"], true);
        assert!(body["max_tokens"].as_u64().unwrap() >= 16384);
        // 思考由 effort 表达，不发 thinking 开关
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn effort_levels_pass_through_and_never_disable_thinking() {
        for (saved, expected) in [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("max", "max"),
            ("xhigh", "xhigh"),
            ("手改的值", "high"),
        ] {
            assert_eq!(output_effort(true, saved), expected);
        }
        // 自适应思考模型拒绝 thinking:disabled，关思考只能降到最低 effort。
        assert_eq!(output_effort(false, "max"), "low");
    }

    #[test]
    fn max_tokens_grows_with_effort() {
        assert_eq!(max_tokens(false, "max"), 4096);
        assert!(max_tokens(true, "low") < max_tokens(true, "high"));
        assert!(max_tokens(true, "high") < max_tokens(true, "max"));
        assert_eq!(max_tokens(true, "xhigh"), 65536);
    }

    #[test]
    fn parses_text_and_thinking_block_deltas() {
        let text = serde_json::json!({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "2"}
        });
        assert_eq!(
            parse_event(&text).unwrap(),
            ParsedPayload {
                reasoning: None,
                content: Some("2".to_string())
            }
        );

        let thinking = serde_json::json!({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "推理"}
        });
        assert_eq!(
            parse_event(&thinking).unwrap(),
            ParsedPayload {
                reasoning: Some("推理".to_string()),
                content: None
            }
        );

        let signature = serde_json::json!({
            "type": "content_block_delta",
            "delta": {"type": "signature_delta", "signature": "abc"}
        });
        assert_eq!(parse_event(&signature).unwrap(), ParsedPayload::default());

        let ping = serde_json::json!({"type": "ping"});
        assert_eq!(parse_event(&ping).unwrap(), ParsedPayload::default());

        let start = serde_json::json!({
            "type": "content_block_start",
            "content_block": {"type": "text", "text": ""}
        });
        assert_eq!(parse_event(&start).unwrap(), ParsedPayload::default());
    }

    #[test]
    fn error_events_and_messages_are_reported() {
        let error_event = serde_json::json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}
        });
        assert!(parse_event(&error_event).unwrap_err().contains("Overloaded"));

        let message = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "想"},
                {"type": "text", "text": "3"}
            ]
        });
        assert_eq!(text_from_message(&message).as_deref(), Some("3"));

        let empty = serde_json::json!({"content": [{"type": "text", "text": ""}]});
        assert_eq!(text_from_message(&empty), None);
    }
}
