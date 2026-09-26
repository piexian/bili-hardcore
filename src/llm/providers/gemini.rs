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

const LABEL: &str = "Gemini";
const MAX_JSON_RESPONSE_SIZE: usize = 4 * 1024 * 1024;
const MAX_OUTPUT_TOKENS: u32 = 128;

pub struct GeminiClient {
    http: Client,
    endpoint: String,
    api_key: String,
    enable_thinking: bool,
    reasoning_effort: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ParsedPayload {
    reasoning: Option<String>,
    content: Option<String>,
}

/// Gemini 3 用 thinkingLevel 表达思考档位，不再用 2.5 系的 thinkingBudget；
/// 各模型支持的档位不同（Pro 只有 low/high），没有"完全关闭"档，
/// 因此关闭思考时降到最低可用档 low。
fn thinking_level(thinking: bool, saved_effort: &str) -> &'static str {
    if !thinking {
        return "low";
    }
    match saved_effort.trim().to_ascii_lowercase().as_str() {
        "minimal" => "minimal",
        "low" => "low",
        "medium" => "medium",
        // max/xhigh 是本项目的档位名，向上收敛到 high。
        "high" | "max" | "xhigh" => "high",
        // 配置文件可能被手动编辑；未知值回退到兼容性最好的 high。
        _ => "high",
    }
}

fn build_request_body(
    system: &str,
    question: &str,
    thinking: bool,
    saved_effort: &str,
) -> serde_json::Value {
    serde_json::json!({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            // 官方示例写作 thinking_level，protobuf JSON 两种拼写都接受。
            "thinkingConfig": {
                "thinkingLevel": thinking_level(thinking, saved_effort),
                "includeThoughts": thinking
            }
        }
    })
}

/// SSE 每个事件都是一段完整 GenerateContentResponse，解析入口与非流式响应相同。
fn parse_chunk(value: &serde_json::Value) -> Result<ParsedPayload, String> {
    if let Some(message) = extract_api_error(value) {
        return Err(format!("{LABEL} API 返回错误: {message}"));
    }
    if let Some(blocked) = value
        .pointer("/promptFeedback/blockReason")
        .and_then(serde_json::Value::as_str)
    {
        return Err(format!("{LABEL} 拒绝了该请求: {blocked}"));
    }

    let Some(candidate) = value
        .get("candidates")
        .and_then(serde_json::Value::as_array)
        .and_then(|candidates| candidates.first())
    else {
        return Ok(ParsedPayload::default());
    };

    if let Some(reason) = candidate.get("finishReason").and_then(serde_json::Value::as_str)
        && matches!(
            reason,
            "SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "SPII" | "IMAGE_SAFETY"
        )
    {
        return Err(format!("{LABEL} 拒绝了该请求: {reason}"));
    }

    let mut reasoning = String::new();
    let mut content = String::new();
    for part in candidate
        .pointer("/content/parts")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        // 思考内容与答案在同一个 parts 数组里，用 thought 标记区分。
        let Some(text) = part.get("text").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if part.get("thought").and_then(serde_json::Value::as_bool) == Some(true) {
            reasoning.push_str(text);
        } else {
            content.push_str(text);
        }
    }

    Ok(ParsedPayload {
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        content: (!content.is_empty()).then_some(content),
    })
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

impl GeminiClient {
    pub fn new(config: &LlmConfig) -> Self {
        let reasoning_effort = if config.reasoning_effort.trim().is_empty() {
            "high".to_string()
        } else {
            config.reasoning_effort.clone()
        };
        Self {
            http: build_http_client(),
            // 模型名在 URL 路径里，不进请求体。
            endpoint: endpoint(Protocol::Gemini, &config.base_url, &config.model),
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
            &system,
            &question,
            self.enable_thinking,
            &self.reasoning_effort,
        );

        let url = self.endpoint.clone();
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let request = apply_auth(
            http.post(&url).header("Content-Type", "application/json"),
            Protocol::Gemini,
            &api_key,
        )
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
                    match parse_chunk(&value) {
                        Ok(payload) => emit_payload(payload, &mut full_content, &tx),
                        Err(error) => {
                            send_error(&tx, &api_key, error);
                            return;
                        }
                    }
                }
                // Gemini 不发 [DONE]，流结束即收尾。
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
                    match parse_chunk(&value) {
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
    fn request_body_uses_contents_and_thinking_level() {
        let body = build_request_body("人设", "题目", true, "high");
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "人设");
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "题目");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], MAX_OUTPUT_TOKENS);
        assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "high");
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
        // 模型名在 URL 路径里，请求体不带 model
        assert!(body.get("model").is_none());
    }

    #[test]
    fn thinking_level_maps_project_levels_and_keeps_thoughts_off() {
        assert_eq!(thinking_level(true, "minimal"), "minimal");
        assert_eq!(thinking_level(true, "low"), "low");
        assert_eq!(thinking_level(true, "medium"), "medium");
        assert_eq!(thinking_level(true, "high"), "high");
        assert_eq!(thinking_level(true, "max"), "high");
        assert_eq!(thinking_level(true, "手改的值"), "high");
        // 没有完全关闭的档位，关闭时降到最低档并且不要思考摘要
        assert_eq!(thinking_level(false, "high"), "low");
        let body = build_request_body("人设", "题目", false, "max");
        assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low");
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            false
        );
    }

    #[test]
    fn parses_candidate_parts_splitting_thoughts_from_answer() {
        let value = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "在推理", "thought": true},
                        {"text": "3"}
                    ]
                }
            }]
        });
        assert_eq!(
            parse_chunk(&value).unwrap(),
            ParsedPayload {
                reasoning: Some("在推理".to_string()),
                content: Some("3".to_string())
            }
        );
    }

    #[test]
    fn empty_and_metadata_only_chunks_are_not_content() {
        let metadata = serde_json::json!({
            "candidates": [{"content": {"role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"totalTokenCount": 42}
        });
        assert_eq!(parse_chunk(&metadata).unwrap(), ParsedPayload::default());

        let empty = serde_json::json!({});
        assert_eq!(parse_chunk(&empty).unwrap(), ParsedPayload::default());
    }

    #[test]
    fn blocked_requests_and_api_errors_surface_messages() {
        let feedback = serde_json::json!({
            "promptFeedback": {"blockReason": "SAFETY"}
        });
        assert!(parse_chunk(&feedback).unwrap_err().contains("SAFETY"));

        let finish = serde_json::json!({
            "candidates": [{"finishReason": "PROHIBITED_CONTENT", "content": {"parts": []}}]
        });
        assert!(parse_chunk(&finish).unwrap_err().contains("PROHIBITED_CONTENT"));

        let error = serde_json::json!({
            "error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}
        });
        assert!(parse_chunk(&error).unwrap_err().contains("API key not valid"));
    }
}
