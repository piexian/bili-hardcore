use crate::config::LlmConfig;
use crate::llm::prompt::{build_chat_prompt, build_quiz_prompt};
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

const LABEL: &str = "OpenAI Responses";
const MAX_JSON_RESPONSE_SIZE: usize = 8 * 1024 * 1024;
const MAX_VISIBLE_OUTPUT_TOKENS: u64 = 128;

pub struct OpenAiResponsesClient {
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

/// xAI 的 Grok 走同一个 Responses 端点，但档位与 OpenAI 不同：
/// 只有 low/medium/high/xhigh，没有 minimal，收到 minimal 会报错。
fn is_xai(endpoint: &str, model: &str) -> bool {
    let host = reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned));
    // 自定义 Grok 中转同样按模型名识别。
    match host.as_deref() {
        Some("api.x.ai") => true,
        _ => model.trim().to_ascii_lowercase().starts_with("grok-"),
    }
}

/// Responses 的 reasoning.effort 取值随厂商不同：OpenAI 是 minimal/low/medium/high，
/// xAI 是 low/medium/high/xhigh。关闭思考时降到各家最低档。
fn reasoning_effort(xai: bool, thinking: bool, saved_effort: &str) -> &'static str {
    if !thinking {
        return if xai { "low" } else { "minimal" };
    }
    match saved_effort.trim().to_ascii_lowercase().as_str() {
        "minimal" if !xai => "minimal",
        // xAI 不认 minimal，向上落到 low。
        "minimal" => "low",
        "low" => "low",
        "medium" => "medium",
        // xAI 独占 xhigh，OpenAI 侧向上收敛到 high。
        "max" | "xhigh" if xai => "xhigh",
        "high" | "max" | "xhigh" => "high",
        // 配置文件可能被手动编辑；未知值回退到兼容性最好的 high。
        _ => "high",
    }
}

fn build_request_body(
    model: &str,
    prompt: &str,
    thinking: bool,
    saved_effort: &str,
    xai: bool,
) -> serde_json::Value {
    let mut reasoning = serde_json::Map::new();
    reasoning.insert(
        "effort".to_string(),
        serde_json::json!(reasoning_effort(xai, thinking, saved_effort)),
    );
    // 只有开了思考才有推理摘要可流式展示；xAI 是否接受 summary 未文档化，
    // 省掉它最稳——少一段思考展示，总好过整个请求被拒。
    if thinking && !xai {
        reasoning.insert("summary".to_string(), serde_json::json!("auto"));
    }

    serde_json::json!({
        "model": model,
        "stream": true,
        "input": prompt,
        "max_output_tokens": MAX_VISIBLE_OUTPUT_TOKENS,
        "reasoning": serde_json::Value::Object(reasoning),
        // 一次性答题，不需要服务端留存对话。
        "store": false
    })
}

fn non_empty(text: Option<&str>) -> Option<String> {
    text.filter(|value| !value.is_empty()).map(str::to_owned)
}

fn parse_event(value: &serde_json::Value) -> Result<ParsedPayload, String> {
    if let Some(message) = extract_api_error(value) {
        return Err(format!("{LABEL} API 返回错误: {message}"));
    }
    if let Some(message) = terminal_error(value) {
        return Err(message);
    }

    let delta = value.get("delta");
    Ok(match value.get("type").and_then(serde_json::Value::as_str) {
        Some("response.output_text.delta") => ParsedPayload {
            content: non_empty(delta.and_then(serde_json::Value::as_str)),
            ..Default::default()
        },
        Some("response.reasoning_summary_text.delta" | "response.reasoning_text.delta") => {
            ParsedPayload {
                reasoning: non_empty(delta.and_then(serde_json::Value::as_str)),
                ..Default::default()
            }
        }
        _ => ParsedPayload::default(),
    })
}

/// 终止类事件：失败或未完成都要显式报错，否则会被当成"没有正文"。
fn terminal_error(value: &serde_json::Value) -> Option<String> {
    let event_type = value.get("type").and_then(serde_json::Value::as_str)?;
    if !matches!(
        event_type,
        "error" | "response.failed" | "response.incomplete"
    ) {
        return None;
    }
    if let Some(message) = extract_api_error(value)
        .or_else(|| value.get("response").and_then(extract_api_error))
    {
        return Some(format!("{LABEL} 返回错误: {message}"));
    }
    let reason = value
        .pointer("/response/incomplete_details/reason")
        .and_then(serde_json::Value::as_str);
    Some(format!("{LABEL} 响应未完成: {}", reason.unwrap_or("未知原因")))
}

/// `response.completed` 里带完整 output，增量丢失时用它兜底。
fn text_from_response(response: &serde_json::Value) -> Option<String> {
    let mut text = String::new();
    for item in response.get("output")?.as_array()? {
        // reasoning 条目承载的是推理摘要，不计入答案正文。
        if item.get("type").and_then(serde_json::Value::as_str) == Some("reasoning") {
            continue;
        }
        for part in item.get("content")?.as_array()? {
            if part.get("type").and_then(serde_json::Value::as_str) == Some("output_text")
                && let Some(value) = part.get("text").and_then(serde_json::Value::as_str)
            {
                text.push_str(value);
            }
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

impl OpenAiResponsesClient {
    pub fn new(config: &LlmConfig) -> Self {
        let reasoning_effort = if config.reasoning_effort.trim().is_empty() {
            "high".to_string()
        } else {
            config.reasoning_effort.clone()
        };
        Self {
            http: build_http_client(),
            endpoint: endpoint(Protocol::OpenAiResponses, &config.base_url, &config.model),
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
        let prompt = build_quiz_prompt(
            request.categories,
            &build_chat_prompt(request.question, request.options),
            self.enable_thinking,
        );
        tracing::info!("{LABEL} prompt:\n{}", prompt);
        let body = build_request_body(
            &self.model,
            &prompt,
            self.enable_thinking,
            &self.reasoning_effort,
            is_xai(&self.endpoint, &self.model),
        );

        let url = self.endpoint.clone();
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let request = apply_auth(
            http.post(&url).header("Content-Type", "application/json"),
            Protocol::OpenAiResponses,
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
                    if let Some(message) = terminal_error(&value) {
                        send_error(&tx, &api_key, message);
                        return;
                    }
                    let text = value
                        .get("response")
                        .and_then(text_from_response)
                        .or_else(|| text_from_response(&value));
                    match text {
                        Some(text) => {
                            full_content = text;
                            let _ = tx.send(LlmChunk::Content(full_content.clone()));
                        }
                        None => {
                            send_error(&tx, &api_key, format!("{LABEL} JSON 响应不包含 output_text"));
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
                    // Responses 不发 [DONE]，以 response.completed 收尾；
                    // 保留判断只为兼容会补发该哨兵的中转。
                    if event.data == "[DONE]" {
                        break;
                    }
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
                    // 增量丢失（部分中转会跳过 delta 事件）时用完整 output 兜底。
                    if full_content.is_empty()
                        && value.get("type").and_then(serde_json::Value::as_str)
                            == Some("response.completed")
                        && let Some(text) = value.get("response").and_then(text_from_response)
                    {
                        full_content = text;
                        let _ = tx.send(LlmChunk::Content(full_content.clone()));
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
    fn request_body_uses_input_and_responses_reasoning_fields() {
        let body = build_request_body("gpt-test", "prompt", true, "high", false);
        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["input"], "prompt");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_output_tokens"], MAX_VISIBLE_OUTPUT_TOKENS);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert_eq!(body["store"], false);
        // Responses 没有 messages 兼容层
        assert!(body.get("messages").is_none());
    }

    #[test]
    fn openai_effort_clamps_project_levels_to_responses_values() {
        assert_eq!(reasoning_effort(false, true, "minimal"), "minimal");
        assert_eq!(reasoning_effort(false, true, "low"), "low");
        assert_eq!(reasoning_effort(false, true, "medium"), "medium");
        assert_eq!(reasoning_effort(false, true, "high"), "high");
        assert_eq!(reasoning_effort(false, true, "max"), "high");
        assert_eq!(reasoning_effort(false, true, "xhigh"), "high");
        assert_eq!(reasoning_effort(false, true, "手改的值"), "high");
        // 关闭思考降到最低档，OpenAI 侧是 minimal
        assert_eq!(reasoning_effort(false, false, "high"), "minimal");
        let body = build_request_body("gpt-test", "prompt", false, "max", false);
        assert_eq!(body["reasoning"]["effort"], "minimal");
        assert!(
            body["reasoning"].get("summary").is_none(),
            "不思考时没有摘要可展示，整个字段省掉"
        );
    }

    #[test]
    fn xai_gets_its_own_effort_scale_and_no_summary() {
        // Grok 没有 minimal，关思考必须落到 low
        assert_eq!(reasoning_effort(true, false, "high"), "low");
        assert_eq!(reasoning_effort(true, false, "max"), "low");
        assert_eq!(reasoning_effort(true, true, "minimal"), "low");
        // xAI 独占 xhigh，不该被收敛掉
        assert_eq!(reasoning_effort(true, true, "max"), "xhigh");
        assert_eq!(reasoning_effort(true, true, "xhigh"), "xhigh");
        assert_eq!(reasoning_effort(true, true, "high"), "high");
        assert_eq!(reasoning_effort(true, true, "medium"), "medium");
        assert_eq!(reasoning_effort(true, true, "low"), "low");
        assert_eq!(reasoning_effort(true, true, "手改的值"), "high");

        // summary 是否被 xAI 接受没有文档，省掉最稳
        let thinking = build_request_body("grok-4.6", "prompt", true, "max", true);
        assert_eq!(thinking["reasoning"]["effort"], "xhigh");
        assert!(thinking["reasoning"].get("summary").is_none());
        let plain = build_request_body("gpt-test", "prompt", false, "high", true);
        assert_eq!(plain["reasoning"]["effort"], "low");
        assert!(plain["reasoning"].get("summary").is_none());
    }

    #[test]
    fn detects_xai_by_host_and_by_model_for_relays() {
        assert!(is_xai("https://api.x.ai/v1/responses", "grok-4.6"));
        assert!(is_xai("http://proxy.example/v1/responses", " Grok-4.6 "));
        assert!(!is_xai("https://api.x.ai.example/v1/responses", "gpt-test"));
        assert!(!is_xai("https://api.openai.com/v1/responses", "gpt-test"));
    }

    #[test]
    fn parses_text_and_reasoning_deltas() {
        let text = serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "3"
        });
        assert_eq!(
            parse_event(&text).unwrap(),
            ParsedPayload {
                reasoning: None,
                content: Some("3".to_string())
            }
        );

        let reasoning = serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "分析中"
        });
        assert_eq!(
            parse_event(&reasoning).unwrap(),
            ParsedPayload {
                reasoning: Some("分析中".to_string()),
                content: None
            }
        );

        // 生命周期事件不产生内容
        let created = serde_json::json!({"type": "response.created", "delta": "x"});
        assert_eq!(parse_event(&created).unwrap(), ParsedPayload::default());
    }

    #[test]
    fn extracts_final_text_from_completed_output() {
        let value = serde_json::json!({
            "type": "response.completed",
            "response": {
                "output": [
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "想"}]},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "答"}]
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "案2"}]
                    }
                ]
            }
        });
        assert_eq!(text_from_response(&value["response"]).as_deref(), Some("答案2"));
        assert_eq!(parse_event(&value).unwrap(), ParsedPayload::default());
    }

    #[test]
    fn failed_and_incomplete_events_surface_errors() {
        let failed = serde_json::json!({
            "type": "response.failed",
            "response": {"error": {"message": "boom"}}
        });
        assert!(parse_event(&failed).unwrap_err().contains("boom"));

        let incomplete = serde_json::json!({
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "max_output_tokens"}}
        });
        assert!(
            parse_event(&incomplete)
                .unwrap_err()
                .contains("max_output_tokens")
        );

        let raw_error = serde_json::json!({"type": "error", "message": "stream interrupted"});
        assert!(parse_event(&raw_error).is_err());
    }
}
