use crate::config::LlmConfig;
use crate::llm::prompt::{build_chat_prompt, build_quiz_prompt};
use crate::llm::protocol::{Protocol, endpoint};
use crate::llm::request::QuizRequest;
use crate::llm::shared::auth::apply_auth;
use crate::llm::shared::errors::extract_api_error;
use crate::llm::shared::http::{build_http_client, format_reqwest_error, safe_preview, send_error};
use crate::llm::shared::stream::{BodyStream, open_body_stream, read_error_message};
use crate::llm::LlmChunk;
use futures::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const LABEL: &str = "Gemini Interactions";
const MAX_JSON_RESPONSE_SIZE: usize = 4 * 1024 * 1024;

pub struct GeminiInteractionsClient {
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

/// 与 generateContent 共用一套思考档位，但没有"完全关闭"档，关闭时降到最低档。
fn thinking_level(thinking: bool, saved_effort: &str) -> &'static str {
    if !thinking {
        return "low";
    }
    match saved_effort.trim().to_ascii_lowercase().as_str() {
        "minimal" => "minimal",
        "low" => "low",
        "medium" => "medium",
        "high" | "max" | "xhigh" => "high",
        _ => "high",
    }
}

fn build_request_body(
    model: &str,
    input: &str,
    thinking: bool,
    saved_effort: &str,
) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": input,
        // Interactions 用 snake_case 且没有 thinkingConfig 这层嵌套。
        "generation_config": {"thinking_level": thinking_level(thinking, saved_effort)},
        // 默认会留存交互记录（付费层 55 天），答题是一次性的，没必要存。
        "store": false
    })
}

/// 思考以 steps 里的 thought 步骤呈现。官方只描述了这一层，
/// 字段名随版本可能变，所以只认显式标了 thought/thinking 的步骤——
/// 认不出来就不显示思考，答案走 output_text，不受影响。
fn collect_thoughts(steps: Option<&serde_json::Value>) -> Option<String> {
    let mut text = String::new();
    for step in steps?.as_array()? {
        let is_thought = step
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind.contains("thought") || kind.contains("thinking"));
        if !is_thought {
            continue;
        }
        for key in ["thought", "text"] {
            if let Some(value) = step.get(key).and_then(serde_json::Value::as_str) {
                text.push_str(value);
            }
        }
    }
    (!text.is_empty()).then_some(text)
}

fn parse_interaction(value: &serde_json::Value) -> Result<ParsedPayload, String> {
    if let Some(message) = extract_api_error(value) {
        return Err(format!("{LABEL} API 返回错误: {message}"));
    }
    Ok(ParsedPayload {
        reasoning: collect_thoughts(value.get("steps")),
        content: value
            .get("output_text")
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .map(str::to_owned),
    })
}

impl GeminiInteractionsClient {
    pub fn new(config: &LlmConfig) -> Self {
        let reasoning_effort = if config.reasoning_effort.trim().is_empty() {
            "high".to_string()
        } else {
            config.reasoning_effort.clone()
        };
        Self {
            http: build_http_client(),
            endpoint: endpoint(
                Protocol::GeminiInteractions,
                &config.base_url,
                &config.model,
            ),
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
        // Interactions 没有独立的 system 字段，人设只能拼进 input。
        let input = build_quiz_prompt(
            request.categories,
            &build_chat_prompt(request.question, request.options),
            self.enable_thinking,
        );
        tracing::info!("{LABEL} prompt:\n{}", input);
        let body = build_request_body(
            &self.model,
            &input,
            self.enable_thinking,
            &self.reasoning_effort,
        );

        let url = self.endpoint.clone();
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        // 流式事件格式官方未文档化，这里只发非流式请求拿完整 JSON，
        // 答题只需要最终答案。SSE 分支只是防止中转擅自改写响应格式的兜底。
        let sent = apply_auth(
            http.post(&url).header("Content-Type", "application/json"),
            Protocol::GeminiInteractions,
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
                result = sent.send() => match result {
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

            let mut payload = ParsedPayload::default();
            let mut failed = None;
            match body {
                BodyStream::Json(bytes) => {
                    match serde_json::from_slice::<serde_json::Value>(&bytes) {
                        Ok(value) => match parse_interaction(&value) {
                            Ok(parsed) => payload = parsed,
                            Err(error) => failed = Some(error),
                        },
                        Err(error) => {
                            let preview = safe_preview(&String::from_utf8_lossy(&bytes), &api_key, 200);
                            failed = Some(format!(
                                "解析 {LABEL} JSON 响应失败: {error}; 响应摘要: {preview}"
                            ));
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
                                format!("读取 {LABEL} SSE 响应失败: {error}"),
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
                    match parse_interaction(&value) {
                        // 兜底分支：每个事件都带完整 output_text，留最后一条非空的。
                        Ok(parsed) => {
                            if parsed.reasoning.is_some() {
                                payload.reasoning = parsed.reasoning;
                            }
                            if parsed.content.is_some() {
                                payload.content = parsed.content;
                            }
                        }
                        Err(error) => {
                            send_error(&tx, &api_key, error);
                            return;
                        }
                    }
                },
            }

            if let Some(error) = failed {
                send_error(&tx, &api_key, error);
                return;
            }
            if token.is_cancelled() {
                return;
            }

            let Some(content) = payload.content else {
                send_error(&tx, &api_key, format!("{LABEL} 响应结束但未包含可用正文"));
                return;
            };
            if let Some(reasoning) = payload.reasoning {
                let _ = tx.send(LlmChunk::Thinking(reasoning));
            }
            let _ = tx.send(LlmChunk::Content(content.clone()));
            let _ = tx.send(LlmChunk::Done(content));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_uses_input_and_flat_generation_config() {
        let body = build_request_body("gemini-3-flash-preview", "题目", true, "high");
        assert_eq!(body["model"], "gemini-3-flash-preview");
        assert_eq!(body["input"], "题目");
        assert_eq!(body["stream"].as_bool(), None, "只发非流式请求");
        // Interactions 是平铺的 snake_case，没有 thinkingConfig 那层
        assert_eq!(body["generation_config"]["thinking_level"], "high");
        assert!(body.get("generationConfig").is_none());
        assert!(body["generation_config"].get("thinkingConfig").is_none());
        assert_eq!(body["store"], false, "答题不需要留存交互记录");
    }

    #[test]
    fn thinking_level_matches_generate_content_rules() {
        assert_eq!(thinking_level(true, "minimal"), "minimal");
        assert_eq!(thinking_level(true, "medium"), "medium");
        assert_eq!(thinking_level(true, "max"), "high");
        assert_eq!(thinking_level(true, "手改的值"), "high");
        assert_eq!(thinking_level(false, "high"), "low");
    }

    #[test]
    fn parses_output_text_as_the_answer() {
        let value = serde_json::json!({
            "id": "interactions/abc",
            "output_text": "3",
            "steps": [{"type": "thought", "text": "在推理"}]
        });
        assert_eq!(
            parse_interaction(&value).unwrap(),
            ParsedPayload {
                reasoning: Some("在推理".to_string()),
                content: Some("3".to_string())
            }
        );
    }

    #[test]
    fn unmarked_steps_never_leak_into_the_thinking_channel() {
        // 最终答案所在步骤没有 thought 标记，不能被当成思考内容重复展示。
        let value = serde_json::json!({
            "output_text": "3",
            "steps": [
                {"type": "model_output", "text": "3"},
                {"type": "tool_call", "text": "忽略我"}
            ]
        });
        assert_eq!(
            parse_interaction(&value).unwrap(),
            ParsedPayload {
                reasoning: None,
                content: Some("3".to_string())
            }
        );
    }

    #[test]
    fn missing_or_blank_output_text_is_not_content() {
        assert_eq!(parse_interaction(&serde_json::json!({})).unwrap().content, None);
        let blank = serde_json::json!({"output_text": "   "});
        assert_eq!(parse_interaction(&blank).unwrap().content, None);
    }

    #[test]
    fn api_errors_are_reported() {
        let value = serde_json::json!({
            "error": {"code": 400, "message": "model not found", "status": "INVALID_ARGUMENT"}
        });
        assert!(parse_interaction(&value).unwrap_err().contains("model not found"));
    }
}
