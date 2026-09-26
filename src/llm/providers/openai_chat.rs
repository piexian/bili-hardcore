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

const LABEL: &str = "LLM";
const MAX_JSON_RESPONSE_SIZE: usize = 8 * 1024 * 1024;
const MAX_VISIBLE_OUTPUT_TOKENS: u64 = 128;

pub struct OpenAiChatClient {
    http: Client,
    endpoint: String,
    model: String,
    api_key: String,
    enable_thinking: bool,
    reasoning_effort: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApiDialect {
    OpenAi,
    XAi,
    Extended,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ParsedPayload {
    reasoning: Option<String>,
    content: Option<String>,
}

fn api_dialect(endpoint: &str, model: &str) -> ApiDialect {
    let host = reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned));

    match host.as_deref() {
        Some("api.openai.com") => ApiDialect::OpenAi,
        Some("api.x.ai") => ApiDialect::XAi,
        // 自定义 Grok 中转仍使用 xAI-compatible Chat Completions 参数。
        _ if model.trim().to_ascii_lowercase().starts_with("grok-") => ApiDialect::XAi,
        _ => ApiDialect::Extended,
    }
}

fn xai_reasoning_effort(enable_thinking: bool, saved_effort: &str) -> &'static str {
    if !enable_thinking {
        // 当前 Grok 推理模型仍会推理，low 是可用的最低档位。
        return "low";
    }

    match saved_effort.trim() {
        effort if effort.eq_ignore_ascii_case("low") => "low",
        effort if effort.eq_ignore_ascii_case("medium") => "medium",
        effort if effort.eq_ignore_ascii_case("high") => "high",
        effort if effort.eq_ignore_ascii_case("max") || effort.eq_ignore_ascii_case("xhigh") => {
            "xhigh"
        }
        // 配置文件可能被手动编辑；未知值回退到兼容性最好的 high。
        _ => "high",
    }
}

fn build_request_body(
    dialect: ApiDialect,
    model: &str,
    prompt: &str,
    enable_thinking: bool,
    reasoning_effort: &str,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "stream": true,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    });
    let effort = if enable_thinking {
        reasoning_effort
    } else {
        "none"
    };

    match dialect {
        ApiDialect::OpenAi => {
            body["max_completion_tokens"] = serde_json::json!(MAX_VISIBLE_OUTPUT_TOKENS);
            body["reasoning_effort"] = serde_json::json!(effort);
        }
        ApiDialect::XAi => {
            // xAI 已弃用 max_tokens；该字段限制可见正文，推理开销由 effort 控制。
            body["max_completion_tokens"] = serde_json::json!(MAX_VISIBLE_OUTPUT_TOKENS);
            body["reasoning_effort"] =
                serde_json::json!(xai_reasoning_effort(enable_thinking, reasoning_effort,));
        }
        ApiDialect::Extended => {
            body["max_tokens"] = serde_json::json!(MAX_VISIBLE_OUTPUT_TOKENS);
            body["enable_thinking"] = serde_json::json!(enable_thinking);
            body["thinking"] = serde_json::json!({
                "type": if enable_thinking { "enabled" } else { "disabled" }
            });
            body["reasoning_effort"] = serde_json::json!(effort);
        }
    }

    body
}

fn extract_text(value: Option<&serde_json::Value>) -> Result<Option<String>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(text) => Ok((!text.is_empty()).then(|| text.clone())),
        serde_json::Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    serde_json::Value::Null => {}
                    serde_json::Value::String(value) => text.push_str(value),
                    serde_json::Value::Object(object) => {
                        let part_type = object.get("type").and_then(serde_json::Value::as_str);
                        let part_text = object.get("text").and_then(serde_json::Value::as_str);
                        if matches!(part_type, None | Some("text") | Some("output_text")) {
                            if let Some(value) = part_text {
                                text.push_str(value);
                            } else if !object.is_empty() {
                                return Err("LLM content 数组包含无法识别的文本结构".to_string());
                            }
                        } else {
                            return Err("LLM content 数组包含不支持的内容类型".to_string());
                        }
                    }
                    _ => return Err("LLM content 数组包含不支持的内容类型".to_string()),
                }
            }
            Ok((!text.is_empty()).then_some(text))
        }
        serde_json::Value::Object(object) => {
            if let Some(text) = object.get("text").and_then(serde_json::Value::as_str) {
                Ok((!text.is_empty()).then(|| text.to_string()))
            } else {
                Err("LLM content 使用了无法识别的对象结构".to_string())
            }
        }
        _ => Err("LLM content 不是文本或文本数组".to_string()),
    }
}

fn parse_payload(value: &serde_json::Value) -> Result<Option<ParsedPayload>, String> {
    if let Some(message) = extract_api_error(value) {
        return Err(format!("{LABEL} API 返回错误: {message}"));
    }

    let Some(choice) = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
    else {
        return Ok(None);
    };
    let Some(payload) = choice
        .get("delta")
        .filter(|value| value.is_object())
        .or_else(|| choice.get("message").filter(|value| value.is_object()))
    else {
        return Ok(None);
    };

    Ok(Some(ParsedPayload {
        reasoning: extract_text(
            payload
                .get("reasoning_content")
                .or_else(|| payload.get("reasoning")),
        )?,
        content: extract_text(payload.get("content"))?,
    }))
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

impl OpenAiChatClient {
    pub fn new(config: &LlmConfig) -> Self {
        // 兜底：配置文件手动编辑导致空值时回退到默认 high
        let reasoning_effort = if config.reasoning_effort.trim().is_empty() {
            "high".to_string()
        } else {
            config.reasoning_effort.clone()
        };
        Self {
            http: build_http_client(),
            endpoint: endpoint(Protocol::OpenAiChat, &config.base_url, &config.model),
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
            api_dialect(&self.endpoint, &self.model),
            &self.model,
            &prompt,
            self.enable_thinking,
            &self.reasoning_effort,
        );

        let url = self.endpoint.clone();
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let request = apply_auth(
            http.post(&url).header("Content-Type", "application/json"),
            Protocol::OpenAiChat,
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

            let status = resp.status();
            if !status.is_success() {
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
                    match parse_payload(&value) {
                        Ok(Some(payload)) => emit_payload(payload, &mut full_content, &tx),
                        Ok(None) => {
                            send_error(&tx, &api_key, format!("{LABEL} JSON 响应不包含 choices[0] 内容"));
                            return;
                        }
                        Err(error) => {
                            send_error(&tx, &api_key, error);
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
                    match parse_payload(&value) {
                        Ok(Some(payload)) => emit_payload(payload, &mut full_content, &tx),
                        Ok(None) => {}
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
    fn detects_api_dialect_by_host_and_model() {
        assert_eq!(
            api_dialect("https://api.openai.com/v1/chat/completions", "gpt-test"),
            ApiDialect::OpenAi
        );
        assert_eq!(
            api_dialect("https://API.X.AI:443/v1/chat/completions", "grok-4.6"),
            ApiDialect::XAi
        );
        assert_eq!(
            api_dialect("http://proxy.example/chat/completions", " Grok-4.6 "),
            ApiDialect::XAi
        );
        assert_eq!(
            api_dialect(
                "https://api.x.ai.example/v1/chat/completions",
                "other-model"
            ),
            ApiDialect::Extended
        );
        assert_eq!(
            api_dialect("http://proxy.example/chat/completions", "deepseek-chat"),
            ApiDialect::Extended
        );
    }

    #[test]
    fn xai_body_maps_reasoning_effort_and_limits_completion() {
        for (saved, expected) in [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("max", "xhigh"),
            ("xhigh", "xhigh"),
            ("edited-value", "high"),
        ] {
            let body = build_request_body(ApiDialect::XAi, "grok-4.6", "question", true, saved);
            assert_eq!(body["reasoning_effort"], expected);
            assert_eq!(body["max_completion_tokens"], MAX_VISIBLE_OUTPUT_TOKENS);
        }

        let disabled = build_request_body(ApiDialect::XAi, "grok-4.6", "question", false, "max");
        assert_eq!(disabled["reasoning_effort"], "low");
        assert!(disabled.get("enable_thinking").is_none());
        assert!(disabled.get("thinking").is_none());
    }

    #[test]
    fn existing_dialects_keep_their_reasoning_fields_and_output_limit() {
        let openai = build_request_body(ApiDialect::OpenAi, "gpt-test", "question", true, "high");
        assert_eq!(openai["reasoning_effort"], "high");
        assert_eq!(openai["max_completion_tokens"], MAX_VISIBLE_OUTPUT_TOKENS);
        assert!(openai.get("max_tokens").is_none());
        assert!(openai.get("enable_thinking").is_none());

        let extended = build_request_body(ApiDialect::Extended, "other", "question", false, "max");
        assert_eq!(extended["enable_thinking"], false);
        assert_eq!(extended["thinking"]["type"], "disabled");
        assert_eq!(extended["reasoning_effort"], "none");
        assert_eq!(extended["max_tokens"], MAX_VISIBLE_OUTPUT_TOKENS);
        assert!(extended.get("max_completion_tokens").is_none());
    }

    #[test]
    fn parses_synchronous_message_content() {
        let value = serde_json::json!({
            "choices": [{"message": {"content": "3"}}]
        });
        assert_eq!(
            parse_payload(&value),
            Ok(Some(ParsedPayload {
                reasoning: None,
                content: Some("3".to_string())
            }))
        );
    }

    #[test]
    fn parses_reasoning_and_text_part_arrays() {
        let value = serde_json::json!({
            "choices": [{"message": {
                "reasoning_content": "分析",
                "content": [
                    {"type": "text", "text": "答案"},
                    {"type": "output_text", "text": "是2"}
                ]
            }}]
        });
        assert_eq!(
            parse_payload(&value),
            Ok(Some(ParsedPayload {
                reasoning: Some("分析".to_string()),
                content: Some("答案是2".to_string())
            }))
        );
    }

    #[test]
    fn parses_stream_delta_and_message_fallback() {
        let delta = serde_json::json!({"choices": [{"delta": {"content": "1"}}]});
        let message = serde_json::json!({"choices": [{"message": {"content": "4"}}]});
        assert_eq!(
            parse_payload(&delta).unwrap().unwrap().content.as_deref(),
            Some("1")
        );
        assert_eq!(
            parse_payload(&message).unwrap().unwrap().content.as_deref(),
            Some("4")
        );
    }

    #[test]
    fn empty_choices_and_api_errors_are_not_content() {
        assert_eq!(parse_payload(&serde_json::json!({"choices": []})), Ok(None));
        assert!(
            parse_payload(&serde_json::json!({"error": {"message": "bad request"}}))
                .unwrap_err()
                .contains("bad request")
        );
        assert!(
            parse_payload(&serde_json::json!({
                "choices": [{"message": {"content": {"unsupported": true}}}]
            }))
            .is_err()
        );
    }
}
