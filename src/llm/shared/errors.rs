/// 各家原生协议的报错体都把可读信息放在 `error` 里，只是外层包装不同：
/// OpenAI 是 `{"error":{"message"}}`，Claude 是 `{"type":"error","error":{...}}`，
/// Gemini 是 `{"error":{"code","message","status"}}`。统一提取一次给用户看。
pub(crate) fn extract_api_error(value: &serde_json::Value) -> Option<String> {
    let is_claude_error = value.get("type").and_then(serde_json::Value::as_str) == Some("error");
    if is_claude_error
        && let Some(error) = value.get("error")
    {
        return describe(error);
    }
    value.get("error").and_then(describe)
}

fn describe(error: &serde_json::Value) -> Option<String> {
    match error {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Object(object) => {
            if let Some(message) = object.get("message").and_then(serde_json::Value::as_str) {
                return Some(message.to_string());
            }
            // Gemini 的部分错误只给 status 或 code，没有 message。
            if let Some(status) = object.get("status").and_then(serde_json::Value::as_str) {
                return Some(status.to_string());
            }
            object
                .get("code")
                .and_then(serde_json::Value::as_u64)
                .map(|code| code.to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_messages_from_all_three_error_shapes() {
        let openai = serde_json::json!({"error": {"message": "bad request", "code": "x"}});
        assert_eq!(extract_api_error(&openai).as_deref(), Some("bad request"));

        let claude = serde_json::json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "max_tokens too large"}
        });
        assert_eq!(
            extract_api_error(&claude).as_deref(),
            Some("max_tokens too large")
        );

        let gemini = serde_json::json!({
            "error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}
        });
        assert_eq!(
            extract_api_error(&gemini).as_deref(),
            Some("API key not valid")
        );
    }

    #[test]
    fn falls_back_to_status_or_code_and_ignores_non_errors() {
        let status_only = serde_json::json!({"error": {"status": "RESOURCE_EXHAUSTED"}});
        assert_eq!(
            extract_api_error(&status_only).as_deref(),
            Some("RESOURCE_EXHAUSTED")
        );

        let code_only = serde_json::json!({"error": {"code": 429}});
        assert_eq!(extract_api_error(&code_only).as_deref(), Some("429"));

        let plain = serde_json::json!({"error": "quota exceeded"});
        assert_eq!(extract_api_error(&plain).as_deref(), Some("quota exceeded"));

        assert_eq!(extract_api_error(&serde_json::json!({"choices": []})), None);
    }
}
