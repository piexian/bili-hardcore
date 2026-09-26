use crate::llm::protocol::{Protocol, versioned_base};
use crate::llm::shared::auth::apply_auth;
use crate::llm::shared::errors::extract_api_error;
use crate::llm::shared::http::{build_http_client_with, format_reqwest_error, safe_preview};
use std::time::Duration;

/// 模型表是给人看的，超过这个量在选择器里已经翻不动了。
const MAX_MODELS: usize = 200;
const MAX_PAGE_REQUESTS: usize = 3;
const MAX_RESPONSE_SIZE: usize = 4 * 1024 * 1024;
const READ_TIMEOUT: Duration = Duration::from_secs(30);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// 写进配置、实际请求里用的模型名。
    pub id: String,
    /// 展示名，接口没给就退回 id。
    pub label: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelList {
    pub models: Vec<ModelEntry>,
    /// 命中上限被截断，界面需要提示还有更多。
    pub truncated: bool,
}

/// OpenAI 对未识别的 query 参数直接 400，分页参数不能照抄 Claude。
/// 模型列表走用户填的那个版本前缀——服务在 /v2 上时，列表也在 /v2 上。
pub fn models_url(protocol: Protocol, base_url: &str) -> String {
    let base = versioned_base(protocol, base_url);
    match protocol {
        Protocol::OpenAiChat | Protocol::OpenAiResponses | Protocol::Jev => {
            format!("{base}/models")
        }
        Protocol::Claude => format!("{base}/models?limit=1000"),
        Protocol::Gemini | Protocol::GeminiInteractions => {
            format!("{base}/models?pageSize=1000")
        }
    }
}

/// 游标续页的下一个地址；无游标的协议返回空串。
fn next_url(protocol: Protocol, base_url: &str, cursor: &str) -> String {
    let cursor = urlencoding::encode(cursor);
    let base = versioned_base(protocol, base_url);
    match protocol {
        Protocol::Claude => format!("{base}/models?limit=1000&after_id={cursor}"),
        Protocol::Gemini | Protocol::GeminiInteractions => {
            format!("{base}/models?pageSize=1000&pageToken={cursor}")
        }
        _ => String::new(),
    }
}

fn next_cursor(protocol: Protocol, value: &serde_json::Value) -> Option<String> {
    match protocol {
        Protocol::Claude => {
            if value.get("has_more").and_then(serde_json::Value::as_bool) != Some(true) {
                return None;
            }
            value
                .get("last_id")
                .and_then(serde_json::Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
        }
        Protocol::Gemini | Protocol::GeminiInteractions => value
            .get("nextPageToken")
            .and_then(serde_json::Value::as_str)
            .filter(|token| !token.is_empty())
            .map(str::to_owned),
        _ => None,
    }
}

/// 选择器每页展示的条数。
pub const MODELS_PER_PAGE: usize = 20;

/// 搜索按 id 和展示名同时匹配，大小写无关；空过滤返回全部。
pub fn filter_models<'a>(models: &'a [ModelEntry], filter: &str) -> Vec<&'a ModelEntry> {
    let needle = filter.trim().to_lowercase();
    if needle.is_empty() {
        return models.iter().collect();
    }
    models
        .iter()
        .filter(|model| {
            model.id.to_lowercase().contains(&needle)
                || model.label.to_lowercase().contains(&needle)
        })
        .collect()
}

pub fn page_count(count: usize) -> usize {
    if count == 0 {
        0
    } else {
        count.div_ceil(MODELS_PER_PAGE)
    }
}

/// 四种响应结构：OpenAI/Claude 是 `data[].id`，Gemini/JEV 是 `models[].name`。
/// 按响应形状判定而不是按协议判定，中转把某个协议转成另一种形状时也能读。
pub fn parse_list(value: &serde_json::Value) -> Vec<ModelEntry> {
    if let Some(data) = value.get("data").and_then(serde_json::Value::as_array) {
        return data
            .iter()
            .filter_map(|entry| {
                let id = entry.get("id").and_then(serde_json::Value::as_str)?;
                Some(ModelEntry {
                    label: entry
                        .get("display_name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(id)
                        .to_string(),
                    id: id.to_string(),
                    detail: None,
                })
            })
            .collect();
    }

    value
        .get("models")
        .and_then(serde_json::Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|entry| {
                    let raw = entry.get("name").and_then(serde_json::Value::as_str)?;
                    // Gemini 的 name 带 models/ 前缀，JEV 不带，存在就剥。
                    let id = raw.strip_prefix("models/").unwrap_or(raw);
                    if id.is_empty() {
                        return None;
                    }
                    // Gemini 会把嵌入、图像等模型一并列出，选中也答不了题。
                    if let Some(methods) =
                        entry.get("supportedGenerationMethods").and_then(serde_json::Value::as_array)
                        && !methods
                            .iter()
                            .any(|method| method.as_str() == Some("generateContent"))
                    {
                        return None;
                    }
                    Some(ModelEntry {
                        id: id.to_string(),
                        label: entry
                            .get("displayName")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or(id)
                            .to_string(),
                        detail: entry
                            .get("description")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn describe_http_error(
    protocol: Protocol,
    status: reqwest::StatusCode,
    body: &str,
    api_key: &str,
) -> String {
    let label = protocol.display_name();
    if status.as_u16() == 404 {
        return format!(
            "{label} 未提供模型列表接口 (HTTP 404)，可手动输入模型名；确认基址为 {label} 官方域名或已实现该接口的中转"
        );
    }
    if status.as_u16() == 401 || status.as_u16() == 403 {
        return format!("{label} 拒绝了请求 (HTTP {status})，请检查 API Key");
    }
    let message = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| extract_api_error(&value));
    match message {
        Some(message) => format!("拉取 {label} 模型列表失败 (HTTP {status}): {message}"),
        None => format!(
            "拉取 {label} 模型列表失败 (HTTP {status}): {}",
            safe_preview(body, api_key, 200)
        ),
    }
}

pub async fn fetch_models(
    protocol: Protocol,
    base_url: &str,
    api_key: &str,
) -> Result<ModelList, String> {
    let http = build_http_client_with(READ_TIMEOUT, TOTAL_TIMEOUT);
    let mut url = models_url(protocol, base_url);
    let mut models: Vec<ModelEntry> = Vec::new();
    let mut truncated = false;

    for page in 0..MAX_PAGE_REQUESTS {
        let response = apply_auth(http.get(&url), protocol, api_key)
            .send()
            .await
            .map_err(|error| {
                format_reqwest_error(&format!("拉取 {} 模型列表失败", protocol.display_name()), &error, api_key)
            })?;

        let status = response.status();
        let bytes = response.bytes().await.map_err(|error| {
            format_reqwest_error("读取模型列表响应失败", &error, api_key)
        })?;
        if bytes.len() > MAX_RESPONSE_SIZE {
            return Err("模型列表响应超过 4 MiB 限制".to_string());
        }
        let body = String::from_utf8_lossy(&bytes);
        if !status.is_success() {
            return Err(describe_http_error(protocol, status, &body, api_key));
        }

        let value: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
            let preview = safe_preview(&body, api_key, 200);
            format!("解析模型列表 JSON 失败: {error}; 响应摘要: {preview}")
        })?;
        models.extend(parse_list(&value));

        if models.len() >= MAX_MODELS {
            models.truncate(MAX_MODELS);
            truncated = true;
            break;
        }
        match next_cursor(protocol, &value) {
            Some(cursor) => {
                if page + 1 == MAX_PAGE_REQUESTS {
                    truncated = true;
                }
                url = next_url(protocol, base_url, &cursor);
            }
            None => break,
        }
    }

    if models.is_empty() {
        return Err(format!(
            "{} 返回了空的模型列表，请确认该 Key 有权访问模型",
            protocol.display_name()
        ));
    }
    Ok(ModelList { models, truncated })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_each_protocols_documented_models_endpoint() {
        assert_eq!(
            models_url(Protocol::OpenAiChat, "https://api.openai.com"),
            "https://api.openai.com/v1/models"
        );
        assert_eq!(
            models_url(Protocol::OpenAiResponses, "https://api.openai.com"),
            "https://api.openai.com/v1/models"
        );
        assert_eq!(
            models_url(Protocol::Jev, "https://api.typesafe.ai"),
            "https://api.typesafe.ai/v1/models"
        );
        assert_eq!(
            models_url(Protocol::Claude, "https://api.anthropic.com"),
            "https://api.anthropic.com/v1/models?limit=1000"
        );
        assert_eq!(
            models_url(Protocol::Gemini, "https://generativelanguage.googleapis.com"),
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        );
        assert_eq!(
            models_url(
                Protocol::GeminiInteractions,
                "https://generativelanguage.googleapis.com"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        );
    }

    #[test]
    fn openai_gets_no_pagination_parameters() {
        // OpenAI 对未识别的 query 参数会 400，带上 limit 就废了。
        let url = models_url(Protocol::OpenAiChat, "https://api.openai.com");
        assert!(!url.contains('?'));
    }

    #[test]
    fn endpoints_follow_the_same_base_rules_as_ask() {
        assert_eq!(
            models_url(Protocol::OpenAiChat, "https://open.bigmodel.cn/api/paas/v4"),
            "https://open.bigmodel.cn/api/paas/v4/models"
        );
        assert_eq!(
            models_url(Protocol::Claude, "https://relay.example.com/v1"),
            "https://relay.example.com/v1/models?limit=1000"
        );
        // 旧配置里的完整端点也要能直接拿来拉列表
        assert_eq!(
            models_url(Protocol::Jev, "http://192.168.0.2:3000/v1/systemone"),
            "http://192.168.0.2:3000/v1/models"
        );
    }

    #[test]
    fn model_list_lands_on_the_same_version_prefix_as_ask() {
        // 服务挂在 /v2 上时，答题和模型列表都必须在 /v2，不能一个 v1 一个 v2。
        assert_eq!(
            models_url(Protocol::OpenAiChat, "https://relay.example.com/v2"),
            "https://relay.example.com/v2/models"
        );
        assert_eq!(
            models_url(Protocol::OpenAiResponses, "https://relay.example.com/v3"),
            "https://relay.example.com/v3/models"
        );
        assert_eq!(
            next_url(
                Protocol::Claude,
                "https://relay.example.com/v2",
                "claude-x"
            ),
            "https://relay.example.com/v2/models?limit=1000&after_id=claude-x"
        );
        assert_eq!(
            next_url(Protocol::Gemini, "https://relay.example.com/v1beta", "tok"),
            "https://relay.example.com/v1beta/models?pageSize=1000&pageToken=tok"
        );
    }

    #[test]
    fn parses_openai_list_shape() {
        let value = serde_json::json!({
            "object": "list",
            "data": [
                {"id": "gpt-5.4-nano", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.4", "object": "model", "owned_by": "openai"}
            ]
        });
        let models = parse_list(&value);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-5.4-nano");
        assert_eq!(models[0].label, "gpt-5.4-nano");
    }

    #[test]
    fn parses_claude_list_shape_with_display_name() {
        let value = serde_json::json!({
            "data": [{"id": "claude-sonnet-5", "display_name": "Claude Sonnet 5", "type": "model"}],
            "has_more": false
        });
        let models = parse_list(&value);
        assert_eq!(models[0].id, "claude-sonnet-5");
        assert_eq!(models[0].label, "Claude Sonnet 5");
    }

    #[test]
    fn parses_gemini_list_shape_stripping_prefix_and_filtering_methods() {
        let value = serde_json::json!({
            "models": [
                {
                    "name": "models/gemini-3-flash-preview",
                    "displayName": "Gemini 3 Flash",
                    "description": "frontier",
                    "supportedGenerationMethods": ["generateContent", "countTokens"]
                },
                {
                    "name": "models/text-embedding-004",
                    "displayName": "Embedding",
                    "supportedGenerationMethods": ["embedContent"]
                }
            ],
            "nextPageToken": "tok"
        });
        let models = parse_list(&value);
        assert_eq!(models.len(), 1, "只支持 generateContent 的模型应被过滤掉");
        assert_eq!(models[0].id, "gemini-3-flash-preview", "要剥掉 models/ 前缀");
        assert_eq!(models[0].label, "Gemini 3 Flash");
        assert_eq!(models[0].detail.as_deref(), Some("frontier"));
    }

    #[test]
    fn parses_jev_list_shape_without_prefix() {
        let value = serde_json::json!({
            "models": [
                {"name": "jev-latest", "description": "always newest", "release_date": "2026-09-15"},
                {"name": "jev-1.13.0", "description": "pinned", "release_date": "2026-09-11"}
            ]
        });
        let models = parse_list(&value);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "jev-latest", "JEV 名字不带前缀，不能动它");
        assert_eq!(models[0].label, "jev-latest", "JEV 没有 displayName，退回 id");
        assert_eq!(models[0].detail.as_deref(), Some("always newest"));
    }

    #[test]
    fn skips_entries_without_usable_names() {
        let value = serde_json::json!({
            "models": [{"displayName": "no name"}, {"name": ""}, {"name": "models/ok"}]
        });
        let models = parse_list(&value);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "ok");
    }

    #[test]
    fn unknown_shapes_yield_nothing_rather_than_panicking() {
        assert!(parse_list(&serde_json::json!({})).is_empty());
        assert!(parse_list(&serde_json::json!({"error": {}})).is_empty());
        assert!(parse_list(&serde_json::json!([1, 2, 3])).is_empty());
    }

    #[test]
    fn cursors_only_continue_when_there_is_more() {
        let claude_more = serde_json::json!({"data": [], "has_more": true, "last_id": "claude-x"});
        assert_eq!(
            next_cursor(Protocol::Claude, &claude_more).as_deref(),
            Some("claude-x")
        );
        let claude_last = serde_json::json!({"data": [], "has_more": false, "last_id": "claude-x"});
        assert_eq!(next_cursor(Protocol::Claude, &claude_last), None);

        let gemini = serde_json::json!({"models": [], "nextPageToken": "tok2"});
        assert_eq!(next_cursor(Protocol::Gemini, &gemini).as_deref(), Some("tok2"));
        let gemini_last = serde_json::json!({"models": []});
        assert_eq!(next_cursor(Protocol::Gemini, &gemini_last), None);

        // OpenAI / JEV 一次性返回，没有游标
        let openai = serde_json::json!({"data": [{"id": "m"}], "has_more": true, "last_id": "m"});
        assert_eq!(next_cursor(Protocol::OpenAiChat, &openai), None);
        assert_eq!(next_cursor(Protocol::Jev, &openai), None);
    }

    #[test]
    fn next_url_encodes_the_cursor() {
        assert_eq!(
            next_url(Protocol::Claude, "https://api.anthropic.com", "claude a/b"),
            "https://api.anthropic.com/v1/models?limit=1000&after_id=claude%20a%2Fb"
        );
        assert_eq!(
            next_url(Protocol::Gemini, "https://generativelanguage.googleapis.com", "a b"),
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&pageToken=a%20b"
        );
        assert_eq!(next_url(Protocol::OpenAiChat, "https://api.openai.com", "x"), "");
    }
    #[test]
    fn filter_matches_id_and_label_case_insensitively() {
        let models = vec![
            ModelEntry {
                id: "claude-sonnet-5".to_string(),
                label: "Claude Sonnet 5".to_string(),
                detail: None,
            },
            ModelEntry {
                id: "gemini-3-flash-preview".to_string(),
                label: "Gemini 3 Flash".to_string(),
                detail: None,
            },
            ModelEntry {
                id: "jev-latest".to_string(),
                label: "jev-latest".to_string(),
                detail: None,
            },
        ];

        assert_eq!(filter_models(&models, "").len(), 3);
        assert_eq!(filter_models(&models, "   ").len(), 3);
        assert_eq!(filter_models(&models, "CLAUDE").len(), 1, "展示名要能搜到");
        assert_eq!(filter_models(&models, "flash").len(), 1);
        assert_eq!(filter_models(&models, "3").len(), 1, "片段匹配");
        assert_eq!(filter_models(&models, "Sonnet").len(), 1);
        assert!(filter_models(&models, "不存在").is_empty());
    }

    #[test]
    fn pages_hold_twenty_entries() {
        assert_eq!(MODELS_PER_PAGE, 20);
        assert_eq!(page_count(0), 0);
        assert_eq!(page_count(1), 1);
        assert_eq!(page_count(20), 1);
        assert_eq!(page_count(21), 2);
        assert_eq!(page_count(200), 10);
        // 光标落在第几页：0/19 与 20/21 分属两页
        for (cursor, page) in [(0usize, 0usize), (19, 0), (20, 1), (21, 1), (199, 9)] {
            assert_eq!(cursor / MODELS_PER_PAGE, page, "cursor={cursor}");
        }
    }

    #[test]
    fn missing_endpoint_and_bad_key_get_actionable_messages() {
        let not_found = describe_http_error(
            Protocol::GeminiInteractions,
            reqwest::StatusCode::NOT_FOUND,
            "<html>404</html>",
            "key",
        );
        assert!(not_found.contains("404"));
        assert!(not_found.contains("手动输入模型名"));
        assert!(!not_found.contains("key"));

        let unauthorized = describe_http_error(
            Protocol::Claude,
            reqwest::StatusCode::UNAUTHORIZED,
            "{}",
            "key",
        );
        assert!(unauthorized.contains("API Key"));
    }
}
