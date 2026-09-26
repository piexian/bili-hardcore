use reqwest::RequestBuilder;

use crate::llm::protocol::Protocol;

/// Claude 所有接口都要求这个版本头，缺失直接 400。
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";

/// 鉴权只有这一处定义：答题适配器与模型列表拉取共用，
/// 以后新增协议不会出现"改了 A 忘了 B"。
pub(crate) fn apply_auth(
    request: RequestBuilder,
    protocol: Protocol,
    api_key: &str,
) -> RequestBuilder {
    match protocol {
        Protocol::OpenAiChat | Protocol::OpenAiResponses | Protocol::Jev => {
            request.header("Authorization", format!("Bearer {api_key}"))
        }
        Protocol::Claude => request
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION),
        Protocol::Gemini | Protocol::GeminiInteractions => {
            request.header("x-goog-api-key", api_key)
        }
    }
}
