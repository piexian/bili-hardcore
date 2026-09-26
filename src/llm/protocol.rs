use serde::{Deserialize, Serialize};

/// 配置页与配置里选择的目标协议。每种协议的端点、鉴权、请求体、流式事件都不同，
/// 各自由 providers/ 下的适配器实现。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Protocol {
    /// OpenAI Chat Completions，同时覆盖 xAI / DeepSeek / GLM 等兼容方言。
    #[default]
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    /// OpenAI Responses。
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    /// Claude Messages 原生。
    #[serde(rename = "claude")]
    Claude,
    /// Gemini generateContent 原生。
    #[serde(rename = "gemini")]
    Gemini,
    /// Gemini Interactions API（单端点、携带服务端交互状态）。
    #[serde(rename = "gemini_interactions")]
    GeminiInteractions,
    /// JEV /systemone 结构化决策。
    #[serde(rename = "jev")]
    Jev,
}

impl Protocol {
    pub const ALL: [Protocol; 6] = [
        Protocol::OpenAiChat,
        Protocol::OpenAiResponses,
        Protocol::Claude,
        Protocol::Gemini,
        Protocol::GeminiInteractions,
        Protocol::Jev,
    ];

    pub fn display_name(self) -> &'static str {
        match self {
            Self::OpenAiChat => "Chat Completions",
            Self::OpenAiResponses => "Responses",
            Self::Claude => "Claude Messages",
            Self::Gemini => "Gemini",
            Self::GeminiInteractions => "Gemini Interactions",
            Self::Jev => "JEV",
        }
    }

    pub fn next(self) -> Self {
        let index = Self::ALL.iter().position(|item| *item == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }

    pub fn previous(self) -> Self {
        let index = Self::ALL.iter().position(|item| *item == self).unwrap_or(0);
        Self::ALL[(index + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

/// JEV 直接返回选项概率分布，没有思考开关。
pub fn supports_thinking(protocol: Protocol) -> bool {
    protocol != Protocol::Jev
}
const ENDPOINT_SUFFIXES: [&str; 4] = [
    "/chat/completions",
    "/responses",
    "/messages",
    "/systemone",
];
const VERSION_SUFFIXES: [&str; 2] = ["/v1beta", "/v1"];

fn strip_version_segment(base: &str) -> String {
    let lower = base.to_ascii_lowercase();
    match VERSION_SUFFIXES.iter().find(|item| lower.ends_with(**item)) {
        // to_ascii_lowercase 只映射 ASCII，字节长度不变，按字节截断安全。
        Some(version) => base[..base.len() - version.len()].to_string(),
        None => base.to_string(),
    }
}

/// 配置统一只存基址（`v1` 之前那节）。历史配置里存的是完整端点，读取时剥掉
/// 已知端点后缀及其紧邻的版本段，保存后即为基址，用户无感。
pub fn normalize_base(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    // Gemini 端点自带模型名与 :method 后缀，反解时从 /models/ 截断。
    for method in [":streamGenerateContent", ":generateContent", ":countTokens"] {
        if let Some(method_at) = trimmed.find(method)
            && let Some(models_at) = trimmed[..method_at].rfind("/models/")
        {
            return strip_version_segment(&trimmed[..models_at]);
        }
    }
    let lower = trimmed.to_ascii_lowercase();
    let Some(suffix) = ENDPOINT_SUFFIXES.iter().find(|item| lower.ends_with(**item)) else {
        return trimmed.to_string();
    };
    strip_version_segment(&trimmed[..trimmed.len() - suffix.len()])
}

/// 版本段由用户在基址里自己指定：OpenAI 兼容的服务未必是 v1（可能是 v2/v3，
/// 智谱是 /api/paas/v4，中转站也常挂在 /v1 下），所以适配器不猜版本，
/// 只负责在基址后面追加端点路径。
fn has_version_segment(base: &str) -> bool {
    let Some(last) = base.rsplit('/').next() else {
        return false;
    };
    let Some(rest) = last.strip_prefix(['v', 'V']) else {
        return false;
    };
    // v1 / v2 / v1beta / v2beta 都算版本段（Gemini 就是 v1beta）；
    // version、models 这类不以数字开头的目录不算。
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    digits > 0 && rest[digits..].bytes().all(|byte| byte.is_ascii_alphabetic())
}

/// 基址没写版本段时才补的默认值，收敛到这一处。
fn default_version(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Gemini | Protocol::GeminiInteractions => "/v1beta",
        _ => "/v1",
    }
}

/// 基址归一到"一定带版本段"：写了就用，没写才补默认。
pub fn versioned_base(protocol: Protocol, base_url: &str) -> String {
    let base = normalize_base(base_url);
    if has_version_segment(&base) {
        base
    } else {
        format!("{base}{}", default_version(protocol))
    }
}

/// 由基址拼出完整端点。Gemini 的模型名进 URL 路径，故需要 model。
pub fn endpoint(protocol: Protocol, base_url: &str, model: &str) -> String {
    let base = versioned_base(protocol, base_url);
    match protocol {
        Protocol::OpenAiChat => format!("{base}/chat/completions"),
        Protocol::OpenAiResponses => format!("{base}/responses"),
        Protocol::Claude => format!("{base}/messages"),
        Protocol::Gemini => format!(
            "{base}/models/{}:streamGenerateContent?alt=sse",
            urlencoding::encode(model)
        ),
        Protocol::GeminiInteractions => format!("{base}/interactions"),
        Protocol::Jev => format!("{base}/systemone"),
    }
}

/// 从历史完整端点迁移：必须先按原 URL 推断协议，再归一成基址。
/// 顺序不能反——归一后 `/systemone` 之类的端点特征就没了，
/// 走中转的 JEV（例如 `http://192.168.0.2:3000/v1/systemone`）会被误判成 Chat Completions。
pub fn resolve(base_url: &str) -> (Protocol, String) {
    (infer_protocol(base_url), normalize_base(base_url))
}

/// 仅用于旧配置迁移与 CLI 快速启动：按基址或历史完整端点猜协议。
/// 用户在配置页显式选择的协议不走这里。
pub fn infer_protocol(base_url: &str) -> Protocol {
    if crate::llm::providers::is_jev_endpoint(base_url) {
        return Protocol::Jev;
    }
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return Protocol::OpenAiChat;
    };
    let host = url.host_str().unwrap_or_default();
    let path = url.path().trim_end_matches('/').to_ascii_lowercase();

    if host.eq_ignore_ascii_case("generativelanguage.googleapis.com") {
        return Protocol::Gemini;
    }
    if host.eq_ignore_ascii_case("api.anthropic.com") || path.ends_with("/messages") {
        return Protocol::Claude;
    }
    if path.ends_with("/responses") {
        return Protocol::OpenAiResponses;
    }
    Protocol::OpenAiChat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_known_endpoints_and_their_version_segment() {
        assert_eq!(
            normalize_base("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com"
        );
        assert_eq!(
            normalize_base("https://api.openai.com/v1/responses/"),
            "https://api.openai.com"
        );
        assert_eq!(
            normalize_base("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_base("https://api.typesafe.ai/v1/systemone"),
            "https://api.typesafe.ai"
        );
        assert_eq!(
            normalize_base("https://generativelanguage.googleapis.com/v1beta/models/x:generateContent"),
            "https://generativelanguage.googleapis.com"
        );
    }

    #[test]
    fn keeps_bare_bases_and_unrelated_paths() {
        assert_eq!(normalize_base("https://api.openai.com"), "https://api.openai.com");
        assert_eq!(normalize_base("  https://api.openai.com//  "), "https://api.openai.com");
        assert_eq!(
            normalize_base("https://relay.example.com/typesafe/systemone"),
            "https://relay.example.com/typesafe"
        );
        assert_eq!(normalize_base("not a url"), "not a url");
    }

    #[test]
    fn version_in_base_is_used_verbatim_for_every_protocol() {
        // 版本由用户决定，适配器不再猜：OpenAI 标准的也可能是 v2/v3。
        assert_eq!(
            endpoint(Protocol::OpenAiChat, "https://relay.example.com/v2", "m"),
            "https://relay.example.com/v2/chat/completions"
        );
        assert_eq!(
            endpoint(Protocol::OpenAiResponses, "https://relay.example.com/v3", "m"),
            "https://relay.example.com/v3/responses"
        );
        assert_eq!(
            endpoint(Protocol::Claude, "https://relay.example.com/v1", "m"),
            "https://relay.example.com/v1/messages"
        );
        assert_eq!(
            endpoint(Protocol::Jev, "http://192.168.0.2:3000/v1", "m"),
            "http://192.168.0.2:3000/v1/systemone"
        );
        assert_eq!(
            endpoint(
                Protocol::GeminiInteractions,
                "https://generativelanguage.googleapis.com/v1beta",
                "m"
            ),
            "https://generativelanguage.googleapis.com/v1beta/interactions"
        );
        // 智谱把版本写在中转路径里
        assert_eq!(
            endpoint(
                Protocol::OpenAiChat,
                "https://open.bigmodel.cn/api/paas/v4",
                "glm-4.7"
            ),
            "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        );
    }

    #[test]
    fn missing_version_falls_back_to_the_protocol_default() {
        assert_eq!(
            versioned_base(Protocol::OpenAiChat, "https://api.openai.com"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            versioned_base(Protocol::OpenAiResponses, "https://api.x.ai"),
            "https://api.x.ai/v1"
        );
        assert_eq!(
            versioned_base(Protocol::Claude, "https://api.anthropic.com"),
            "https://api.anthropic.com/v1"
        );
        assert_eq!(
            versioned_base(Protocol::Jev, "https://api.typesafe.ai"),
            "https://api.typesafe.ai/v1"
        );
        // Gemini 家族是 v1beta，不是 v1
        assert_eq!(
            versioned_base(Protocol::Gemini, "https://generativelanguage.googleapis.com"),
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert_eq!(
            versioned_base(
                Protocol::GeminiInteractions,
                "https://generativelanguage.googleapis.com"
            ),
            "https://generativelanguage.googleapis.com/v1beta"
        );
        // 写了的版本不被覆盖
        assert_eq!(
            versioned_base(Protocol::Gemini, "https://host.example/v1beta"),
            "https://host.example/v1beta"
        );
    }

    #[test]
    fn builds_endpoints_from_base_only() {
        assert_eq!(
            endpoint(Protocol::OpenAiChat, "https://api.openai.com", "gpt-test"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            endpoint(Protocol::OpenAiResponses, "https://api.openai.com/", "gpt-test"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            endpoint(Protocol::Claude, "https://api.anthropic.com", "claude-sonnet-5"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            endpoint(Protocol::Gemini, "https://generativelanguage.googleapis.com", "gemini-3-flash-preview"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            endpoint(Protocol::Jev, "https://api.typesafe.ai", "jev-latest"),
            "https://api.typesafe.ai/v1/systemone"
        );
    }

    #[test]
    fn gemini_endpoint_escapes_the_model_path_segment() {
        assert_eq!(
            endpoint(Protocol::Gemini, "https://generativelanguage.googleapis.com", "models/x y"),
            "https://generativelanguage.googleapis.com/v1beta/models/models%2Fx%20y:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn endpoint_round_trips_legacy_full_urls() {
        for (protocol, full) in [
            (Protocol::OpenAiChat, "https://api.openai.com/v1/chat/completions"),
            (Protocol::OpenAiResponses, "https://api.openai.com/v1/responses"),
            (Protocol::Claude, "https://api.anthropic.com/v1/messages"),
            (Protocol::Jev, "https://api.typesafe.ai/v1/systemone"),
        ] {
            assert_eq!(endpoint(protocol, full, "model"), full);
        }
    }

    #[test]
    fn infers_protocol_from_legacy_configs() {
        assert_eq!(
            infer_protocol("https://api.typesafe.ai/v1/systemone"),
            Protocol::Jev
        );
        assert_eq!(
            infer_protocol("https://relay.example.com/typesafe/systemone"),
            Protocol::Jev
        );
        assert_eq!(
            infer_protocol("https://api.anthropic.com/v1/messages"),
            Protocol::Claude
        );
        assert_eq!(
            infer_protocol("https://api.openai.com/v1/responses"),
            Protocol::OpenAiResponses
        );
        assert_eq!(
            infer_protocol("https://generativelanguage.googleapis.com"),
            Protocol::Gemini
        );
        assert_eq!(
            infer_protocol("https://api.openai.com/v1/chat/completions"),
            Protocol::OpenAiChat
        );
        assert_eq!(
            infer_protocol("https://api.siliconflow.cn/v1/chat/completions"),
            Protocol::OpenAiChat
        );
        assert_eq!(infer_protocol("not a url"), Protocol::OpenAiChat);
    }

    #[test]
    fn relay_endpoints_resolve_before_the_base_is_normalized() {
        // 中转的 JEV：/systemone 是唯一的识别特征，归一后就没了，顺序不能反。
        let (protocol, base) = resolve("http://192.168.0.2:3000/v1/systemone");
        assert_eq!(protocol, Protocol::Jev);
        assert_eq!(base, "http://192.168.0.2:3000");
        assert_eq!(
            endpoint(protocol, &base, "jev-latest"),
            "http://192.168.0.2:3000/v1/systemone"
        );

        // 裸基址本身推不出 JEV，必须显式选协议，这也是配置页要有协议选择行的原因。
        assert_eq!(infer_protocol(&base), Protocol::OpenAiChat);
    }

    #[test]
    fn only_jev_lacks_a_thinking_switch() {
        assert!(supports_thinking(Protocol::OpenAiChat));
        assert!(supports_thinking(Protocol::OpenAiResponses));
        assert!(supports_thinking(Protocol::Claude));
        assert!(supports_thinking(Protocol::Gemini));
        assert!(!supports_thinking(Protocol::Jev));
    }

    #[test]
    fn protocol_cycles_through_every_adapter() {
        let mut protocol = Protocol::OpenAiChat;
        for _ in 0..Protocol::ALL.len() {
            protocol = protocol.next();
        }
        assert_eq!(protocol, Protocol::OpenAiChat);
        assert_eq!(Protocol::OpenAiChat.previous(), Protocol::Jev);
    }
}
