use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::llm::protocol::{self, Protocol};

const CONFIG_DIR_NAME: &str = ".bili-hardcore";

// --- Preset Templates ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetConfig {
    pub base_url: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetTemplate {
    /// 选择预设即选定协议，请求形态随之确定。
    pub protocol: Protocol,
    pub provider_name: String,
    pub config: PresetConfig,
}

const PRESETS_JSON: &str = include_str!("presets.json");

pub fn load_presets() -> Vec<PresetTemplate> {
    serde_json::from_str(PRESETS_JSON).unwrap_or_default()
}

/// LLM 连接配置。落盘文件名仍是历史遗留的 `openai_config.json`，
/// 现在覆盖全部协议，故以 LlmConfig 命名。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// 只存基址（`v1` 之前那节），完整端点由协议拼装。
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    /// 历史配置没有该字段，加载时按基址推断。
    #[serde(default)]
    pub protocol: Option<Protocol>,
    #[serde(default)]
    pub enable_thinking: bool,
    /// 思考模式强度（DeepSeek: low/high/max），默认 high
    #[serde(default = "default_reasoning_effort")]
    pub reasoning_effort: String,
    #[serde(default)]
    pub enable_fast_mode: bool,
}

impl LlmConfig {
    /// 显式选择的协议优先，缺失则按基址推断，兼容旧配置。
    pub fn protocol(&self) -> Protocol {
        self.protocol
            .unwrap_or_else(|| protocol::infer_protocol(&self.base_url))
    }
}

fn default_reasoning_effort() -> String {
    "high".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthData {
    pub access_token: String,
    pub csrf: String,
    pub mid: String,
    pub cookie: String,
}

fn config_dir() -> PathBuf {
    dirs::home_dir()
        .expect("无法获取用户主目录")
        .join(CONFIG_DIR_NAME)
}

pub fn openai_config_path() -> PathBuf {
    config_dir().join("openai_config.json")
}

pub fn auth_path() -> PathBuf {
    config_dir().join("auth.json")
}

pub fn ensure_config_dir() -> Result<()> {
    let dir = config_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).context("创建配置目录失败")?;
    }
    Ok(())
}

// --- OpenAI Config ---

/// 旧配置迁移：按原始 URL 推断协议并归一成基址。
fn migrate_legacy(mut config: LlmConfig) -> LlmConfig {
    if config.protocol.is_none() {
        let (inferred, base_url) = protocol::resolve(&config.base_url);
        config.protocol = Some(inferred);
        config.base_url = base_url;
    }
    config
}

pub fn load_openai_config() -> Result<Option<LlmConfig>> {
    let path = openai_config_path();
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path).context("读取 API 配置失败")?;
    let config: LlmConfig = serde_json::from_str(&content).context("解析 API 配置失败")?;
    // 旧配置存的是完整端点，读取时归一成基址，用户再次保存后即为新格式。
    Ok(Some(migrate_legacy(config)))
}

pub fn save_openai_config(config: &LlmConfig) -> Result<()> {
    ensure_config_dir()?;
    let path = openai_config_path();
    let content = serde_json::to_string_pretty(config).context("序列化 API 配置失败")?;
    fs::write(&path, content).context("写入 API 配置失败")?;
    Ok(())
}

// --- Auth ---

pub fn load_auth() -> Result<Option<AuthData>> {
    let path = auth_path();
    if !path.exists() {
        return Ok(None);
    }

    let metadata = fs::metadata(&path).context("读取认证文件元数据失败")?;
    let modified = metadata.modified().context("获取文件修改时间失败")?;
    let elapsed = modified.elapsed().unwrap_or_default();
    if elapsed.as_secs() > 7 * 24 * 3600 {
        return Ok(None);
    }

    let content = fs::read_to_string(&path).context("读取认证信息失败")?;
    let auth: AuthData = serde_json::from_str(&content).context("解析认证信息失败")?;
    Ok(Some(auth))
}

pub fn save_auth(auth: &AuthData) -> Result<()> {
    ensure_config_dir()?;
    let path = auth_path();
    let content = serde_json::to_string_pretty(auth).context("序列化认证信息失败")?;
    fs::write(&path, content).context("写入认证信息失败")?;
    Ok(())
}

pub fn delete_openai_config() -> Result<()> {
    let path = openai_config_path();
    if path.exists() {
        fs::remove_file(path).context("删除 API 配置失败")?;
    }
    Ok(())
}

pub fn delete_auth() -> Result<()> {
    let path = auth_path();
    if path.exists() {
        fs::remove_file(path).context("删除认证信息失败")?;
    }
    Ok(())
}

// --- Selected Categories ---

pub fn categories_path() -> PathBuf {
    config_dir().join("categories.json")
}

pub fn load_categories() -> Vec<String> {
    let path = categories_path();
    if !path.exists() {
        return vec![];
    }
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    serde_json::from_str(&content).unwrap_or_default()
}

pub fn save_categories(categories: &[String]) -> Result<()> {
    ensure_config_dir()?;
    let path = categories_path();
    let content = serde_json::to_string_pretty(categories).context("序列化分类失败")?;
    fs::write(&path, content).context("写入分类失败")?;
    Ok(())
}

// --- Quiz History ---

use crate::app::HistoryItem;

pub fn history_path() -> PathBuf {
    config_dir().join("history.json")
}

pub fn load_history() -> Vec<HistoryItem> {
    let path = history_path();
    if !path.exists() {
        return vec![];
    }
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    serde_json::from_str(&content).unwrap_or_default()
}

pub fn save_history(history: &[HistoryItem]) -> Result<()> {
    ensure_config_dir()?;
    let path = history_path();
    let content = serde_json::to_string_pretty(history).context("序列化答题记录失败")?;
    fs::write(&path, content).context("写入答题记录失败")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_json_parses_and_every_base_carries_its_version() {
        let presets = load_presets();
        assert!(!presets.is_empty());
        for preset in &presets {
            // 基址约定：写到版本段，适配器只追加端点路径。
            let last = preset.config.base_url.rsplit('/').next().unwrap_or_default();
            assert!(
                protocol::versioned_base(preset.protocol, &preset.config.base_url)
                    .ends_with(last),
                "{} 的基址应已带版本段: {}",
                preset.provider_name,
                preset.config.base_url
            );
            assert!(!preset.config.model.trim().is_empty());
        }
    }

    #[test]
    fn every_protocol_has_at_least_one_preset() {
        let presets = load_presets();
        for protocol in Protocol::ALL {
            assert!(
                presets.iter().any(|preset| preset.protocol == protocol),
                "{protocol:?} 缺少预设"
            );
        }
    }

    #[test]
    fn presets_build_their_documented_endpoints() {
        let presets = load_presets();
        let by_name = |name: &str| {
            presets
                .iter()
                .find(|preset| preset.provider_name == name)
                .unwrap_or_else(|| panic!("缺少预设 {name}"))
        };

        let jev = by_name("JEV (TypeSafe)");
        assert_eq!(jev.protocol, Protocol::Jev);
        assert_eq!(
            protocol::infer_protocol(&jev.config.base_url),
            Protocol::Jev,
            "预设基址必须能被识别为 JEV 端点"
        );

        let gemini = by_name("Gemini 3");
        assert_eq!(gemini.protocol, Protocol::Gemini);
        assert!(
            protocol::endpoint(Protocol::Gemini, &gemini.config.base_url, &gemini.config.model)
                .ends_with(":streamGenerateContent?alt=sse"),
            "Gemini 预设应拼出 SSE 流式端点"
        );
    }

    #[test]
    fn legacy_config_without_protocol_resolves_by_base_url() {
        let legacy = serde_json::json!({
            "base_url": "https://api.typesafe.ai/v1/systemone",
            "model": "jev-latest",
            "api_key": "k"
        });
        let config: LlmConfig = serde_json::from_value(legacy).expect("旧配置应可解析");
        assert_eq!(config.protocol, None);
        assert_eq!(config.protocol(), Protocol::Jev);
        assert_eq!(
            protocol::normalize_base(&config.base_url),
            "https://api.typesafe.ai"
        );

        let chat = serde_json::json!({
            "base_url": "https://api.x.ai/v1/chat/completions",
            "model": "grok-4.6",
            "api_key": "k"
        });
        let config: LlmConfig = serde_json::from_value(chat).expect("旧配置应可解析");
        assert_eq!(config.protocol(), Protocol::OpenAiChat);
        assert_eq!(
            protocol::endpoint(Protocol::OpenAiChat, &config.base_url, &config.model),
            "https://api.x.ai/v1/chat/completions"
        );
    }

    #[test]
    fn legacy_jev_relay_keeps_the_jev_protocol() {
        // 走中转的 JEV：端点特征在 /systemone 上，先归一再推断会退化成 Chat Completions。
        let legacy = serde_json::json!({
            "base_url": "http://192.168.0.2:3000/v1/systemone",
            "model": "jev-1.13.0",
            "api_key": "k"
        });
        let config = migrate_legacy(serde_json::from_value(legacy).expect("旧配置应可解析"));
        assert_eq!(config.protocol(), Protocol::Jev);
        assert_eq!(config.base_url, "http://192.168.0.2:3000");
        assert_eq!(
            protocol::endpoint(Protocol::Jev, &config.base_url, &config.model),
            "http://192.168.0.2:3000/v1/systemone"
        );
    }

    #[test]
    fn explicit_protocol_wins_over_url_inference() {
        let config = LlmConfig {
            base_url: "https://relay.example.com".to_string(),
            model: "m".to_string(),
            api_key: "k".to_string(),
            protocol: Some(Protocol::Claude),
            enable_thinking: true,
            reasoning_effort: "high".to_string(),
            enable_fast_mode: false,
        };
        assert_eq!(config.protocol(), Protocol::Claude);
    }
}
