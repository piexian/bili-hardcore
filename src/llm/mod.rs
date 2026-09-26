pub mod models;
pub(crate) mod protocol;
pub mod prompt;
pub(crate) mod providers;
mod request;
pub(crate) mod shared;

pub use protocol::{Protocol, supports_thinking};
pub use request::QuizRequest;

use crate::config::LlmConfig;
use providers::{
    ClaudeClient, GeminiClient, GeminiInteractionsClient, JevClient, OpenAiChatClient,
    OpenAiResponsesClient,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// 适配器与 UI 之间的唯一契约：各协议把流式响应翻译成这四个事件，
/// 界面与答题状态机不需要知道任何协议细节。
#[derive(Debug)]
pub enum LlmChunk {
    Thinking(String),
    Content(String),
    Done(String),
    Error(String),
}

/// 协议分派。配置显式选择协议；旧配置缺该字段时按基址推断。
pub enum LlmClient {
    OpenAiChat(OpenAiChatClient),
    OpenAiResponses(OpenAiResponsesClient),
    Claude(ClaudeClient),
    Gemini(GeminiClient),
    GeminiInteractions(GeminiInteractionsClient),
    Jev(JevClient),
}

impl LlmClient {
    pub fn new(config: &LlmConfig) -> Self {
        match config.protocol() {
            Protocol::OpenAiChat => Self::OpenAiChat(OpenAiChatClient::new(config)),
            Protocol::OpenAiResponses => Self::OpenAiResponses(OpenAiResponsesClient::new(config)),
            Protocol::Claude => Self::Claude(ClaudeClient::new(config)),
            Protocol::Gemini => Self::Gemini(GeminiClient::new(config)),
            Protocol::GeminiInteractions => {
                Self::GeminiInteractions(GeminiInteractionsClient::new(config))
            }
            Protocol::Jev => Self::Jev(JevClient::new(config)),
        }
    }

    pub fn ask(
        &self,
        request: &QuizRequest,
        tx: mpsc::UnboundedSender<LlmChunk>,
        token: CancellationToken,
    ) {
        match self {
            Self::OpenAiChat(client) => client.ask(request, tx, token),
            Self::OpenAiResponses(client) => client.ask(request, tx, token),
            Self::Claude(client) => client.ask(request, tx, token),
            Self::Gemini(client) => client.ask(request, tx, token),
            Self::GeminiInteractions(client) => client.ask(request, tx, token),
            Self::Jev(client) => client.ask(request, tx, token),
        }
    }
}
