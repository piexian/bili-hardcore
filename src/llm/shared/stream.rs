use super::errors::extract_api_error;
use super::http::{format_reqwest_error, safe_error_chain, safe_preview};
use bytes::Bytes;
use eventsource_stream::{Event, EventStreamError, Eventsource};
use futures::{Stream, StreamExt, stream};
use reqwest::{
    Response,
    header::{CONTENT_ENCODING, CONTENT_TYPE},
};
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

const MAX_RESPONSE_SNIFF_SIZE: usize = 8 * 1024;
const UTF8_BOM: &[u8] = b"\xef\xbb\xbf";
const SSE_PREFIXES: [&[u8]; 4] = [b"data:", b"event:", b"id:", b"retry:"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseFormat {
    Sse,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixDecision {
    Detected(ResponseFormat),
    NeedMore,
    NoMatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SniffStop {
    Mismatch,
    EndOfResponse,
    Limit,
}

pub(crate) type SseEvents =
    Pin<Box<dyn Stream<Item = Result<Event, EventStreamError<reqwest::Error>>> + Send>>;

/// 嗅探后的响应体：SSE 保持增量流式，JSON 收齐后交给适配器整体解析。
/// 上游即使收到 `stream: true` 也可能回 JSON（超时、网关、错误），因此不能假定格式。
pub(crate) enum BodyStream {
    Sse(SseEvents),
    Json(Vec<u8>),
}

fn inspect_body_signature(prefix: &[u8]) -> PrefixDecision {
    let mut body = prefix;
    if body.starts_with(UTF8_BOM) {
        body = &body[UTF8_BOM.len()..];
    } else if !body.is_empty() && body.len() < UTF8_BOM.len() && UTF8_BOM.starts_with(body) {
        return PrefixDecision::NeedMore;
    }

    body = body.trim_ascii_start();
    let Some(first) = body.first() else {
        return PrefixDecision::NeedMore;
    };
    if matches!(first, b'{' | b'[') {
        return PrefixDecision::Detected(ResponseFormat::Json);
    }
    if *first == b':' {
        return PrefixDecision::Detected(ResponseFormat::Sse);
    }
    for marker in SSE_PREFIXES {
        if body.starts_with(marker) {
            return PrefixDecision::Detected(ResponseFormat::Sse);
        }
        if body.len() < marker.len() && marker.starts_with(body) {
            return PrefixDecision::NeedMore;
        }
    }
    PrefixDecision::NoMatch
}

fn response_format_from_header(content_type: Option<&str>) -> Option<ResponseFormat> {
    let media_type = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .map(str::to_ascii_lowercase);
    match media_type.as_deref() {
        Some("text/event-stream") => Some(ResponseFormat::Sse),
        Some("application/json") => Some(ResponseFormat::Json),
        Some(value) if value.starts_with("application/") && value.ends_with("+json") => {
            Some(ResponseFormat::Json)
        }
        _ => None,
    }
}

fn strip_utf8_bom_from_chunks(chunks: &mut [Bytes]) {
    let mut remaining = UTF8_BOM.len();
    for chunk in chunks {
        let skipped = remaining.min(chunk.len());
        *chunk = chunk.slice(skipped..);
        remaining -= skipped;
        if remaining == 0 {
            break;
        }
    }
}

fn single_line_preview(message: &str, api_key: &str, max_chars: usize) -> String {
    safe_preview(message, api_key, max_chars)
        .chars()
        .flat_map(char::escape_debug)
        .collect()
}

fn has_unsupported_content_encoding(content_encoding: Option<&str>) -> bool {
    content_encoding.is_some_and(|value| {
        value
            .split(',')
            .map(str::trim)
            .any(|encoding| !encoding.is_empty() && !encoding.eq_ignore_ascii_case("identity"))
    })
}

fn fallback_response_format(
    content_type: Option<&str>,
    content_encoding: Option<&str>,
    prefix: &[u8],
    stop: SniffStop,
    api_key: &str,
) -> Result<ResponseFormat, String> {
    if has_unsupported_content_encoding(content_encoding) {
        return Err(unrecognized_format_error(
            content_type,
            content_encoding,
            prefix,
            stop,
            api_key,
        ));
    }
    response_format_from_header(content_type).ok_or_else(|| {
        unrecognized_format_error(content_type, content_encoding, prefix, stop, api_key)
    })
}

fn unrecognized_format_error(
    content_type: Option<&str>,
    content_encoding: Option<&str>,
    prefix: &[u8],
    stop: SniffStop,
    api_key: &str,
) -> String {
    let unsupported_encoding = has_unsupported_content_encoding(content_encoding);
    let content_type = single_line_preview(content_type.unwrap_or("<缺失>"), api_key, 100);
    let content_encoding = single_line_preview(content_encoding.unwrap_or("<缺失>"), api_key, 100);
    let preview = single_line_preview(&String::from_utf8_lossy(prefix), api_key, 200);
    let stop = match stop {
        SniffStop::Mismatch => "正文前缀不匹配 JSON/SSE",
        SniffStop::EndOfResponse => "响应已结束",
        SniffStop::Limit => "已达到 8 KiB 探测上限",
    };
    if unsupported_encoding {
        format!(
            "无法识别 LLM 响应格式，响应可能仍使用不支持的 Content-Encoding; Content-Type={content_type}; Content-Encoding={content_encoding}; {stop}; 响应摘要={preview}"
        )
    } else {
        format!(
            "无法识别 LLM 响应格式; Content-Type={content_type}; Content-Encoding={content_encoding}; {stop}; 响应摘要={preview}"
        )
    }
}

pub(crate) fn format_event_stream_error(
    error: &EventStreamError<reqwest::Error>,
    label: &str,
    api_key: &str,
) -> String {
    match error {
        EventStreamError::Transport(error) => {
            format_reqwest_error(&format!("读取 {label} SSE 响应失败"), error, api_key)
        }
        _ => format!(
            "读取 {label} SSE 响应失败 [响应解码]: {}",
            safe_error_chain(error, api_key)
        ),
    }
}

/// 非 2xx 响应统一转成给用户看的错误：能提取到结构化 message 就用 message，
/// 否则退回脱敏摘要。返回 `None` 表示任务已取消。
pub(crate) async fn read_error_message(
    response: Response,
    label: &str,
    api_key: &str,
    token: &CancellationToken,
) -> Option<String> {
    let status = response.status();
    let body = tokio::select! {
        biased;
        _ = token.cancelled() => return None,
        result = response.text() => match result {
            Ok(body) => body,
            Err(error) => {
                return Some(format_reqwest_error(
                    &format!("读取 {label} 错误响应失败"),
                    &error,
                    api_key,
                ));
            }
        }
    };

    let message = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| extract_api_error(&value));
    Some(match message {
        Some(message) => format!("{label} 请求失败 (HTTP {status}): {message}"),
        None => format!(
            "{label} 请求失败 (HTTP {status}): {}",
            safe_preview(&body, api_key, 300)
        ),
    })
}

/// 读掉响应前缀判定 JSON/SSE：SSE 把已嗅探的分片接回流头继续增量消费，
/// JSON 收齐后返回。`Ok(None)` 是任务已取消，`Err` 是给用户看的格式/读取错误。
pub(crate) async fn open_body_stream(
    mut response: Response,
    label: &str,
    api_key: &str,
    max_json_bytes: usize,
    token: &CancellationToken,
) -> Result<Option<BodyStream>, String> {
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let content_encoding = response
        .headers()
        .get(CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);

    let mut sniff_prefix = Vec::new();
    let mut buffered_chunks = Vec::new();
    let (format, stop) = loop {
        match inspect_body_signature(&sniff_prefix) {
            PrefixDecision::Detected(format) => break (Some(format), None),
            PrefixDecision::NoMatch => break (None, Some(SniffStop::Mismatch)),
            PrefixDecision::NeedMore if sniff_prefix.len() >= MAX_RESPONSE_SNIFF_SIZE => {
                break (None, Some(SniffStop::Limit));
            }
            PrefixDecision::NeedMore => {}
        }

        let chunk = tokio::select! {
            biased;
            _ = token.cancelled() => return Ok(None),
            result = response.chunk() => result,
        };
        match chunk {
            Ok(Some(chunk)) => {
                let remaining = MAX_RESPONSE_SNIFF_SIZE - sniff_prefix.len();
                let inspected = remaining.min(chunk.len());
                sniff_prefix.extend_from_slice(&chunk[..inspected]);
                buffered_chunks.push(chunk);
            }
            Ok(None) => break (None, Some(SniffStop::EndOfResponse)),
            Err(error) => {
                return Err(format_reqwest_error(
                    &format!("读取 {label} 响应失败"),
                    &error,
                    api_key,
                ));
            }
        }
    };

    let format = match format {
        Some(format) => format,
        None => fallback_response_format(
            content_type.as_deref(),
            content_encoding.as_deref(),
            &sniff_prefix,
            stop.expect("未识别格式必须包含停止原因"),
            api_key,
        )?,
    };
    if sniff_prefix.starts_with(UTF8_BOM) {
        strip_utf8_bom_from_chunks(&mut buffered_chunks);
    }

    match format {
        ResponseFormat::Sse => {
            let replay =
                stream::iter(buffered_chunks.into_iter().map(Ok::<_, reqwest::Error>));
            let events = replay.chain(response.bytes_stream()).eventsource();
            Ok(Some(BodyStream::Sse(Box::pin(events))))
        }
        ResponseFormat::Json => {
            let mut bytes = Vec::new();
            for chunk in buffered_chunks {
                bytes.extend_from_slice(&chunk);
            }
            loop {
                if bytes.len() > max_json_bytes {
                    return Err(format!(
                        "{label} JSON 响应超过 {} MiB 限制",
                        max_json_bytes / 1024 / 1024
                    ));
                }
                let chunk = tokio::select! {
                    biased;
                    _ = token.cancelled() => return Ok(None),
                    result = response.chunk() => result,
                };
                match chunk {
                    Ok(Some(chunk)) => bytes.extend_from_slice(&chunk),
                    Ok(None) => break,
                    Err(error) => {
                        return Err(format_reqwest_error(
                            &format!("读取 {label} JSON 响应失败"),
                            &error,
                            api_key,
                        ));
                    }
                }
            }
            Ok(Some(BodyStream::Json(bytes)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_complete_json_and_sse_signatures() {
        assert_eq!(
            inspect_body_signature(br#"{"ok":true}"#),
            PrefixDecision::Detected(ResponseFormat::Json)
        );
        assert_eq!(
            inspect_body_signature(b" \r\ndata: {}\n\n"),
            PrefixDecision::Detected(ResponseFormat::Sse)
        );
        assert_eq!(
            inspect_body_signature(b"event: message\n"),
            PrefixDecision::Detected(ResponseFormat::Sse)
        );
        assert_eq!(
            inspect_body_signature(b": keep-alive\n"),
            PrefixDecision::Detected(ResponseFormat::Sse)
        );
    }

    #[test]
    fn waits_for_split_bom_whitespace_and_sse_markers() {
        for prefix in [
            &b""[..],
            &b" \r\n"[..],
            &b"\xef"[..],
            &b"\xef\xbb"[..],
            &b"d"[..],
            &b"data"[..],
            &b"ev"[..],
            &b"retr"[..],
        ] {
            assert_eq!(inspect_body_signature(prefix), PrefixDecision::NeedMore);
        }
        assert_eq!(
            inspect_body_signature(b"\xef\xbb\xbf \r\ndata:"),
            PrefixDecision::Detected(ResponseFormat::Sse)
        );
        assert_eq!(
            inspect_body_signature(b"\xef\xbb\xbf [1]"),
            PrefixDecision::Detected(ResponseFormat::Json)
        );
        assert_eq!(inspect_body_signature(b"<html>"), PrefixDecision::NoMatch);
    }

    #[test]
    fn body_signatures_override_headers_and_headers_are_strict_fallbacks() {
        assert_eq!(
            inspect_body_signature(b"data: {}\n\n"),
            PrefixDecision::Detected(ResponseFormat::Sse)
        );
        assert_eq!(
            response_format_from_header(Some("application/json; charset=utf-8")),
            Some(ResponseFormat::Json)
        );
        assert_eq!(
            response_format_from_header(Some(" Application/Problem+Json ; charset=utf-8")),
            Some(ResponseFormat::Json)
        );
        assert_eq!(
            response_format_from_header(Some("TEXT/EVENT-STREAM")),
            Some(ResponseFormat::Sse)
        );
        assert_eq!(response_format_from_header(Some("text/plain")), None);
        assert_eq!(response_format_from_header(None), None);
        assert_eq!(
            fallback_response_format(
                Some("application/json"),
                None,
                b" ",
                SniffStop::EndOfResponse,
                "key"
            ),
            Ok(ResponseFormat::Json)
        );
        assert!(
            fallback_response_format(
                Some("application/json"),
                Some("zstd"),
                b"binary",
                SniffStop::Mismatch,
                "key"
            )
            .is_err()
        );
        assert!(
            fallback_response_format(
                Some("text/plain"),
                None,
                b"plain text",
                SniffStop::Mismatch,
                "key"
            )
            .is_err()
        );
    }

    #[test]
    fn strips_only_one_leading_utf8_bom_across_chunks() {
        let mut chunks = vec![
            Bytes::from_static(b"\xef"),
            Bytes::from_static(b"\xbb"),
            Bytes::from_static(b"\xbf {\"choices\":[]}"),
        ];
        strip_utf8_bom_from_chunks(&mut chunks);
        let json: Vec<u8> = chunks
            .iter()
            .flat_map(|chunk| chunk.iter().copied())
            .collect();
        assert_eq!(json, b" {\"choices\":[]}");
        assert!(serde_json::from_slice::<serde_json::Value>(&json).is_ok());
    }

    #[test]
    fn unrecognized_format_diagnostics_are_redacted_and_actionable() {
        let key = "secret-test-key";
        let error = unrecognized_format_error(
            Some("text/plain"),
            Some("zstd"),
            format!("<html>Bearer {key}\nblocked</html>").as_bytes(),
            SniffStop::Mismatch,
            key,
        );
        assert!(error.contains("不支持的 Content-Encoding"));
        assert!(error.contains("Content-Type=text/plain"));
        assert!(error.contains("Content-Encoding=zstd"));
        assert!(error.contains("\\n"));
        assert!(!error.contains(key));
    }
}
