use crate::app::{App, ConfigFocus};
use crate::config;
use ratatui::style::{Color, Modifier, Style};

const LABELS: [&str; 3] = ["API URL", "模型名称", "API Key"];
fn focus_index(focus: ConfigFocus) -> usize {
    match focus {
        ConfigFocus::BaseUrl => 0,
        ConfigFocus::Model => 1,
        ConfigFocus::ApiKey => 2,
        ConfigFocus::Protocol => 3,
        ConfigFocus::ThinkingToggle => 4,
        ConfigFocus::ThinkingEffort => 5,
        ConfigFocus::FastModeToggle => 6,
        ConfigFocus::SaveBtn => 7,
        ConfigFocus::TemplateBtn => 8,
        ConfigFocus::ResetBtn => 9,
    }
}

fn selected_style(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

fn dim_style(color: Color) -> Style {
    Style::default().fg(color)
}

pub fn draw(f: &mut ratatui::Frame, app: &App) {
    use ratatui::{
        layout::{Alignment, Constraint, Layout},
        style::{Color, Style},
        widgets::{Block, Borders, Paragraph},
    };

    let size = f.area();
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" API 配置 ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(size);
    f.render_widget(block, size);

    // Preset selection overlay
    if app.cfg_preset_open {
        draw_preset_select(f, inner, app.cfg_preset_sel);
        return;
    }

    // Model picker overlay
    if app.model_picker_open {
        draw_model_select(f, inner, app);
        return;
    }

    // Confirmation dialog overlay
    if app.config_confirm_reset {
        draw_reset_confirm(f, inner, app.config_reset_choice);
        return;
    }

    let focus = focus_index(app.cfg_focus);
    // JEV 直接返回选项概率分布，没有推理开关可调
    let supports_thinking = crate::llm::supports_thinking(app.cfg_protocol);
    let thinking_on = app.thinking_visible();

    let mut layout_constraints: Vec<Constraint> = vec![
        Constraint::Length(2), // header
        Constraint::Length(3), // BaseUrl
        Constraint::Length(3), // Model
        Constraint::Length(3), // ApiKey
        Constraint::Length(3), // Protocol
        Constraint::Length(3), // Thinking toggle
    ];
    if thinking_on {
        layout_constraints.push(Constraint::Length(3)); // Thinking effort
    }
    layout_constraints.extend_from_slice(&[
        Constraint::Length(3), // Fast mode toggle
        Constraint::Length(2), // Save
        Constraint::Length(2), // Template
        Constraint::Length(2), // Reset
        Constraint::Min(1),    // spacer
        Constraint::Length(1), // help
    ]);

    let chunks = Layout::vertical(layout_constraints).split(inner);
    let effort_shift: usize = if thinking_on { 0 } else { 1 };

    f.render_widget(
        Paragraph::new("请输入 API 配置信息（URL 写到版本段，如 https://api.openai.com/v1）")
            .style(Style::default().fg(Color::Yellow))
            .alignment(Alignment::Center),
        chunks[0],
    );

    for (i, label) in LABELS.iter().enumerate() {
        let is_focused = focus == i;
        let border_color = if is_focused {
            Color::Cyan
        } else {
            Color::DarkGray
        };
        let field_block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" {} ", label))
            .style(Style::default().fg(border_color));
        let fi = field_block.inner(chunks[i + 1]);
        f.render_widget(field_block, chunks[i + 1]);

        let val = &app.cfg_fields[i];
        let display = if i == 2 && !val.is_empty() && !is_focused {
            "*".repeat(val.len().min(20))
        } else {
            val.clone()
        };

        let mut text = display;
        if is_focused {
            let char_count = text.chars().count();
            let pos = app.cfg_cursors[i].min(char_count);
            let byte_pos = text.char_indices().nth(pos).map_or(text.len(), |(i, _)| i);
            text.insert(byte_pos, '|');
        }
        f.render_widget(
            Paragraph::new(text).style(Style::default().fg(Color::White)),
            fi,
        );
    }

    // Protocol selector (chunks[4])：同一基址可能对应多种协议，必须显式选择
    let protocol_focused = app.cfg_focus == ConfigFocus::Protocol;
    let protocol_border_color = if protocol_focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let protocol_block = Block::default()
        .borders(Borders::ALL)
        .title(" 接口协议 ")
        .style(Style::default().fg(protocol_border_color));
    let protocol_inner = protocol_block.inner(chunks[4]);
    f.render_widget(protocol_block, chunks[4]);

    let protocol_text = format!(
        "[ {} ]  ← → 切换",
        app.cfg_protocol.display_name()
    );
    f.render_widget(
        Paragraph::new(protocol_text).style(Style::default().fg(if protocol_focused {
            Color::White
        } else {
            Color::DarkGray
        })),
        protocol_inner,
    );

    // Thinking toggle (chunks[5])
    let thinking_focused = app.cfg_focus == ConfigFocus::ThinkingToggle && supports_thinking;
    let toggle_border_color = if thinking_focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let toggle_block = Block::default()
        .borders(Borders::ALL)
        .title(" 思考模式 ")
        .style(Style::default().fg(toggle_border_color));
    let toggle_inner = toggle_block.inner(chunks[5]);
    f.render_widget(toggle_block, chunks[5]);

    let toggle_text = if !supports_thinking {
        "[ - ] JEV 为结构化决策模型，直接返回选项概率，无此开关".to_string()
    } else if app.cfg_thinking {
        "[✓] 开启 - 准确率高，速度慢".to_string()
    } else {
        "[ ] 关闭/最低 - 速度优先（部分模型仍会低度推理）".to_string()
    };
    let toggle_color = if thinking_focused {
        Color::White
    } else {
        Color::DarkGray
    };
    f.render_widget(
        Paragraph::new(toggle_text).style(Style::default().fg(toggle_color)),
        toggle_inner,
    );

    // Thinking effort selector (chunks[6], only when thinking is ON)
    if thinking_on {
        let effort_focused = app.cfg_focus == ConfigFocus::ThinkingEffort;
        let effort_border_color = if effort_focused {
            Color::Cyan
        } else {
            Color::DarkGray
        };
        let effort_block = Block::default()
            .borders(Borders::ALL)
            .title(" 思考强度 ")
            .style(Style::default().fg(effort_border_color));
        let effort_inner = effort_block.inner(chunks[6]);
        f.render_widget(effort_block, chunks[6]);

        const EFFORTS: [&str; 3] = ["低", "高", "最大"];
        let effort_text = EFFORTS
            .iter()
            .enumerate()
            .map(|(i, label)| {
                if i == app.cfg_effort {
                    format!("[ {} ]", label)
                } else {
                    format!("  {}  ", label)
                }
            })
            .collect::<String>();
        let effort_color = if effort_focused {
            Color::White
        } else {
            Color::DarkGray
        };
        f.render_widget(
            Paragraph::new(effort_text).style(Style::default().fg(effort_color)),
            effort_inner,
        );
    }

    // Fast mode toggle
    let fast_focused = app.cfg_focus == ConfigFocus::FastModeToggle;
    let fast_border_color = if fast_focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let fast_chunk = &chunks[7 - effort_shift];
    let fast_block = Block::default()
        .borders(Borders::ALL)
        .title(" 快速模式 ")
        .style(Style::default().fg(fast_border_color));
    let fast_inner = fast_block.inner(*fast_chunk);
    f.render_widget(fast_block, *fast_chunk);

    let fast_text = if app.cfg_fast_mode {
        "[✓] 开启 - 取消答题间隔"
    } else {
        "[ ] 关闭 - 每题间隔约1秒"
    };
    let fast_color = if fast_focused {
        Color::White
    } else {
        Color::DarkGray
    };
    f.render_widget(
        Paragraph::new(fast_text).style(Style::default().fg(fast_color)),
        fast_inner,
    );

    // Save button
    let save_focused = app.cfg_focus == ConfigFocus::SaveBtn;
    let save_style = if save_focused {
        selected_style(Color::Green)
    } else {
        dim_style(Color::DarkGray)
    };
    let save_text = if save_focused {
        "[ 保存 ]"
    } else {
        "  保存  "
    };
    f.render_widget(
        Paragraph::new(save_text)
            .style(save_style)
            .alignment(Alignment::Center),
        chunks[8 - effort_shift],
    );

    // Template button
    let tpl_focused = app.cfg_focus == ConfigFocus::TemplateBtn;
    let tpl_style = if tpl_focused {
        selected_style(Color::Cyan)
    } else {
        dim_style(Color::DarkGray)
    };
    let tpl_text = if tpl_focused {
        "[ 选择预设模板 ]"
    } else {
        "  选择预设模板  "
    };
    f.render_widget(
        Paragraph::new(tpl_text)
            .style(tpl_style)
            .alignment(Alignment::Center),
        chunks[9 - effort_shift],
    );

    // Reset button
    let reset_focused = app.cfg_focus == ConfigFocus::ResetBtn;
    let reset_style = if reset_focused {
        selected_style(Color::Red)
    } else {
        dim_style(Color::DarkGray)
    };
    let reset_text = if reset_focused {
        "[ 重置 ]"
    } else {
        "  重置  "
    };
    f.render_widget(
        Paragraph::new(reset_text)
            .style(reset_style)
            .alignment(Alignment::Center),
        chunks[10 - effort_shift],
    );

    f.render_widget(
        Paragraph::new("↑↓ 切换  Space 勾选  Enter 确认  ESC 返回  (模型框 Enter 拉模型列表)")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        chunks[12 - effort_shift],
    );
}

fn draw_model_select(f: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    use ratatui::{
        layout::{Alignment, Constraint, Layout},
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::Paragraph,
    };

    let outer = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(area);
    let chunks = Layout::vertical([
        Constraint::Length(1), // title
        Constraint::Length(1), // search
        Constraint::Min(1),    // list
        Constraint::Length(1), // page indicator
    ])
    .split(outer[0]);

    let filtered = app.filtered_models();
    let title = format!(
        " 选择模型 · {} ",
        app.cfg_protocol.display_name()
    );
    f.render_widget(
        Paragraph::new(title).style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        chunks[0],
    );

    let search = if app.model_filter.is_empty() {
        "搜索: 直接输入关键词过滤".to_string()
    } else {
        format!("搜索: {}_", app.model_filter)
    };
    f.render_widget(
        Paragraph::new(search).style(Style::default().fg(Color::Yellow)),
        chunks[1],
    );

    if app.models_loading {
        f.render_widget(
            Paragraph::new("正在拉取模型列表…").style(Style::default().fg(Color::DarkGray)),
            chunks[2],
        );
    } else if let Some(error) = &app.models_error {
        f.render_widget(
            Paragraph::new(error.as_str())
                .style(Style::default().fg(Color::Red))
                .wrap(ratatui::widgets::Wrap { trim: true }),
            chunks[2],
        );
    } else if filtered.is_empty() {
        let hint = if app.models.is_empty() {
            "该服务没有返回任何模型"
        } else {
            "没有匹配的模型"
        };
        f.render_widget(
            Paragraph::new(hint).style(Style::default().fg(Color::DarkGray)),
            chunks[2],
        );
    } else {
        // 每页 20 个；终端不够高时只截取当前页能显示的行，光标跟着滚动。
        let page_start = app.model_page() * App::MODELS_PER_PAGE;
        let visible = chunks[2].height as usize;
        let window_start = if app.model_cursor >= page_start + visible {
            app.model_cursor + 1 - visible
        } else {
            page_start
        };
        let page_end = (page_start + App::MODELS_PER_PAGE).min(filtered.len());
        for (row, index) in (window_start..page_end.min(window_start + visible)).enumerate() {
            let model = filtered[index];
            let is_sel = index == app.model_cursor;
            let in_use = model.id == app.cfg_fields[1];
            let dim = Style::default().fg(Color::DarkGray);
            let mut spans = Vec::new();
            if in_use {
                spans.push(Span::styled("● ", Style::default().fg(Color::Green)));
            }
            spans.push(Span::raw(format!(
                "{}{}",
                if is_sel { "[ " } else { "  " },
                model.id
            )));
            if model.label != model.id {
                spans.push(Span::styled(format!("  {}", model.label), dim));
            }
            if let Some(detail) = &model.detail {
                // 描述可能很长，截断后由 Paragraph 按宽度再裁一次。
                let short: String = detail.chars().take(60).collect();
                spans.push(Span::styled(format!(" · {short}"), dim));
            }
            f.render_widget(
                Paragraph::new(Line::from(spans)).style(
                    Style::default()
                        .fg(if is_sel { Color::Yellow } else { Color::White })
                        .add_modifier(if is_sel {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                chunks[2].offset(ratatui::layout::Offset {
                    x: 0,
                    y: row as i32,
                }),
            );
        }
    }

    let mut indicator = format!(
        "第 {}/{} 页 · 共 {} 个",
        if filtered.is_empty() { 0 } else { app.model_page() + 1 },
        app.model_page_count(),
        filtered.len()
    );
    if app.models_truncated {
        indicator.push_str(" (仅显示前 200 个)");
    }
    f.render_widget(
        Paragraph::new(indicator)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        chunks[3],
    );

    f.render_widget(
        Paragraph::new("↑↓ 选择  ←→ 翻页  输入过滤  Backspace 退格  Enter 确认  ESC 关闭")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        outer[1],
    );
}

fn draw_preset_select(f: &mut ratatui::Frame, area: ratatui::layout::Rect, sel: usize) {
    use ratatui::{
        layout::{Alignment, Constraint, Layout},
        style::{Color, Modifier, Style},
        widgets::Paragraph,
    };

    let presets = config::load_presets();
    let _count = presets.len();

    let outer = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(area);

    // title + separator + presets (each 2 rows for breathing room)
    let preset_rows: Vec<Constraint> = std::iter::once(Constraint::Length(1)) // title
        .chain(std::iter::once(Constraint::Length(1))) // separator
        .chain(presets.iter().map(|_| Constraint::Length(2))) // each preset (taller)
        .chain(std::iter::once(Constraint::Min(1))) // flexible spacer
        .collect();

    let chunks = Layout::vertical(preset_rows).split(outer[0]);

    // Title
    f.render_widget(
        Paragraph::new("选择预设模板")
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(Alignment::Center),
        chunks[0],
    );

    // Separator
    f.render_widget(
        Paragraph::new("─".repeat(chunks[1].width as usize))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        chunks[1],
    );

    // Preset items
    for (i, preset) in presets.iter().enumerate() {
        let is_sel = i == sel;
        let text_color = if is_sel { Color::Yellow } else { Color::White };
        let style = if is_sel {
            Style::default().fg(text_color).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(text_color)
        };

        let text = if is_sel {
            format!("[ {} ]", preset.provider_name)
        } else {
            format!("  {}  ", preset.provider_name)
        };
        f.render_widget(
            Paragraph::new(text)
                .style(style)
                .alignment(Alignment::Center),
            chunks[2 + i],
        );
    }

    // Help text at the absolute bottom
    f.render_widget(
        Paragraph::new("↑↓ 选择  Enter 确认  ESC 取消")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        outer[1],
    );
}

fn draw_reset_confirm(f: &mut ratatui::Frame, area: ratatui::layout::Rect, choice: u8) {
    use ratatui::{
        layout::{Alignment, Constraint, Layout},
        style::{Color, Modifier, Style},
        widgets::Paragraph,
    };

    let outer = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(area);

    let chunks = Layout::vertical([
        Constraint::Percentage(30),
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Percentage(30),
    ])
    .split(outer[0]);

    f.render_widget(
        Paragraph::new("确认重置")
            .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center),
        chunks[1],
    );

    f.render_widget(
        Paragraph::new("将会重置所有配置项以及登录状态")
            .style(Style::default().fg(Color::Yellow))
            .alignment(Alignment::Center),
        chunks[2],
    );

    f.render_widget(
        Paragraph::new(if choice == 0 {
            "[ 取消 ]"
        } else {
            "  取消  "
        })
        .style(if choice == 0 {
            selected_style(Color::Green)
        } else {
            dim_style(Color::DarkGray)
        })
        .alignment(Alignment::Center),
        chunks[3],
    );

    f.render_widget(
        Paragraph::new(if choice == 1 {
            "[ 仅退出登录 ]"
        } else {
            "  仅退出登录  "
        })
        .style(if choice == 1 {
            selected_style(Color::Yellow)
        } else {
            dim_style(Color::DarkGray)
        })
        .alignment(Alignment::Center),
        chunks[4],
    );

    f.render_widget(
        Paragraph::new(if choice == 2 {
            "[ 确认重置 ]"
        } else {
            "  确认重置  "
        })
        .style(if choice == 2 {
            selected_style(Color::Red)
        } else {
            dim_style(Color::DarkGray)
        })
        .alignment(Alignment::Center),
        chunks[5],
    );

    f.render_widget(
        Paragraph::new("↑↓ 选择  Enter 确认  ESC 取消")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        outer[1],
    );
}
