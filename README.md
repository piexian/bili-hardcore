# Bili-Hardcore

B 站硬核会员自动答题工具，利用 LLM 实现智能答题功能。

## 使用前须知
- 请确保您的 B 站账号已满 6 级，根据 B 站规则，6 级用户才可以进行硬核会员试炼
- 硬核会员试炼每天有 3 次答题机会，达到限制后需要 24 小时后才能重新答题，具体时间可以前往 B 站 APP 答题页面查看
- 没有 API Key 的可以免费去硅基流动注册一个账号，会送 14 元免费额度，这是我的[邀请链接](https://cloud.siliconflow.cn/i/9Fur0aVC)

## 支持的模型服务

在「配置 → 选择预设模板」中可选择以下预设，也可以手动填写任意兼容服务：

| 预设 | 接口地址 | 默认模型 |
|------|---------|---------|
| 硅基流动 | `https://api.siliconflow.cn/v1/chat/completions` | `deepseek-ai/DeepSeek-V4-Flash` |
| DeepSeek | `https://api.deepseek.com/v1/chat/completions` | `deepseek-v4-flash` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | `glm-4.7` |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `gpt-5.4-nano` |
| Grok (xAI) | `https://api.x.ai/v1/chat/completions` | `grok-4.6` |
| JEV (TypeSafe) | `https://api.typesafe.ai/v1/systemone` | `jev-latest` |

前五个预设使用 OpenAI 兼容的 Chat Completions 协议。

### JEV (TypeSafe)

JEV 不是对话模型，而是 TypeSafe 的 System One 决策模型：它不生成文本，只针对给定内容返回结构化决策。本项目使用其 `choice` 原语——把题干作为 `state` 传入、四个选项作为 `criteria`，直接取回被选中的选项序号与各选项概率。接口地址以 `/systemone` 结尾时程序会自动使用该协议，因此手动填入中转地址同样可用。

- API Key 在 [TypeSafe 控制台](https://console.typesafe.ai/keys) 获取，接口文档见 [docs.typesafe.ai](https://docs.typesafe.ai/api)。
- JEV 没有「思考模式」开关，配置页会标注该开关不适用；答题过程中会显示返回的模型版本、置信度与各选项概率。
- 命中 `429`（限流）或 `529`（过载）时按官方建议做指数退避重试；这类响应发生在推理开始前，不会重复计费。
- JEV 以英语为主训练语言，中文题目准确率低于英文，使用时请留意界面显示的置信度。

## 安装

> 之前用过 0.x 版本的老用户请先删除配置文件：
> ```bash
> # macOS / Linux
> rm -rf ~/.bili-hardcore
> ```
> ```powershell
> # Windows (PowerShell)
> Remove-Item -Recurse -Force "$env:USERPROFILE\.bili-hardcore"
> ```


### 快速安装（推荐）

**macOS / Linux:**
```bash
curl -fsSL https://github.com/Karben233/bili-hardcore/releases/latest/download/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://github.com/Karben233/bili-hardcore/releases/latest/download/install.ps1 | iex
```

### 手动下载

前往 [Releases](https://github.com/Karben233/bili-hardcore/releases/latest) 下载对应平台的文件：

| 平台 | 推荐文件 |
|------|---------|
| macOS (Intel / Apple Silicon) | `bili-hardcore-*-darwin-universal.tar.gz` |
| Windows (x64) | `bili-hardcore-*-windows-x64.zip` |
| Linux (x64) | `bili-hardcore-*-linux-x64-musl.tar.gz` |
| Linux (ARM64) | `bili-hardcore-*-linux-arm64-musl.tar.gz` |

解压后赋予执行权限即可运行：
```bash
chmod +x bili-hardcore
./bili-hardcore
```

> **macOS 提示**: 如遇"无法验证开发者"，执行 `xattr -cr /path/to/bili-hardcore`
> **Linux 提示**: 优先使用 `-musl` 版本（静态链接，兼容所有发行版）

### 从源码构建

需要 Rust 1.88 及以上版本：

```bash
git clone https://github.com/Karben233/bili-hardcore.git
cd bili-hardcore
cargo build --release
./target/release/bili-hardcore
```

## 使用

### 启动
```bash
bili-hardcore                    # 交互式配置后启动
bili-hardcore <url> <model> -k <api-key>   # 通过命令行参数直接启动
```

### 命令
```bash
bili-hardcore update             # 检查并更新到最新版本
bili-hardcore uninstall          # 卸载
```

## 作为 agent skill 使用

本项目同时提供了一个通用的 agent skill（位于 `.agents/skills/bili-hardcore/`）。装上后对任意支持该约定的 AI 编程助手说"帮我答硬核会员"即可触发，由 **agent 自身作答**。

## 安装

与 agent 对话：`帮我安装这个skill https://github.com/Karben233/bili-hardcore/tree/main/.agents/skills/bili-hardcore`

### 使用

在任意 agent 会话里对 agent 说：

- "帮我答 B 站硬核会员"
- "开始硬核会员答题"
- "查一下我的硬核会员答题得分"

## 运行截图
![运行截图](screenshot.png)
