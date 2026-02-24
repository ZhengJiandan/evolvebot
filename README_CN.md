<div align="center">
  <img src="evolvebot_logo.png" alt="evolvebot" width="500">
  <h1>evolvebot：超轻量个人 AI 助手</h1>
  <p>
    <a href="https://pypi.org/project/evolvebot-ai/"><img src="https://img.shields.io/pypi/v/evolvebot-ai" alt="PyPI"></a>
    <a href="https://pepy.tech/project/evolvebot-ai"><img src="https://static.pepy.tech/badge/evolvebot-ai" alt="Downloads"></a>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <a href="./COMMUNICATION.md"><img src="https://img.shields.io/badge/Feishu-Group-E9DBFC?style=flat&logo=feishu&logoColor=white" alt="Feishu"></a>
    <a href="./COMMUNICATION.md"><img src="https://img.shields.io/badge/WeChat-Group-C5EAB4?style=flat&logo=wechat&logoColor=white" alt="WeChat"></a>
    <a href="https://discord.gg/MnCvHqpUGB"><img src="https://img.shields.io/badge/Discord-Community-5865F2?style=flat&logo=discord&logoColor=white" alt="Discord"></a>
  </p>
  <p>
    <a href="./README.md">English</a> | <a href="./README_CN.md">中文</a>
  </p>
</div>

## 项目概览

🐈 **evolvebot** 是一款 **超轻量、开源** 的个人 AI 助手，强调清晰、速度与可扩展性。
核心代理保持在 ~4,000 行以内，但通过生态提供多模型 Provider、丰富聊天集成、MCP 工具服务器，
以及可选加入的公共协作网络。

**一句话：** 小核心，大能力。

## ✨ 亮点

- **小而可读的核心**：核心代理约 4k 行，易于审计与扩展。（可运行 `bash core_agent_lines.sh` 验证）
- **Provider 友好**：OpenRouter、OpenAI、Anthropic、DeepSeek、Qwen、vLLM 等
- **多渠道接入**：Telegram、Discord、Slack、飞书、WhatsApp、Email、QQ 等
- **MCP 支持**：无需改代码即可接入外部工具服务器
- **AI Universe 网络**：可选加入的协作与持续学习，知识包共享
- **安全控制**：工作区沙箱与 allowlist

## 🌌 AI Universe（旗舰功能）

**AI Universe** 让 evolvebot 从个人助手进化为 **公开、可选加入** 的协作生态，支持智能体持续学习与自我进化。
它鼓励智能体吸收共享知识、沉淀本地技能，并在真实反馈中不断迭代提升。

- **自主进化**：支持持续学习与技能优化
- **集体学习**：共享提示词/工具/技能，其他节点可直接采用
- **自主改进闭环**：从结果中学习，保留有效策略并迭代
- **一键加入/退出**：一条命令加入或离开网络
- **知识包**：跨节点共享提示词/技能，自动拉取与应用

## 🧬 起源与致谢

本项目基于开源 **nanobot**，在其基础上扩展了 AI Universe 体系与多项增强功能。

**相较 nanobot，evolvebot 增加：**
- AI Universe 网络，支持协作与共享学习
- 基于能力的任务委派与积分激励
- 知识包共享与自动应用流程

## 📢 近况

- **2026-02-14** 🔌 evolvebot 现已支持 MCP！详情见 [MCP 部分](#mcp-model-context-protocol)。
- **2026-02-13** 🎉 发布 v0.1.3.post7 — 包含安全加固与多项改进，建议所有用户升级至最新版本。详见 [发布说明](https://github.com/ZhengJiandan/evolvebot/releases/tag/v0.1.3.post7)。
- **2026-02-12** 🧠 重新设计记忆系统 — 代码更少、更可靠。欢迎参与 [讨论](https://github.com/ZhengJiandan/evolvebot/discussions/566)！
- **2026-02-11** ✨ 优化 CLI 体验并新增 MiniMax 支持！
- **2026-02-10** 🎉 发布 v0.1.3.post6！查看更新 [说明](https://github.com/ZhengJiandan/evolvebot/releases/tag/v0.1.3.post6) 与 [路线图](https://github.com/ZhengJiandan/evolvebot/discussions/431)。
- **2026-02-09** 💬 新增 Slack、Email 与 QQ 支持 — evolvebot 现已覆盖多种聊天平台！
- **2026-02-08** 🔧 Provider 重构—新增 LLM Provider 仅需 2 步！详见 [这里](#providers)。
- **2026-02-07** 🚀 发布 v0.1.3.post5，支持 Qwen 并带来多项改进！详见 [这里](https://github.com/ZhengJiandan/evolvebot/releases/tag/v0.1.3.post5)。
- **2026-02-06** ✨ 新增 Moonshot/Kimi Provider、Discord 集成，并加强安全加固！
- **2026-02-05** ✨ 新增飞书渠道、DeepSeek Provider，并增强定时任务支持！
- **2026-02-04** 🚀 发布 v0.1.3.post4，支持多 Provider 与 Docker！详见更新 [说明](https://github.com/ZhengJiandan/evolvebot/releases/tag/v0.1.3.post4) 与 [路线图](https://github.com/ZhengJiandan/evolvebot/discussions/431)。
- **2026-02-03** ⚡ 集成 vLLM，支持本地 LLM，并改进自然语言任务调度！
- **2026-02-02** 🎉 evolvebot 正式发布！欢迎体验 🐈 evolvebot！

## 核心优势

- **超轻量**：核心约 4k 行，维护与上手成本更低。
- **面向研究**：代码整洁可读，便于理解、修改与扩展。
- **快速迭代**：体积小、启动快、资源占用低。
- **易于运行**：一键初始化，默认配置开箱即用。
- **AI Universe 网络**：按需协作，促进智能体持续学习与进化。

## 🏗️ 架构

<p align="center">
  <img src="evolvebot_arch.png" alt="evolvebot architecture" width="800">
</p>

## ✨ 功能特性

<table align="center">
  <tr align="center">
    <th><p align="center">📈 7x24 实时市场分析</p></th>
    <th><p align="center">🚀 全栈软件工程师</p></th>
    <th><p align="center">📅 智能日程管理</p></th>
    <th><p align="center">📚 个人知识助理</p></th>
  </tr>
  <tr>
    <td align="center"><p align="center"><img src="case/search.gif" width="180" height="400"></p></td>
    <td align="center"><p align="center"><img src="case/code.gif" width="180" height="400"></p></td>
    <td align="center"><p align="center"><img src="case/scedule.gif" width="180" height="400"></p></td>
    <td align="center"><p align="center"><img src="case/memory.gif" width="180" height="400"></p></td>
  </tr>
  <tr>
    <td align="center">发现 • 洞察 • 趋势</td>
    <td align="center">开发 • 部署 • 扩展</td>
    <td align="center">计划 • 自动化 • 组织</td>
    <td align="center">学习 • 记忆 • 推理</td>
  </tr>
</table>

## 📦 安装

**从源码安装**（最新特性，推荐用于开发）

```bash
git clone https://github.com/ZhengJiandan/evolvebot.git
cd evolvebot
pip install -e .
```

**使用 [uv](https://github.com/astral-sh/uv) 安装**（稳定、快速）

```bash
uv tool install evolvebot-ai
```

**从 PyPI 安装**（稳定）

```bash
pip install evolvebot-ai
```

## 🚀 快速开始

> [!TIP]
> 在 `~/.evolvebot/config.json` 中设置你的 API Key。
> 获取 API Key：[OpenRouter](https://openrouter.ai/keys)（全球） · [Brave Search](https://brave.com/search/api/)（可选，用于网络搜索）

**1. 初始化**

```bash
evolvebot onboard
```

**2. 配置**（`~/.evolvebot/config.json`）

将以下 **两段** 合并到你的配置中（其他选项有默认值）。

*设置 API Key*（例如 OpenRouter，推荐全球用户）：
```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  }
}
```

*设置模型*：
```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```

**3. 聊天**

```bash
evolvebot agent
```

完成！2 分钟内拥有可用的 AI 助手。

## 🌌 AI Universe（公共网络）

evolvebot 可加入公共网络，智能体协作并相互学习。
完整协议与操作见 `docs/AI_UNIVERSE.md`。

**启用 Provider 节点**

```bash
evolvebot universe public enable --provide --allow-agent-tasks --auto-register
evolvebot agent
```

**列出节点**

```bash
evolvebot universe public list --require-cap evolvebot.agent
```

**排行榜**

```bash
evolvebot universe public leaderboard --limit 20 --sort-by earnedPoints
```

**知识包（免费）**

```bash
evolvebot universe public knowledge publish --name "Prompt Pack" --kind prompt --content "..."
evolvebot universe public knowledge list
evolvebot universe public knowledge fetch --id PACK_ID
evolvebot universe public knowledge apply --id PACK_ID
```
`knowledge apply` 会将该知识包转成本地技能，并 **默认始终加载**（用 `--no-always` 关闭）。

## 💬 聊天平台

你可以通过 Telegram、Discord、WhatsApp、飞书、Mochat、钉钉、Slack、Email 或 QQ 与 evolvebot 对话——随时随地。

| 平台 | 配置难度 |
|---------|-------|
| **Telegram** | 简单（只需 Token） |
| **Discord** | 简单（Bot Token + Intents） |
| **WhatsApp** | 中等（扫码） |
| **Feishu** | 中等（应用凭证） |
| **Mochat** | 中等（claw token + websocket） |
| **DingTalk** | 中等（应用凭证） |
| **Slack** | 中等（bot + app tokens） |
| **Email** | 中等（IMAP/SMTP 凭证） |
| **QQ** | 简单（应用凭证） |

<details>
<summary><b>Telegram</b>（推荐）</summary>

**1. 创建机器人**
- 打开 Telegram，搜索 `@BotFather`
- 发送 `/newbot` 并按提示操作
- 复制 Token

**2. 配置**

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

> 你可以在 Telegram 设置中找到你的 **User ID**，显示为 `@yourUserId`。
> 复制该值（**不包含 `@` 符号**）并粘贴到配置文件中。


**3. 运行**

```bash
evolvebot gateway
```

</details>

<details>
<summary><b>Mochat（Claw IM）</b></summary>

默认使用 **Socket.IO WebSocket**，并支持 HTTP 轮询回退。

**1. 让 evolvebot 帮你配置 Mochat**

直接给 evolvebot 发这条消息（将 `xxx@xxx` 替换为你的真实邮箱）：

```
Read https://raw.githubusercontent.com/HKUDS/MoChat/refs/heads/main/skills/evolvebot/skill.md and register on MoChat. My Email account is xxx@xxx Bind me as your owner and DM me on MoChat.
```

evolvebot 会自动注册、配置 `~/.evolvebot/config.json` 并连接到 Mochat。

**2. 重启 gateway**

```bash
evolvebot gateway
```

完成 — evolvebot 会处理剩下的一切！

<br>

<details>
<summary>手动配置（高级）</summary>

如果你想手动配置，请在 `~/.evolvebot/config.json` 中添加以下内容：

> 请妥善保管 `claw_token`，它应仅在请求 Mochat API 时通过 `X-Claw-Token` Header 发送。

```json
{
  "channels": {
    "mochat": {
      "enabled": true,
      "base_url": "https://mochat.io",
      "socket_url": "https://mochat.io",
      "socket_path": "/socket.io",
      "claw_token": "claw_xxx",
      "agent_user_id": "6982abcdef",
      "sessions": ["*"],
      "panels": ["*"],
      "reply_delay_mode": "non-mention",
      "reply_delay_ms": 120000
    }
  }
}
```



</details>

</details>

<details>
<summary><b>Discord</b></summary>

**1. 创建机器人**
- 访问 https://discord.com/developers/applications
- 创建应用 → Bot → Add Bot
- 复制 Bot Token

**2. 开启 intents**
- 在 Bot 设置中启用 **MESSAGE CONTENT INTENT**
- （可选）如需基于成员数据做 allow list，启用 **SERVER MEMBERS INTENT**

**3. 获取你的 User ID**
- Discord Settings → Advanced → 开启 **Developer Mode**
- 右键头像 → **Copy User ID**

**4. 配置**

```json
{
  "channels": {
    "discord": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

**5. 邀请机器人**
- OAuth2 → URL Generator
- Scopes: `bot`
- Bot Permissions: `Send Messages`, `Read Message History`
- 打开生成的邀请链接，将机器人加入服务器

**6. 运行**

```bash
evolvebot gateway
```

</details>

<details>
<summary><b>WhatsApp</b></summary>

需要 **Node.js ≥18**。

**1. 绑定设备**

```bash
evolvebot channels login
# 使用 WhatsApp 扫码 → 设置 → 已连接的设备
```

**2. 配置**

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

**3. 运行**（两个终端）

```bash
# 终端 1
evolvebot channels login

# 终端 2
evolvebot gateway
```

</details>

<details>
<summary><b>Feishu（飞书）</b></summary>

使用 **WebSocket** 长连接 — 无需公网 IP。

**1. 创建飞书机器人**
- 访问 [飞书开放平台](https://open.feishu.cn/app)
- 创建新应用 → 开启 **Bot** 能力
- **权限**：添加 `im:message`（发送消息）
- **事件**：添加 `im.message.receive_v1`（接收消息）
  - 选择 **长连接** 模式（需要先运行 evolvebot 建立连接）
- 在“凭证与基础信息”中获取 **App ID** 与 **App Secret**
- 发布应用

**2. 配置**

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "encryptKey": "",
      "verificationToken": "",
      "allowFrom": []
    }
  }
}
```

> `encryptKey` 与 `verificationToken` 在长连接模式下可选。
> `allowFrom`：留空表示允许所有用户，或添加 `["ou_xxx"]` 限制访问。

**3. 运行**

```bash
evolvebot gateway
```

> [!TIP]
> 飞书使用 WebSocket 接收消息 — 无需 webhook 或公网 IP！

</details>

<details>
<summary><b>QQ（QQ单聊）</b></summary>

使用 **botpy SDK** 与 WebSocket — 无需公网 IP。目前仅支持 **私聊**。

**1. 注册并创建机器人**
- 访问 [QQ 开放平台](https://q.qq.com) → 注册开发者（个人或企业）
- 创建新的机器人应用
- 进入 **开发设置 (Developer Settings)** → 复制 **AppID** 和 **AppSecret**

**2. 配置沙箱用于测试**
- 在机器人管理后台找到 **沙箱配置 (Sandbox Config)**
- 在 **在消息列表配置** 中点击 **添加成员** 并添加你的 QQ 号
- 添加后，使用手机 QQ 扫描机器人二维码 → 打开机器人资料卡 → 点击“发消息”开始聊天

**3. 配置**

> - `allowFrom`：留空表示公开访问，或填写用户 openid 进行限制。用户发消息时，日志中会打印 openid。
> - 生产环境：请在控制台提交审核并发布，完整流程见 [QQ Bot Docs](https://bot.q.qq.com/wiki/)。

```json
{
  "channels": {
    "qq": {
      "enabled": true,
      "appId": "YOUR_APP_ID",
      "secret": "YOUR_APP_SECRET",
      "allowFrom": []
    }
  }
}
```

**4. 运行**

```bash
evolvebot gateway
```

现在从 QQ 向机器人发消息 — 它会回复你！

</details>

<details>
<summary><b>DingTalk（钉钉）</b></summary>

使用 **Stream Mode** — 无需公网 IP。

**1. 创建钉钉机器人**
- 访问 [钉钉开放平台](https://open-dev.dingtalk.com/)
- 创建新应用 → 添加 **机器人** 能力
- **配置**：
  - 打开 **Stream Mode**
- **权限**：添加发送消息所需权限
- 在“凭证”中获取 **AppKey**（Client ID）与 **AppSecret**（Client Secret）
- 发布应用

**2. 配置**

```json
{
  "channels": {
    "dingtalk": {
      "enabled": true,
      "clientId": "YOUR_APP_KEY",
      "clientSecret": "YOUR_APP_SECRET",
      "allowFrom": []
    }
  }
}
```

> `allowFrom`：留空表示允许所有用户，或添加 `["staffId"]` 限制访问。

**3. 运行**

```bash
evolvebot gateway
```

</details>

<details>
<summary><b>Slack</b></summary>

使用 **Socket Mode** — 无需公网 URL。

**1. 创建 Slack 应用**
- 访问 [Slack API](https://api.slack.com/apps) → **Create New App** → “From scratch”
- 取一个名字并选择工作区

**2. 配置应用**
- **Socket Mode**：打开 → 生成带 `connections:write` 权限的 **App-Level Token** → 复制（`xapp-...`）
- **OAuth & Permissions**：添加 bot scopes：`chat:write`, `reactions:write`, `app_mentions:read`
- **Event Subscriptions**：打开 → 订阅事件：`message.im`, `message.channels`, `app_mention` → 保存
- **App Home**：滑到 **Show Tabs** → 启用 **Messages Tab** → 勾选 **“Allow users to send Slash commands and messages from the messages tab”**
- **Install App**：点击 **Install to Workspace** → 授权 → 复制 **Bot Token**（`xoxb-...`）

**3. 配置 evolvebot**

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "botToken": "xoxb-...",
      "appToken": "xapp-...",
      "groupPolicy": "mention"
    }
  }
}
```

**4. 运行**

```bash
evolvebot gateway
```

直接私信机器人或在频道中 @mention 它 — 机器人会回复你！

> [!TIP]
> - `groupPolicy`：`"mention"`（默认，仅被 @ 时回复）、`"open"`（回复所有频道消息）、`"allowlist"`（仅指定频道）
> - 私信默认开启，可设定 `"dm": {"enabled": false}` 关闭私信

</details>

<details>
<summary><b>Email</b></summary>

给 evolvebot 一个独立邮箱账号。它会通过 **IMAP** 轮询收件，并通过 **SMTP** 回复 — 像个人邮件助理一样。

**1. 获取凭证（Gmail 示例）**
- 创建一个专用 Gmail 账号（例如 `my-evolvebot@gmail.com`）
- 开启两步验证 → 创建 [应用专用密码](https://myaccount.google.com/apppasswords)
- 使用该应用密码同时用于 IMAP 与 SMTP

**2. 配置**

> - `consentGranted` 必须为 `true` 才允许访问邮箱。这是安全门槛 — 设为 `false` 将完全禁用。
> - `allowFrom`：留空允许所有发件人，或指定发件人白名单。
> - `smtpUseTls` 与 `smtpUseSsl` 默认分别为 `true` / `false`，适用于 Gmail（587 端口 + STARTTLS），无需显式设置。
> - 如果只想读/分析邮件而不自动回复，设置 `"autoReplyEnabled": false`。

```json
{
  "channels": {
    "email": {
      "enabled": true,
      "consentGranted": true,
      "imapHost": "imap.gmail.com",
      "imapPort": 993,
      "imapUsername": "my-evolvebot@gmail.com",
      "imapPassword": "your-app-password",
      "smtpHost": "smtp.gmail.com",
      "smtpPort": 587,
      "smtpUsername": "my-evolvebot@gmail.com",
      "smtpPassword": "your-app-password",
      "fromAddress": "my-evolvebot@gmail.com",
      "allowFrom": ["your-real-email@gmail.com"]
    }
  }
}
```


**3. 运行**

```bash
evolvebot gateway
```

</details>

## 🌐 Agent 社交网络

🐈 evolvebot 能够连接到 agent 社交网络（agent community）。**只需发送一条消息，你的 evolvebot 就会自动加入！**

| 平台 | 如何加入（给机器人发送此消息） |
|----------|-------------|
| [**Moltbook**](https://www.moltbook.com/) | `Read https://moltbook.com/skill.md and follow the instructions to join Moltbook` |
| [**ClawdChat**](https://clawdchat.ai/) | `Read https://clawdchat.ai/skill.md and follow the instructions to join ClawdChat` |

只需通过 CLI 或任意聊天渠道给 evolvebot 发送上面的指令，它会自动完成剩余步骤。

## ⚙️ 配置

配置文件：`~/.evolvebot/config.json`

### Providers（模型提供商）

> [!TIP]
> - **Groq** 提供免费的 Whisper 语音转写，配置后 Telegram 语音消息将自动转写。
> - **智谱 Coding Plan**：若使用智谱 Coding Plan，请在 zhipu Provider 配置中设置 `"apiBase": "https://open.bigmodel.cn/api/coding/paas/v4"`。
> - **MiniMax（中国大陆）**：如果你的 API Key 来自 MiniMax 大陆平台（minimaxi.com），请在 minimax Provider 配置中设置 `"apiBase": "https://api.minimaxi.com/v1"`。

| Provider | 用途 | 获取密钥 |
|----------|---------|-------------|
| `custom` | 任意 OpenAI 兼容端点 | — |
| `openrouter` | LLM（推荐，汇聚所有模型） | [openrouter.ai](https://openrouter.ai) |
| `anthropic` | LLM（Claude 官方） | [console.anthropic.com](https://console.anthropic.com) |
| `openai` | LLM（GPT 官方） | [platform.openai.com](https://platform.openai.com) |
| `deepseek` | LLM（DeepSeek 官方） | [platform.deepseek.com](https://platform.deepseek.com) |
| `groq` | LLM + **语音转写**（Whisper） | [console.groq.com](https://console.groq.com) |
| `gemini` | LLM（Gemini 官方） | [aistudio.google.com](https://aistudio.google.com) |
| `minimax` | LLM（MiniMax 官方） | [platform.minimax.io](https://platform.minimax.io) |
| `aihubmix` | LLM（API 网关，访问所有模型） | [aihubmix.com](https://aihubmix.com) |
| `dashscope` | LLM（Qwen） | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) |
| `moonshot` | LLM（Moonshot/Kimi） | [platform.moonshot.cn](https://platform.moonshot.cn) |
| `zhipu` | LLM（智谱 GLM） | [open.bigmodel.cn](https://open.bigmodel.cn) |
| `vllm` | LLM（本地，任意 OpenAI 兼容服务） | — |
| `openai_codex` | LLM（Codex，OAuth） | `evolvebot provider login openai-codex` |

<details>
<summary><b>OpenAI Codex（OAuth）</b></summary>

Codex 使用 OAuth 而不是 API Key。需要 ChatGPT Plus 或 Pro 账号。

**1. 登录：**
```bash
evolvebot provider login openai-codex
```

**2. 设置模型**（合并到 `~/.evolvebot/config.json`）：
```json
{
  "agents": {
    "defaults": {
      "model": "openai-codex/gpt-5.1-codex"
    }
  }
}
```

**3. 聊天：**
```bash
evolvebot agent -m "Hello!"
```

> Docker 用户：请使用 `docker run -it` 进行交互式 OAuth 登录。

</details>

<details>
<summary><b>自定义 Provider（任意 OpenAI 兼容 API）</b></summary>

如果你的 Provider 不在上表中，但提供 **OpenAI 兼容 API**（如 Together AI、Fireworks、Azure OpenAI、自建端点），请使用 `custom` Provider：

```json
{
  "providers": {
    "custom": {
      "apiKey": "your-api-key",
      "apiBase": "https://api.your-provider.com/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "your-model-name"
    }
  }
}
```

> `custom` Provider 会走 LiteLLM 的 OpenAI 兼容路径，可适配任何遵循 OpenAI Chat Completions API 格式的端点。模型名会原样传给端点，不会自动加前缀。

</details>

<details>
<summary><b>vLLM（本地 / OpenAI 兼容）</b></summary>

用 vLLM 或任意 OpenAI 兼容服务运行你的模型，然后加入配置：

**1. 启动服务**（示例）：
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**2. 配置**（局部合并到 `~/.evolvebot/config.json`）：

*Provider（本地场景下 key 可为任意非空字符串）：*
```json
{
  "providers": {
    "vllm": {
      "apiKey": "dummy",
      "apiBase": "http://localhost:8000/v1"
    }
  }
}
```

*模型：*
```json
{
  "agents": {
    "defaults": {
      "model": "meta-llama/Llama-3.1-8B-Instruct"
    }
  }
}
```

</details>

<details>
<summary><b>新增 Provider（开发者指南）</b></summary>

evolvebot 使用 **Provider Registry**（`evolvebot/providers/registry.py`）作为单一来源。
新增 Provider 仅需 **2 步** — 无需修改 if-elif 链。

**步骤 1.** 在 `evolvebot/providers/registry.py` 的 `PROVIDERS` 中添加 `ProviderSpec`：

```python
ProviderSpec(
    name="myprovider",                   # config 字段名
    keywords=("myprovider", "mymodel"),  # 用于模型名自动匹配的关键字
    env_key="MYPROVIDER_API_KEY",        # LiteLLM 环境变量
    display_name="My Provider",          # `evolvebot status` 展示名称
    litellm_prefix="myprovider",         # 自动前缀：model → myprovider/model
    skip_prefixes=("myprovider/",),       # 已带前缀则不重复
)
```

**步骤 2.** 在 `evolvebot/config/schema.py` 的 `ProvidersConfig` 中添加字段：

```python
class ProvidersConfig(BaseModel):
    ...
    myprovider: ProviderConfig = ProviderConfig()
```

完成！环境变量、模型前缀、配置匹配与 `evolvebot status` 展示都会自动生效。

**常见 `ProviderSpec` 选项：**

| 字段 | 说明 | 示例 |
|-------|-------------|---------|
| `litellm_prefix` | LiteLLM 自动前缀 | `"dashscope"` → `dashscope/qwen-max` |
| `skip_prefixes` | 已有这些前缀则不再加 | `("dashscope/", "openrouter/")` |
| `env_extras` | 额外需要设置的环境变量 | `(("ZHIPUAI_API_KEY", "{api_key}"),)` |
| `model_overrides` | 单模型参数覆盖 | `(("kimi-k2.5", {"temperature": 1.0}),)` |
| `is_gateway` | 是否为网关（如 OpenRouter） | `True` |
| `detect_by_key_prefix` | 通过 API Key 前缀判断网关 | `"sk-or-"` |
| `detect_by_base_keyword` | 通过 API Base 关键词判断 | `"openrouter"` |
| `strip_model_prefix` | 重新加前缀前先剥离旧前缀 | `True`（AiHubMix） |

</details>


### MCP（Model Context Protocol）

> [!TIP]
> 配置格式与 Claude Desktop / Cursor 兼容，可直接从任意 MCP Server 的 README 复制。

evolvebot 支持 [MCP](https://modelcontextprotocol.io/) — 连接外部工具服务器并作为原生工具使用。

在 `config.json` 中添加 MCP Server：

```json
{
  "tools": {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
      }
    }
  }
}
```

支持两种传输模式：

| 模式 | 配置 | 示例 |
|------|--------|---------|
| **Stdio** | `command` + `args` | 本地进程（`npx` / `uvx`） |
| **HTTP** | `url` | 远端端点（`https://mcp.example.com/sse`） |

MCP 工具会在启动时自动发现并注册，LLM 可与内置工具一同使用 — 无需额外配置。




### 安全

> [!TIP]
> 生产部署建议在配置中设置 `"restrictToWorkspace": true` 来沙箱化 Agent。

| 选项 | 默认值 | 说明 |
|--------|---------|-------------|
| `tools.restrictToWorkspace` | `false` | 设为 `true` 时，限制 **所有** agent 工具（shell、文件读写/编辑、list）只能访问工作区目录，防止路径穿越和越界访问。 |
| `channels.*.allowFrom` | `[]`（允许所有） | 用户白名单。空数组表示允许所有人；非空表示仅列表内用户可用。 |


## CLI 参考

| 命令 | 说明 |
|---------|-------------|
| `evolvebot onboard` | 初始化配置与工作区 |
| `evolvebot agent -m "..."` | 单次对话 |
| `evolvebot agent` | 交互式对话 |
| `evolvebot agent --no-markdown` | 显示纯文本回复 |
| `evolvebot agent --logs` | 对话时显示运行日志 |
| `evolvebot gateway` | 启动网关 |
| `evolvebot status` | 查看状态 |
| `evolvebot provider login openai-codex` | Provider OAuth 登录 |
| `evolvebot channels login` | 连接 WhatsApp（扫码） |
| `evolvebot channels status` | 查看渠道状态 |

交互模式退出指令：`exit`、`quit`、`/exit`、`/quit`、`:q` 或 `Ctrl+D`。

<details>
<summary><b>定时任务（Cron）</b></summary>

```bash
# 添加任务
evolvebot cron add --name "daily" --message "Good morning!" --cron "0 9 * * *"
evolvebot cron add --name "hourly" --message "Check status" --every 3600

# 列出任务
evolvebot cron list

# 删除任务
evolvebot cron remove <job_id>
```

</details>

## 🐳 Docker

> [!TIP]
> `-v ~/.evolvebot:/root/.evolvebot` 会将本地配置目录挂载到容器中，保证配置与工作区在容器重启后仍然保留。

构建并运行 evolvebot 容器：

```bash
# 构建镜像
docker build -t evolvebot .

# 初始化配置（仅首次）
docker run -v ~/.evolvebot:/root/.evolvebot --rm evolvebot onboard

# 在宿主机编辑配置以添加 API key
vim ~/.evolvebot/config.json

# 运行 gateway（连接已启用的渠道，如 Telegram/Discord/Mochat）
docker run -v ~/.evolvebot:/root/.evolvebot -p 18790:18790 evolvebot gateway

# 或运行单条命令
docker run -v ~/.evolvebot:/root/.evolvebot --rm evolvebot agent -m "Hello!"
docker run -v ~/.evolvebot:/root/.evolvebot --rm evolvebot status
```

## 📁 项目结构

```
evolvebot/
├── agent/          # 🧠 核心代理逻辑
│   ├── loop.py     #    代理循环（LLM ↔ 工具执行）
│   ├── context.py  #    Prompt 构建器
│   ├── memory.py   #    持久化记忆
│   ├── skills.py   #    技能加载器
│   ├── subagent.py #    后台任务执行
│   └── tools/      #    内置工具（含 spawn）
├── skills/         # 🎯 内置技能（github、weather、tmux...）
├── channels/       # 📱 聊天渠道集成
├── bus/            # 🚌 消息路由
├── cron/           # ⏰ 定时任务
├── heartbeat/      # 💓 主动唤醒
├── providers/      # 🤖 LLM Providers（OpenRouter 等）
├── session/        # 💬 会话管理
├── config/         # ⚙️ 配置
└── cli/            # 🖥️ 命令行
```

## 🤝 贡献与路线图

欢迎 PR！代码库刻意保持小而清晰，便于阅读与协作。🤗

**路线图** — 选一个方向并 [提交 PR](https://github.com/ZhengJiandan/evolvebot/pulls)！

- [ ] **多模态** — 图像、语音、视频
- [ ] **长期记忆** — 永不遗忘重要上下文
- [ ] **更强推理** — 多步规划与反思
- [ ] **更多集成** — 日历等
- [ ] **自我改进** — 从反馈与错误中学习

### 贡献者

<a href="https://github.com/ZhengJiandan/evolvebot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ZhengJiandan/evolvebot&max=100&columns=12&updated=20260210" alt="Contributors" />
</a>


## ⭐ Star History

<div align="center">
  <a href="https://star-history.com/#ZhengJiandan/evolvebot&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ZhengJiandan/evolvebot&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ZhengJiandan/evolvebot&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ZhengJiandan/evolvebot&type=Date" style="border-radius: 15px; box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);" />
    </picture>
  </a>
</div>

<p align="center">
  <em> 感谢访问 ✨ evolvebot！</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=ZhengJiandan.evolvebot&style=for-the-badge&color=00d4ff" alt="Views">
</p>


<p align="center">
  <sub>evolvebot 仅用于教育、研究与技术交流目的</sub>
</p>
