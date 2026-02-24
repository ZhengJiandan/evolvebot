"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class WhatsAppConfig(BaseModel):
    """WhatsApp channel configuration."""
    enabled: bool = False
    bridge_url: str = "ws://localhost:3001"
    bridge_token: str = ""  # Shared token for bridge auth (optional, recommended)
    allow_from: list[str] = Field(default_factory=list)  # Allowed phone numbers


class TelegramConfig(BaseModel):
    """Telegram channel configuration."""
    enabled: bool = False
    token: str = ""  # Bot token from @BotFather
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs or usernames
    proxy: str | None = None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"


class FeishuConfig(BaseModel):
    """Feishu/Lark channel configuration using WebSocket long connection."""
    enabled: bool = False
    app_id: str = ""  # App ID from Feishu Open Platform
    app_secret: str = ""  # App Secret from Feishu Open Platform
    encrypt_key: str = ""  # Encrypt Key for event subscription (optional)
    verification_token: str = ""  # Verification Token for event subscription (optional)
    allow_from: list[str] = Field(default_factory=list)  # Allowed user open_ids


class DingTalkConfig(BaseModel):
    """DingTalk channel configuration using Stream mode."""
    enabled: bool = False
    client_id: str = ""  # AppKey
    client_secret: str = ""  # AppSecret
    allow_from: list[str] = Field(default_factory=list)  # Allowed staff_ids


class DiscordConfig(BaseModel):
    """Discord channel configuration."""
    enabled: bool = False
    token: str = ""  # Bot token from Discord Developer Portal
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs
    gateway_url: str = "wss://gateway.discord.gg/?v=10&encoding=json"
    intents: int = 37377  # GUILDS + GUILD_MESSAGES + DIRECT_MESSAGES + MESSAGE_CONTENT

class EmailConfig(BaseModel):
    """Email channel configuration (IMAP inbound + SMTP outbound)."""
    enabled: bool = False
    consent_granted: bool = False  # Explicit owner permission to access mailbox data

    # IMAP (receive)
    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password: str = ""
    imap_mailbox: str = "INBOX"
    imap_use_ssl: bool = True

    # SMTP (send)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    from_address: str = ""

    # Behavior
    auto_reply_enabled: bool = True  # If false, inbound email is read but no automatic reply is sent
    poll_interval_seconds: int = 30
    mark_seen: bool = True
    max_body_chars: int = 12000
    subject_prefix: str = "Re: "
    allow_from: list[str] = Field(default_factory=list)  # Allowed sender email addresses


class MochatMentionConfig(BaseModel):
    """Mochat mention behavior configuration."""
    require_in_groups: bool = False


class MochatGroupRule(BaseModel):
    """Mochat per-group mention requirement."""
    require_mention: bool = False


class MochatConfig(BaseModel):
    """Mochat channel configuration."""
    enabled: bool = False
    base_url: str = "https://mochat.io"
    socket_url: str = ""
    socket_path: str = "/socket.io"
    socket_disable_msgpack: bool = False
    socket_reconnect_delay_ms: int = 1000
    socket_max_reconnect_delay_ms: int = 10000
    socket_connect_timeout_ms: int = 10000
    refresh_interval_ms: int = 30000
    watch_timeout_ms: int = 25000
    watch_limit: int = 100
    retry_delay_ms: int = 500
    max_retry_attempts: int = 0  # 0 means unlimited retries
    claw_token: str = ""
    agent_user_id: str = ""
    sessions: list[str] = Field(default_factory=list)
    panels: list[str] = Field(default_factory=list)
    allow_from: list[str] = Field(default_factory=list)
    mention: MochatMentionConfig = Field(default_factory=MochatMentionConfig)
    groups: dict[str, MochatGroupRule] = Field(default_factory=dict)
    reply_delay_mode: str = "non-mention"  # off | non-mention
    reply_delay_ms: int = 120000


class SlackDMConfig(BaseModel):
    """Slack DM policy configuration."""
    enabled: bool = True
    policy: str = "open"  # "open" or "allowlist"
    allow_from: list[str] = Field(default_factory=list)  # Allowed Slack user IDs


class SlackConfig(BaseModel):
    """Slack channel configuration."""
    enabled: bool = False
    mode: str = "socket"  # "socket" supported
    webhook_path: str = "/slack/events"
    bot_token: str = ""  # xoxb-...
    app_token: str = ""  # xapp-...
    user_token_read_only: bool = True
    group_policy: str = "mention"  # "mention", "open", "allowlist"
    group_allow_from: list[str] = Field(default_factory=list)  # Allowed channel IDs if allowlist
    dm: SlackDMConfig = Field(default_factory=SlackDMConfig)


class QQConfig(BaseModel):
    """QQ channel configuration using botpy SDK."""
    enabled: bool = False
    app_id: str = ""  # 机器人 ID (AppID) from q.qq.com
    secret: str = ""  # 机器人密钥 (AppSecret) from q.qq.com
    allow_from: list[str] = Field(default_factory=list)  # Allowed user openids (empty = public access)


class ChannelsConfig(BaseModel):
    """Configuration for chat channels."""
    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    mochat: MochatConfig = Field(default_factory=MochatConfig)
    dingtalk: DingTalkConfig = Field(default_factory=DingTalkConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    qq: QQConfig = Field(default_factory=QQConfig)


class AgentDefaults(BaseModel):
    """Default agent configuration."""
    workspace: str = "~/.evolvebot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20
    memory_window: int = 50


class AgentsConfig(BaseModel):
    """Agent configuration."""
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""
    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)


class ProvidersConfig(BaseModel):
    """Configuration for LLM providers."""
    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)  # 阿里云通义千问
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenAI Codex (OAuth)


class GatewayConfig(BaseModel):
    """Gateway/server configuration."""
    host: str = "0.0.0.0"
    port: int = 18790


class WebSearchConfig(BaseModel):
    """Web search tool configuration."""
    api_key: str = ""  # Brave Search API key
    max_results: int = 5


class WebToolsConfig(BaseModel):
    """Web tools configuration."""
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(BaseModel):
    """Shell exec tool configuration."""
    timeout: int = 60


class MCPServerConfig(BaseModel):
    """MCP server connection configuration (stdio or HTTP)."""
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP: streamable HTTP endpoint URL


class ToolsConfig(BaseModel):
    """Tools configuration."""
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class UniverseMembership(BaseModel):
    """A joined Universe network (typically an org hub)."""

    enabled: bool = True
    org_id: str = ""  # e.g. "acme", "family", "lab"
    hub_url: str = ""  # e.g. "ws://hub.example.com:18888"
    join_secret: str = ""  # org join secret / invite token (stored locally)
    label: str = ""  # optional UI label


class UniverseShareProfile(BaseModel):
    """Local, per-friend sharing policy (what *this* node is willing to share)."""

    share_all: bool = False
    share_knowledge: bool = False
    share_skills: bool = False
    allow_task_help: bool = False


class UniverseConfig(BaseModel):
    """Universe network configuration (org/public hubs)."""

    enabled: bool = False
    node_id: str = ""  # stable identifier for this evolvebot node
    node_name: str = ""  # optional display name

    # Joined networks (default: org hubs). Public network is also just a membership with its own policy.
    memberships: list[UniverseMembership] = Field(default_factory=list)

    # Local sharing policy keyed by friend node_id (friendship itself is managed by the hub).
    share_profiles: dict[str, UniverseShareProfile] = Field(default_factory=dict)

    # Public universe: registry discovery + direct node-to-node calls (opt-in).
    public_enabled: bool = False
    public_registry_url: str = "ws://82.157.31.6:18999"
    public_provide_service: bool = False
    public_service_host: str = "0.0.0.0"
    public_service_port: int = 18998
    public_service_token: str = ""  # optional shared secret required by this node for task calls
    public_registry_token: str = ""  # optional token required by the registry for registration
    public_price_points: int = 1  # points awarded per completed task (MVP bookkeeping)
    public_max_tokens: int = 1024  # hard cap for remote tasks
    public_capabilities: dict[str, Any] = Field(default_factory=lambda: {"llm.chat": True, "echo": True})
    public_auto_register: bool = False
    public_advertise_url: str = ""  # if set, overrides auto detection (e.g. ws://example.com:18998)
    public_advertise_host: str = ""  # optional host for advertise URL (uses public_service_port)
    public_advertise_port: int | None = None  # optional override for advertise port
    public_detect_public_ip: bool = False  # if true, use external IP detection
    public_detect_ip_service: str = "https://api.ipify.org"
    public_rate_limit_per_min: int = 60  # per client IP
    public_rate_limit_burst: int = 60
    public_rate_limit_per_min_by_node: int = 60  # per client nodeId (if provided)
    public_rate_limit_burst_by_node: int = 60
    public_self_check_enabled: bool = True
    public_self_check_timeout_s: float = 3.0
    public_capability_card: dict[str, Any] = Field(default_factory=dict)
    public_tools: list[str] = Field(default_factory=list)
    public_models: list[str] = Field(default_factory=list)
    public_languages: list[str] = Field(default_factory=list)
    public_relay_url: str = "ws://82.157.31.6:19001"  # ws://relay-host:port
    public_relay_token: str = ""
    public_relay_only: bool = False
    public_preauth_enabled: bool = True
    public_preauth_required: bool = False
    public_capability_vocab: list[str] = Field(
        default_factory=lambda: ["web_search", "web_fetch", "exec", "evolvebot.agent", "llm.chat"]
    )
    public_capability_aliases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "web_search": ["search", "lookup", "web_search", "搜", "搜索", "查", "查询", "最新", "新闻"],
            "web_fetch": ["fetch", "web_fetch", "抓取", "网页", "url", "http", "https"],
            "exec": ["exec", "shell", "bash", "cmd", "命令", "运行", "执行", "脚本"],
        }
    )
    public_capability_llm_enabled: bool = True
    public_knowledge_auto_pull: bool = False
    public_knowledge_auto_pull_limit: int = 1
    public_knowledge_auto_pull_tagged_only: bool = True
    public_knowledge_inbox_dir: str = ""
    public_knowledge_publish_dir: str = "~/.evolvebot/knowledge_packs"
    public_knowledge_publish_interval_s: int = 300

    # Learning (curiosity / task-driven / review)
    knowledge_curiosity_enabled: bool = False
    knowledge_curiosity_interval_s: int = 86400
    knowledge_curiosity_daily_limit: int = 2
    knowledge_curiosity_tags: list[str] = Field(default_factory=list)

    knowledge_task_driven_enabled: bool = False
    knowledge_task_driven_max_per_task: int = 1
    knowledge_task_driven_tagged_only: bool = True

    knowledge_review_enabled: bool = False
    knowledge_review_interval_s: int = 86400
    knowledge_review_min_tasks: int = 10
    knowledge_review_max_tasks: int = 20
    knowledge_review_publish: bool = True
    knowledge_review_min_content_chars: int = 300
    knowledge_review_publish_max_retries: int = 3
    knowledge_review_llm_enabled: bool = True
    knowledge_review_llm_gate_enabled: bool = True
    knowledge_review_llm_gate_min_score: int = 70
    knowledge_review_llm_max_tokens: int = 512
    knowledge_review_llm_temperature: float = 0.2

    # Learning quality controls
    knowledge_learned_ids_limit: int = 2000
    knowledge_learning_min_score: float = 10.0

    # Public service behavior
    public_allow_agent_tasks: bool = False  # allow kind="evolvebot.agent"
    public_agent_tool_allowlist: list[str] = Field(default_factory=lambda: ["web_search", "web_fetch"])
    public_agent_max_iterations: int = 8

    # Client-side auto delegation
    public_auto_delegate_enabled: bool = False
    public_auto_delegate_on_tool_error: bool = True
    public_auto_delegate_kind: str = "evolvebot.agent"
    public_auto_delegate_max_price_points: int | None = None
    public_auto_delegate_debug: bool = False


class Config(BaseSettings):
    """Root configuration for evolvebot."""
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    
    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()
    
    def _match_provider(self, model: str | None = None) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from evolvebot.providers.registry import PROVIDERS
        model_lower = (model or self.agents.defaults.model).lower()

        # Match by keyword (order follows PROVIDERS registry)
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(kw in model_lower for kw in spec.keywords):
                if spec.is_oauth or p.api_key:
                    return p, spec.name

        # Fallback: gateways first, then others (follows registry order)
        # OAuth providers are NOT valid fallbacks — they require explicit model selection
        for spec in PROVIDERS:
            if spec.is_oauth:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_key:
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        return p.api_key if p else None
    
    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for known gateways."""
        from evolvebot.providers.registry import find_by_name
        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # Only gateways get a default api_base here. Standard providers
        # (like Moonshot) set their base URL via env vars in _setup_env
        # to avoid polluting the global litellm.api_base.
        if name:
            spec = find_by_name(name)
            if spec and spec.is_gateway and spec.default_api_base:
                return spec.default_api_base
        return None
    
    model_config = ConfigDict(
        env_prefix="EVOLVEBOT_",
        env_nested_delimiter="__"
    )
