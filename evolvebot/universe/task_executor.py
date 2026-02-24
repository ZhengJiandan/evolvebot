"""Shared task execution logic for node service and relay."""

from __future__ import annotations

from dataclasses import dataclass

from evolvebot.config.loader import load_config
from evolvebot.agent.tools.registry import ToolRegistry
from evolvebot.agent.tools.web import WebFetchTool, WebSearchTool
from evolvebot.providers.litellm_provider import LiteLLMProvider
from evolvebot.universe.remote_agent import RemoteAgent, RemoteAgentConfig


@dataclass
class TaskExecutorConfig:
    allow_agent_tasks: bool
    max_tokens: int
    agent_max_iterations: int


class TaskExecutor:
    def __init__(self, cfg: TaskExecutorConfig | None = None) -> None:
        base = load_config()
        self._cfg = cfg or TaskExecutorConfig(
            allow_agent_tasks=base.universe.public_allow_agent_tasks,
            max_tokens=int(base.universe.public_max_tokens or 1024),
            agent_max_iterations=int(base.universe.public_agent_max_iterations or 8),
        )

    async def run(self, kind: str, prompt: str) -> str:
        if kind == "echo":
            return prompt
        if kind == "evolvebot.agent":
            return await self._run_remote_agent(prompt)
        if kind == "llm.chat":
            return await self._run_llm_chat(prompt)
        raise RuntimeError(f"unsupported kind: {kind}")

    async def _run_llm_chat(self, prompt: str) -> str:
        cfg = load_config()
        model = cfg.agents.defaults.model
        provider_cfg, provider_name = cfg._match_provider(model)
        if not provider_cfg:
            raise RuntimeError("No provider configured for model. Set providers.*.apiKey in ~/.evolvebot/config.json")

        provider = LiteLLMProvider(
            api_key=provider_cfg.api_key,
            api_base=provider_cfg.api_base,
            default_model=model,
            extra_headers=provider_cfg.extra_headers,
            provider_name=provider_name,
        )

        max_tokens = min(int(self._cfg.max_tokens or 1024), 2048)
        resp = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            model=model,
            max_tokens=max_tokens,
            temperature=cfg.agents.defaults.temperature,
        )
        return resp.content or ""

    async def _run_remote_agent(self, prompt: str) -> str:
        cfg = load_config()
        if not self._cfg.allow_agent_tasks:
            raise RuntimeError("This node does not allow evolvebot.agent tasks.")

        model = cfg.agents.defaults.model
        provider_cfg, provider_name = cfg._match_provider(model)
        if not provider_cfg:
            raise RuntimeError("No provider configured for model. Set providers.*.apiKey in ~/.evolvebot/config.json")

        provider = LiteLLMProvider(
            api_key=provider_cfg.api_key,
            api_base=provider_cfg.api_base,
            default_model=model,
            extra_headers=provider_cfg.extra_headers,
            provider_name=provider_name,
        )

        allow = set(cfg.universe.public_agent_tool_allowlist or [])
        tools = ToolRegistry()
        if "web_search" in allow:
            tools.register(WebSearchTool(api_key=cfg.tools.web.search.api_key or None))
        if "web_fetch" in allow:
            tools.register(WebFetchTool())

        agent = RemoteAgent(
            provider=provider,
            tools=tools,
            cfg=RemoteAgentConfig(
                model=model,
                max_iterations=int(self._cfg.agent_max_iterations or 8),
                temperature=cfg.agents.defaults.temperature,
                max_tokens=min(int(self._cfg.max_tokens or 1024), 2048),
            ),
        )
        return await agent.run(prompt)
