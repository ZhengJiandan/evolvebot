from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from evolvebot.agent.tools.registry import ToolRegistry
from evolvebot.providers.base import LLMProvider


@dataclass
class RemoteAgentConfig:
    model: str
    max_iterations: int = 8
    temperature: float = 0.7
    max_tokens: int = 1024


class RemoteAgent:
    """A minimal tool-using agent loop for remote execution.

    This is intentionally narrower than the main AgentLoop:
    - no filesystem tools
    - no shell tools
    - no spawn/subagents
    - no memory/workspace context
    """

    def __init__(self, provider: LLMProvider, tools: ToolRegistry, cfg: RemoteAgentConfig) -> None:
        self.provider = provider
        self.tools = tools
        self.cfg = cfg

    async def run(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful remote evolvebot agent. "
                    "Solve the user's request. You MAY use the available tools if needed. "
                    "Keep the answer concise and directly usable."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for _ in range(self.cfg.max_iterations):
            resp = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )

            if not resp.has_tool_calls:
                return resp.content or ""

            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in resp.tool_calls
            ]
            messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": tool_call_dicts})

            for tc in resp.tool_calls:
                args_str = json.dumps(tc.arguments, ensure_ascii=False)
                logger.info(f"remote tool call: {tc.name}({args_str[:200]})")
                result = await self.tools.execute(tc.name, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})

            messages.append({"role": "user", "content": "Continue with the task using the tool results."})

        return "I couldn't complete the task within the remote iteration limit."

