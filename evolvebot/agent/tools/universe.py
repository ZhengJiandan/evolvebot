"""Universe tool: delegate work to public universe nodes."""

from __future__ import annotations

from typing import Any

from evolvebot.agent.tools.base import Tool
from evolvebot.config.loader import load_config
from evolvebot.universe.public_client import delegate_task


class UniverseHelpTool(Tool):
    name = "universe_help"
    description = (
        "Ask the public evolvebot universe for help by delegating a task to another node. "
        "Use when local tools/keys/resources are missing or the task is blocked."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The task to delegate"},
            "kind": {"type": "string", "enum": ["evolvebot.agent", "llm.chat", "echo"], "default": "evolvebot.agent"},
            "requireCapability": {"type": "string", "description": "Capability key required (default: kind)"},
            "toNodeId": {"type": "string", "description": "Optional specific node_id to call"},
            "serviceToken": {"type": "string", "description": "Service token if required by the provider node"},
            "maxPricePoints": {"type": "integer", "minimum": 1, "description": "Optional max price points"},
        },
        "required": ["prompt"],
    }

    async def execute(
        self,
        prompt: str,
        kind: str = "evolvebot.agent",
        requireCapability: str | None = None,
        toNodeId: str | None = None,
        serviceToken: str = "",
        maxPricePoints: int | None = None,
        **kwargs: Any,
    ) -> str:
        cfg = load_config()
        if not cfg.universe.public_enabled:
            return "Error: Public universe is disabled. Enable it with `evolvebot universe public enable`."

        try:
            node, out = await delegate_task(
                registry_url=cfg.universe.public_registry_url,
                kind=kind,
                prompt=prompt,
                require_capability=requireCapability,
                to_node_id=toNodeId,
                service_token=serviceToken,
                max_price_points=maxPricePoints,
                client_id=cfg.universe.node_id or None,
                registry_token=cfg.universe.public_registry_token or None,
                relay_url=cfg.universe.public_relay_url or None,
                relay_token=cfg.universe.public_relay_token or "",
                relay_only=bool(cfg.universe.public_relay_only),
                preauth_enabled=bool(cfg.universe.public_preauth_enabled),
                preauth_required=bool(cfg.universe.public_preauth_required),
            )
            return f"[universe:{node.node_id}] {out}"
        except Exception as e:
            return f"Error: universe delegation failed: {e}"
