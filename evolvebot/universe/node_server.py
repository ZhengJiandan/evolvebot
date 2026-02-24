"""Public Universe Node service (direct task execution endpoint).

MVP supports `llm.chat`, `echo`, and (optionally) `evolvebot.agent` tasks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

import websockets
from websockets.server import WebSocketServerProtocol

from evolvebot.universe.protocol import Envelope, make_envelope
from evolvebot.universe.ratelimit import RateLimiter
from evolvebot.universe.task_executor import TaskExecutor


@dataclass
class NodeServerConfig:
    host: str = "0.0.0.0"
    port: int = 18998
    service_token: str = ""  # if set, required for task_run
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 60
    rate_limit_per_min_by_node: int = 60
    rate_limit_burst_by_node: int = 60


class NodeServer:
    def __init__(self, cfg: NodeServerConfig | None = None) -> None:
        self.cfg = cfg or NodeServerConfig()
        self.bound_port: int = self.cfg.port
        self._server: websockets.server.Serve | None = None
        self._limiter = RateLimiter(self.cfg.rate_limit_per_min, self.cfg.rate_limit_burst)
        self._node_limiter = RateLimiter(self.cfg.rate_limit_per_min_by_node, self.cfg.rate_limit_burst_by_node)
        self._executor = TaskExecutor()

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, self.cfg.host, self.cfg.port, ping_interval=20, ping_timeout=20)
        try:
            if self._server.sockets:
                self.bound_port = int(self._server.sockets[0].getsockname()[1])
        except Exception:
            self.bound_port = self.cfg.port
        logger.info(f"Universe node service listening on ws://{self.cfg.host}:{self.bound_port}")

    async def stop(self) -> None:
        if not self._server:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    def _check_token(self, provided: str) -> bool:
        if not self.cfg.service_token:
            return True
        return provided == self.cfg.service_token

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        try:
            async for raw in ws:
                try:
                    env = Envelope.from_json(raw)
                except Exception as e:
                    await ws.send(make_envelope("error", payload={"message": f"bad json: {e}"}).to_json())
                    continue

                if not self._allow_client(ws, env):
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "rate limited"}).to_json())
                    continue

                if env.type == "ping":
                    await ws.send(make_envelope("pong", id=env.id).to_json())
                    continue

                if env.type != "task_run":
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "expected task_run"}).to_json())
                    continue

                token = (env.payload or {}).get("serviceToken", "")
                if not self._check_token(token):
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "invalid service token"}).to_json())
                    continue

                kind = (env.payload or {}).get("kind", "")
                if kind not in {"llm.chat", "echo", "evolvebot.agent"}:
                    await ws.send(
                        make_envelope("error", id=env.id, payload={"message": f"unsupported kind: {kind}"}).to_json()
                    )
                    continue

                prompt = (env.payload or {}).get("prompt", "")
                if not prompt:
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "missing prompt"}).to_json())
                    continue

                client_id = (env.payload or {}).get("clientId") or ""
                try:
                    peer = ws.remote_address[0] if ws.remote_address else "unknown"
                except Exception:
                    peer = "unknown"
                logger.info(
                    "Universe node received task: kind=%s client_id=%s from=%s",
                    kind,
                    client_id or "-",
                    peer,
                )

                try:
                    result = await self._executor.run(kind, prompt)
                    await ws.send(make_envelope("task_result", id=env.id, payload={"content": result}).to_json())
                except Exception as e:
                    await ws.send(make_envelope("task_error", id=env.id, payload={"message": str(e)}).to_json())
        except websockets.ConnectionClosed:
            return

    def _allow_client(self, ws: WebSocketServerProtocol, env: Envelope) -> bool:
        key = "unknown"
        try:
            if ws.remote_address and ws.remote_address[0]:
                key = str(ws.remote_address[0])
        except Exception:
            pass
        if not self._limiter.allow(key):
            return False
        client_id = None
        try:
            if env.payload and isinstance(env.payload, dict):
                client_id = env.payload.get("clientId") or env.payload.get("client_id")
        except Exception:
            client_id = None
        if client_id:
            return self._node_limiter.allow(str(client_id))
        return True

    async def _run_llm_chat(self, prompt: str) -> str:
        # Deprecated: kept for compatibility with older imports.
        return await self._executor.run("llm.chat", prompt)

    async def _run_remote_agent(self, prompt: str) -> str:
        # Deprecated: kept for compatibility with older imports.
        return await self._executor.run("evolvebot.agent", prompt)
