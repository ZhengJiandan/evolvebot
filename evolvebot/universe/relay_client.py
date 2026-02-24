"""Relay client for provider nodes (executes tasks via relay)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger
import websockets

from evolvebot.universe.protocol import Envelope, make_envelope
from evolvebot.universe.ratelimit import RateLimiter
from evolvebot.universe.task_executor import TaskExecutor


@dataclass
class RelayNodeClientConfig:
    relay_url: str
    node_id: str
    relay_token: str = ""
    service_token: str = ""
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 60
    rate_limit_per_min_by_node: int = 60
    rate_limit_burst_by_node: int = 60


class RelayNodeClient:
    def __init__(self, cfg: RelayNodeClientConfig) -> None:
        self.cfg = cfg
        self._limiter = RateLimiter(cfg.rate_limit_per_min, cfg.rate_limit_burst)
        self._node_limiter = RateLimiter(cfg.rate_limit_per_min_by_node, cfg.rate_limit_burst_by_node)
        self._executor = TaskExecutor()

    async def run_forever(self) -> None:
        backoff = 1.0
        while True:
            try:
                await self._run_once()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"relay client: connection failed ({e}); retrying in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _run_once(self) -> None:
        if not self.cfg.relay_url:
            raise RuntimeError("relay_url required")
        async with websockets.connect(self.cfg.relay_url) as ws:
            await self._hello(ws)
            async for raw in ws:
                try:
                    env = Envelope.from_json(raw)
                except Exception as e:
                    await ws.send(make_envelope("error", payload={"message": f"bad json: {e}"}).to_json())
                    continue

                if env.type == "ping":
                    await ws.send(make_envelope("pong", id=env.id).to_json())
                    continue

                if env.type != "relay_task":
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "expected relay_task"}).to_json())
                    continue

                await self._handle_task(ws, env)

    async def _hello(self, ws: Any) -> None:
        req = make_envelope(
            "relay_hello",
            payload={"nodeId": self.cfg.node_id, "relayToken": self.cfg.relay_token},
        )
        await ws.send(req.to_json())
        resp = Envelope.from_json(await ws.recv())
        if resp.type == "error":
            raise RuntimeError((resp.payload or {}).get("message", "relay hello failed"))
        if resp.type != "relay_hello_ok":
            raise RuntimeError("relay hello rejected")

    async def _handle_task(self, ws: Any, env: Envelope) -> None:
        if not self._allow_client(env):
            await self._send_result(ws, env, ok=False, message="rate limited")
            return

        payload = env.payload if isinstance(env.payload, dict) else {}
        token = payload.get("serviceToken", "")
        if not self._check_token(token):
            await self._send_result(ws, env, ok=False, message="invalid service token")
            return

        kind = payload.get("kind", "")
        if kind not in {"llm.chat", "echo", "evolvebot.agent"}:
            await self._send_result(ws, env, ok=False, message=f"unsupported kind: {kind}")
            return

        prompt = payload.get("prompt", "")
        if not prompt:
            await self._send_result(ws, env, ok=False, message="missing prompt")
            return

        try:
            result = await self._executor.run(kind, prompt)
            await self._send_result(ws, env, ok=True, content=result)
        except Exception as e:
            await self._send_result(ws, env, ok=False, message=str(e))

    async def _send_result(
        self, ws: Any, env: Envelope, *, ok: bool, content: str | None = None, message: str | None = None
    ) -> None:
        payload = {"nodeId": self.cfg.node_id, "ok": ok}
        if ok:
            payload["content"] = content or ""
        else:
            payload["message"] = message or "task failed"
        await ws.send(make_envelope("relay_result", id=env.id, payload=payload).to_json())

    def _check_token(self, provided: str) -> bool:
        if not self.cfg.service_token:
            return True
        return provided == self.cfg.service_token

    def _allow_client(self, env: Envelope) -> bool:
        if not self._limiter.allow("relay"):
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
