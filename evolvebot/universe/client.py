from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

from evolvebot.universe.protocol import Envelope, make_envelope


class UniverseClient:
    """Thin async client for a Universe hub."""

    def __init__(
        self,
        *,
        hub_url: str,
        org_id: str,
        join_secret: str,
        node_id: str,
        node_name: str = "",
        capabilities: dict[str, Any] | None = None,
    ) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.org_id = org_id
        self.join_secret = join_secret
        self.node_id = node_id
        self.node_name = node_name
        self.capabilities = capabilities or {}

        self._ws: WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[Envelope]] = {}
        self.inbox: asyncio.Queue[Envelope] = asyncio.Queue()

    async def connect(self) -> None:
        if self._ws is not None:
            return
        self._ws = await websockets.connect(self.hub_url)
        hello = make_envelope(
            "hello",
            org_id=self.org_id,
            from_node=self.node_id,
            payload={
                "orgId": self.org_id,
                "joinSecret": self.join_secret,
                "nodeId": self.node_id,
                "nodeName": self.node_name,
                "capabilities": self.capabilities,
            },
        )
        await self._ws.send(hello.to_json())
        raw = await self._ws.recv()
        env = Envelope.from_json(raw)
        if env.type != "hello_ok":
            raise RuntimeError(f"hub rejected hello: {env.payload}")

        self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._recv_task
            self._recv_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> "UniverseClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def request(self, msg_type: str, payload: dict[str, Any] | None = None) -> Envelope:
        if not self._ws:
            raise RuntimeError("not connected")
        env = make_envelope(msg_type, org_id=self.org_id, from_node=self.node_id, payload=payload or {})
        fut: asyncio.Future[Envelope] = asyncio.get_running_loop().create_future()
        self._pending[env.id] = fut
        await self._ws.send(env.to_json())
        return await asyncio.wait_for(fut, timeout=10.0)

    async def send_friend_request(self, to_node: str, message: str = "") -> str:
        resp = await self.request("friend_request", {"toNode": to_node, "message": message})
        if resp.type == "error":
            raise RuntimeError(resp.payload.get("message", "error"))
        return (resp.payload or {}).get("requestId", "")

    async def accept_friend_request(self, request_id: str) -> None:
        resp = await self.request("friend_accept", {"requestId": request_id})
        if resp.type == "error":
            raise RuntimeError(resp.payload.get("message", "error"))

    async def dm(self, to_node: str, content: str) -> None:
        resp = await self.request("dm", {"toNode": to_node, "content": content})
        if resp.type == "error":
            raise RuntimeError(resp.payload.get("message", "error"))

    async def list_nodes(self) -> list[dict[str, Any]]:
        resp = await self.request("list_nodes")
        return (resp.payload or {}).get("nodes", [])

    async def get_friends(self) -> list[str]:
        resp = await self.request("get_friends")
        return (resp.payload or {}).get("friends", [])

    async def get_pending_friend_requests(self) -> list[dict[str, Any]]:
        resp = await self.request("get_pending_friend_requests")
        return (resp.payload or {}).get("requests", [])

    async def listen(self, handler: Callable[[Envelope], Awaitable[None]] | None = None) -> None:
        """Listen for push events (presence, friend_request_deliver, dm_deliver, friend_added)."""
        while True:
            env = await self.inbox.get()
            if handler is None:
                continue
            await handler(env)

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        ws = self._ws
        async for raw in ws:
            env = Envelope.from_json(raw)
            fut = self._pending.pop(env.id, None)
            if fut is not None and not fut.done():
                fut.set_result(env)
                continue
            # Push message (deliver/presence)
            await self.inbox.put(env)


import contextlib  # placed at end to keep imports minimal for type checkers
