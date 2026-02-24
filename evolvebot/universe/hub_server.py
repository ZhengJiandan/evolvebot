"""Universe Hub server (org-scoped routing + friend graph).

MVP features:
- org join with shared join secret
- presence (online/offline, capabilities)
- explicit mutual friend add (request/accept)
- dm between friends

Security note: the MVP does not implement end-to-end encryption.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

import websockets
from websockets.server import WebSocketServerProtocol

from evolvebot.universe.hub_state import HubState, NodeConn, NodeInfo
from evolvebot.universe.protocol import Envelope, make_envelope
from evolvebot.universe.ratelimit import RateLimiter


@dataclass
class HubServerConfig:
    host: str = "0.0.0.0"
    port: int = 18888
    path: str = "/ws"
    hello_timeout_s: float = 10.0
    rate_limit_per_min: int = 120
    rate_limit_burst: int = 120
    max_message_bytes: int = 65536
    offline_queue_limit: int = 100
    pending_request_ttl_s: int = 7 * 24 * 3600


class HubServer:
    def __init__(self, state: HubState, cfg: HubServerConfig | None = None) -> None:
        self.state = state
        self.cfg = cfg or HubServerConfig()
        self.bound_port: int = self.cfg.port
        self._server: websockets.server.Serve | None = None
        self._limiter = RateLimiter(self.cfg.rate_limit_per_min, self.cfg.rate_limit_burst)

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self.cfg.host,
            self.cfg.port,
            ping_interval=20,
            ping_timeout=20,
        )
        # If port=0 was used, capture the actual bound port for tests/clients.
        try:
            if self._server.sockets:
                self.bound_port = int(self._server.sockets[0].getsockname()[1])
        except Exception:
            self.bound_port = self.cfg.port
        logger.info(f"Universe hub listening on ws://{self.cfg.host}:{self.bound_port}{self.cfg.path}")

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        org_id = None
        node_id = None
        try:
            hello_raw = await asyncio.wait_for(ws.recv(), timeout=self.cfg.hello_timeout_s)
            if self.cfg.max_message_bytes and len(hello_raw) > self.cfg.max_message_bytes:
                await ws.send(make_envelope("error", payload={"message": "message too large"}).to_json())
                return
            hello = Envelope.from_json(hello_raw)
            if not self._allow_client(ws):
                await ws.send(make_envelope("error", payload={"message": "rate limited"}).to_json())
                return
            if hello.type != "hello":
                await ws.send(make_envelope("error", payload={"message": "expected hello"}).to_json())
                return

            org_id = (hello.payload or {}).get("orgId") or hello.org_id
            join_secret = (hello.payload or {}).get("joinSecret", "")
            node_id = (hello.payload or {}).get("nodeId") or hello.from_node
            node_name = (hello.payload or {}).get("nodeName", "")
            capabilities = (hello.payload or {}).get("capabilities", {})

            if not org_id or not node_id:
                await ws.send(make_envelope("error", payload={"message": "missing orgId/nodeId"}).to_json())
                return

            if not self.state.validate_org_join(org_id, join_secret):
                await ws.send(make_envelope("error", payload={"message": "invalid join secret"}).to_json())
                return

            org = self.state.get_org(org_id)
            async with org.lock:
                org.nodes[node_id] = NodeConn(node_id=node_id, node_name=node_name, capabilities=capabilities, ws=ws)

            await ws.send(
                make_envelope(
                    "hello_ok",
                    org_id=org_id,
                    from_node="hub",
                    to_node=node_id,
                    payload={
                        "nodeId": node_id,
                        "orgId": org_id,
                    },
                ).to_json()
            )

            # Deliver offline queue (friend requests, dms, etc.)
            async with org.lock:
                queued = org.pop_offline(node_id)
            for msg in queued:
                await ws.send(msg.to_json())

            # Broadcast presence
            await self._broadcast(org_id, self.state.presence_event(org_id, NodeInfo(node_id, node_name, capabilities), True))

            async for raw in ws:
                if self.cfg.max_message_bytes and len(raw) > self.cfg.max_message_bytes:
                    await ws.send(make_envelope("error", payload={"message": "message too large"}).to_json())
                    continue
                try:
                    env = Envelope.from_json(raw)
                except Exception as e:
                    await ws.send(make_envelope("error", payload={"message": f"bad json: {e}"}).to_json())
                    continue
                if not self._allow_client(ws):
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "rate limited"}).to_json())
                    continue

                await self._handle_message(org_id, node_id, env, ws)
        except asyncio.TimeoutError:
            try:
                await ws.send(make_envelope("error", payload={"message": "hello timeout"}).to_json())
            except Exception:
                pass
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.exception(f"hub handler error: {e}")
        finally:
            if org_id and node_id:
                org = self.state.get_org(org_id)
                removed: NodeConn | None = None
                async with org.lock:
                    removed = org.nodes.pop(node_id, None)
                if removed:
                    await self._broadcast(
                        org_id,
                        self.state.presence_event(
                            org_id,
                            NodeInfo(removed.node_id, removed.node_name, removed.capabilities),
                            False,
                        ),
                    )

    async def _handle_message(self, org_id: str, node_id: str, env: Envelope, ws: WebSocketServerProtocol) -> None:
        org = self.state.get_org(org_id)
        msg_type = env.type
        async with org.lock:
            org.cleanup_pending(self.cfg.pending_request_ttl_s)

        if msg_type == "ping":
            await ws.send(make_envelope("pong", id=env.id).to_json())
            return

        if msg_type == "list_nodes":
            async with org.lock:
                nodes = [
                    {"nodeId": n.node_id, "nodeName": n.node_name, "capabilities": n.capabilities}
                    for n in org.nodes.values()
                ]
            await ws.send(make_envelope("list_nodes_ok", id=env.id, payload={"nodes": nodes}).to_json())
            return

        if msg_type == "get_friends":
            async with org.lock:
                friends = sorted(list(org.friends.get(node_id, set())))
            await ws.send(make_envelope("get_friends_ok", id=env.id, payload={"friends": friends}).to_json())
            return

        if msg_type == "get_pending_friend_requests":
            async with org.lock:
                pending = [
                    {
                        "requestId": fr.req_id,
                        "fromNode": fr.from_node,
                        "toNode": fr.to_node,
                        "message": fr.message,
                    }
                    for fr in org.pending_requests.values()
                    if fr.to_node == node_id
                ]
            await ws.send(make_envelope("get_pending_friend_requests_ok", id=env.id, payload={"requests": pending}).to_json())
            return

        if msg_type == "friend_request":
            to_node = (env.payload or {}).get("toNode") or env.to_node
            message = (env.payload or {}).get("message", "")
            if not to_node:
                await ws.send(make_envelope("error", id=env.id, payload={"message": "missing toNode"}).to_json())
                return
            if to_node == node_id:
                await ws.send(make_envelope("error", id=env.id, payload={"message": "cannot friend self"}).to_json())
                return

            async with org.lock:
                fr = org.new_friend_request(org_id, node_id, to_node, message)
                # Deliver to target or queue offline
                target = org.nodes.get(to_node)
                deliver = make_envelope(
                    "friend_request_deliver",
                    org_id=org_id,
                    from_node=node_id,
                    to_node=to_node,
                    payload={"requestId": fr.req_id, "fromNode": node_id, "message": message},
                )
                if target and target.ws:
                    await target.ws.send(deliver.to_json())
                else:
                    org.queue_offline_limited(to_node, deliver, self.cfg.offline_queue_limit)

            await ws.send(make_envelope("friend_request_ok", id=env.id, payload={"requestId": fr.req_id}).to_json())
            return

        if msg_type == "friend_accept":
            request_id = (env.payload or {}).get("requestId")
            if not request_id:
                await ws.send(make_envelope("error", id=env.id, payload={"message": "missing requestId"}).to_json())
                return

            async with org.lock:
                fr = org.pending_requests.get(request_id)
                if not fr:
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "unknown requestId"}).to_json())
                    return
                if fr.to_node != node_id:
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "not your request"}).to_json())
                    return
                org.pending_requests.pop(request_id, None)
                org.make_friends(fr.from_node, fr.to_node)

                notify_a = make_envelope(
                    "friend_added",
                    org_id=org_id,
                    from_node="hub",
                    to_node=fr.from_node,
                    payload={"friendNode": fr.to_node},
                )
                notify_b = make_envelope(
                    "friend_added",
                    org_id=org_id,
                    from_node="hub",
                    to_node=fr.to_node,
                    payload={"friendNode": fr.from_node},
                )

                # Deliver / queue notifications
                for target_id, msg in [(fr.from_node, notify_a), (fr.to_node, notify_b)]:
                    target = org.nodes.get(target_id)
                    if target and target.ws:
                        await target.ws.send(msg.to_json())
                    else:
                        org.queue_offline_limited(target_id, msg, self.cfg.offline_queue_limit)

            await ws.send(make_envelope("friend_accept_ok", id=env.id, payload={"requestId": request_id}).to_json())
            return

        if msg_type == "dm":
            to_node = (env.payload or {}).get("toNode") or env.to_node
            content = (env.payload or {}).get("content", "")
            if not to_node or not content:
                await ws.send(make_envelope("error", id=env.id, payload={"message": "missing toNode/content"}).to_json())
                return

            async with org.lock:
                if not org.are_friends(node_id, to_node):
                    await ws.send(make_envelope("error", id=env.id, payload={"message": "not friends"}).to_json())
                    return
                deliver = make_envelope(
                    "dm_deliver",
                    org_id=org_id,
                    from_node=node_id,
                    to_node=to_node,
                    payload={"fromNode": node_id, "content": content},
                )
                target = org.nodes.get(to_node)
                if target and target.ws:
                    await target.ws.send(deliver.to_json())
                else:
                    org.queue_offline_limited(to_node, deliver, self.cfg.offline_queue_limit)

            await ws.send(make_envelope("dm_ok", id=env.id).to_json())
            return

        await ws.send(make_envelope("error", id=env.id, payload={"message": f"unknown type: {msg_type}"}).to_json())

    async def _broadcast(self, org_id: str, msg: Envelope) -> None:
        org = self.state.get_org(org_id)
        async with org.lock:
            conns = [n.ws for n in org.nodes.values() if n.ws is not None]

        if not conns:
            return
        payload = msg.to_json()
        await asyncio.gather(*[self._safe_send(ws, payload) for ws in conns], return_exceptions=True)

    async def _safe_send(self, ws: WebSocketServerProtocol, payload: str) -> None:
        try:
            await ws.send(payload)
        except Exception:
            return

    def _allow_client(self, ws: WebSocketServerProtocol) -> bool:
        key = "unknown"
        try:
            if ws.remote_address and ws.remote_address[0]:
                key = str(ws.remote_address[0])
        except Exception:
            pass
        return self._limiter.allow(key)
