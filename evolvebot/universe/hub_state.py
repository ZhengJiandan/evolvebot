from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from evolvebot.universe.protocol import Envelope, make_envelope


@dataclass
class FriendRequest:
    req_id: str
    org_id: str
    from_node: str
    to_node: str
    message: str = ""
    created_ts: float = 0.0


@dataclass
class NodeInfo:
    node_id: str
    node_name: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeConn(NodeInfo):
    ws: Any = None  # websockets.server.WebSocketServerProtocol


class OrgState:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeConn] = {}
        self.friends: dict[str, set[str]] = {}
        self.pending_requests: dict[str, FriendRequest] = {}
        self.offline_queue: dict[str, list[Envelope]] = {}
        self.lock = asyncio.Lock()

    def _ensure_user(self, node_id: str) -> None:
        if node_id not in self.friends:
            self.friends[node_id] = set()

    def are_friends(self, a: str, b: str) -> bool:
        return b in self.friends.get(a, set())

    def make_friends(self, a: str, b: str) -> None:
        self._ensure_user(a)
        self._ensure_user(b)
        self.friends[a].add(b)
        self.friends[b].add(a)

    def queue_offline(self, node_id: str, msg: Envelope) -> None:
        self.queue_offline_limited(node_id, msg, limit=None)

    def queue_offline_limited(self, node_id: str, msg: Envelope, limit: int | None) -> None:
        queue = self.offline_queue.setdefault(node_id, [])
        if limit and limit > 0:
            while len(queue) >= limit:
                queue.pop(0)
        queue.append(msg)

    def pop_offline(self, node_id: str) -> list[Envelope]:
        return self.offline_queue.pop(node_id, [])

    def new_friend_request(self, org_id: str, from_node: str, to_node: str, message: str) -> FriendRequest:
        req_id = str(uuid4())
        fr = FriendRequest(
            req_id=req_id,
            org_id=org_id,
            from_node=from_node,
            to_node=to_node,
            message=message,
            created_ts=time.time(),
        )
        self.pending_requests[req_id] = fr
        return fr

    def cleanup_pending(self, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        cutoff = time.time() - float(ttl_s)
        for req_id, fr in list(self.pending_requests.items()):
            if fr.created_ts and fr.created_ts < cutoff:
                self.pending_requests.pop(req_id, None)


class HubState:
    def __init__(self, org_join_secrets: dict[str, str]) -> None:
        self.org_join_secrets = org_join_secrets
        self.orgs: dict[str, OrgState] = {org_id: OrgState() for org_id in org_join_secrets.keys()}

    def validate_org_join(self, org_id: str, join_secret: str) -> bool:
        expected = self.org_join_secrets.get(org_id, "")
        return bool(expected) and join_secret == expected

    def get_org(self, org_id: str) -> OrgState:
        if org_id not in self.orgs:
            # Allow dynamic orgs if caller configured them later; default is fixed.
            self.orgs[org_id] = OrgState()
        return self.orgs[org_id]

    @staticmethod
    def presence_event(org_id: str, node: NodeInfo, online: bool) -> Envelope:
        return make_envelope(
            "presence",
            org_id=org_id,
            from_node=node.node_id,
            payload={
                "online": online,
                "nodeId": node.node_id,
                "nodeName": node.node_name,
                "capabilities": node.capabilities,
            },
        )
