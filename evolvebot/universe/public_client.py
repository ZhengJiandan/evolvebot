from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import websockets

from evolvebot.universe.protocol import Envelope, make_envelope


@dataclass
class PublicNode:
    node_id: str
    node_name: str
    endpoint_url: str
    capabilities: dict[str, Any]
    price_points: int
    success_count: int = 0
    fail_count: int = 0
    avg_latency_ms: int = 0


@dataclass
class KnowledgePackMeta:
    pack_id: str
    name: str
    kind: str
    summary: str
    tags: list[str]
    version: str
    owner_node: str
    status: str
    score: float
    quality_score: int
    uses: int
    upvotes: int
    downvotes: int
    created_ts: float
    updated_ts: float
    content_hash: str
    size_bytes: int


@dataclass
class KnowledgePack(KnowledgePackMeta):
    content: str


async def list_public_nodes(
    *,
    registry_url: str,
    registry_token: str | None = None,
    require_capabilities: list[str] | None = None,
    online_only: bool = True,
    timeout_s: float = 10.0,
) -> list[PublicNode]:
    require_capabilities = require_capabilities or []
    async with websockets.connect(registry_url) as ws:
        req = make_envelope(
            "list",
            payload={
                "onlineOnly": bool(online_only),
                "requireCapabilities": require_capabilities,
                "registryToken": registry_token or "",
            },
        )
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "registry error"))
            if env.type != "list_result":
                continue
            nodes = []
            for n in (env.payload or {}).get("nodes", []):
                nodes.append(
                    PublicNode(
                        node_id=n.get("nodeId", ""),
                        node_name=n.get("nodeName", ""),
                        endpoint_url=n.get("endpointUrl", ""),
                        capabilities=n.get("capabilities", {}) or {},
                        price_points=int(n.get("pricePoints", 1) or 1),
                        success_count=int(n.get("successCount", 0) or 0),
                        fail_count=int(n.get("failCount", 0) or 0),
                        avg_latency_ms=int(n.get("avgLatencyMs", 0) or 0),
                    )
                )
            return nodes


async def knowledge_publish(
    *,
    registry_url: str,
    registry_token: str,
    name: str,
    kind: str,
    content: str,
    summary: str = "",
    tags: list[str] | None = None,
    version: str = "1.0",
    pack_id: str | None = None,
    owner_node: str | None = None,
    allow_update: bool = False,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "kind": kind,
        "content": content,
        "summary": summary,
        "tags": tags or [],
        "version": version,
        "registryToken": registry_token,
        "allowUpdate": bool(allow_update),
    }
    if pack_id:
        payload["id"] = pack_id
    if owner_node:
        payload["ownerNode"] = owner_node
    async with websockets.connect(registry_url) as ws:
        req = make_envelope("knowledge_publish", payload=payload)
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "knowledge_publish_ok":
                return env.payload or {}
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "publish failed"))


async def knowledge_list(
    *,
    registry_url: str,
    registry_token: str | None = None,
    kind: str | None = None,
    tag: str | None = None,
    owner_node: str | None = None,
    limit: int = 50,
    include_pending: bool = False,
) -> list[KnowledgePackMeta]:
    payload: dict[str, Any] = {"limit": limit}
    if kind:
        payload["kind"] = kind
    if tag:
        payload["tag"] = tag
    if owner_node:
        payload["ownerNode"] = owner_node
    if registry_token:
        payload["registryToken"] = registry_token
    if include_pending:
        payload["includePending"] = True
    async with websockets.connect(registry_url) as ws:
        req = make_envelope("knowledge_list", payload=payload)
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "knowledge_list_result":
                packs = []
                for p in (env.payload or {}).get("packs", []):
                    packs.append(
                        KnowledgePackMeta(
                            pack_id=p.get("id", ""),
                            name=p.get("name", ""),
                            kind=p.get("kind", ""),
                            summary=p.get("summary", ""),
                            tags=p.get("tags", []) or [],
                            version=p.get("version", "1.0"),
                            owner_node=p.get("ownerNode", ""),
                            status=p.get("status", "pending"),
                            score=float(p.get("score", 0) or 0),
                            quality_score=int(p.get("qualityScore", 0) or 0),
                            uses=int(p.get("uses", 0) or 0),
                            upvotes=int(p.get("upvotes", 0) or 0),
                            downvotes=int(p.get("downvotes", 0) or 0),
                            created_ts=float(p.get("createdTs", 0) or 0),
                            updated_ts=float(p.get("updatedTs", 0) or 0),
                            content_hash=p.get("contentHash", ""),
                            size_bytes=int(p.get("sizeBytes", 0) or 0),
                        )
                    )
                return packs
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "list failed"))


async def knowledge_get(
    *,
    registry_url: str,
    pack_id: str,
    registry_token: str | None = None,
) -> KnowledgePack:
    payload: dict[str, Any] = {"id": pack_id}
    if registry_token:
        payload["registryToken"] = registry_token
    async with websockets.connect(registry_url) as ws:
        req = make_envelope("knowledge_get", payload=payload)
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "knowledge_get_result":
                p = env.payload or {}
                return KnowledgePack(
                    pack_id=p.get("id", ""),
                    name=p.get("name", ""),
                    kind=p.get("kind", ""),
                    summary=p.get("summary", ""),
                    tags=p.get("tags", []) or [],
                    version=p.get("version", "1.0"),
                    owner_node=p.get("ownerNode", ""),
                    status=p.get("status", "pending"),
                    score=float(p.get("score", 0) or 0),
                    quality_score=int(p.get("qualityScore", 0) or 0),
                    uses=int(p.get("uses", 0) or 0),
                    upvotes=int(p.get("upvotes", 0) or 0),
                    downvotes=int(p.get("downvotes", 0) or 0),
                    created_ts=float(p.get("createdTs", 0) or 0),
                    updated_ts=float(p.get("updatedTs", 0) or 0),
                    content_hash=p.get("contentHash", ""),
                    size_bytes=int(p.get("sizeBytes", 0) or 0),
                    content=p.get("content", ""),
                )
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "get failed"))


async def knowledge_rate(
    *,
    registry_url: str,
    pack_id: str,
    delta: int,
    registry_token: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": pack_id, "delta": int(delta or 0)}
    if registry_token:
        payload["registryToken"] = registry_token
    async with websockets.connect(registry_url) as ws:
        req = make_envelope("knowledge_rate", payload=payload)
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "knowledge_rate_ok":
                return env.payload or {}
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "rate failed"))


async def resolve_endpoint(
    *,
    registry_url: str,
    node_id: str,
    registry_token: str,
) -> str:
    if not registry_token:
        raise RuntimeError("registry token required to resolve endpoint")
    async with websockets.connect(registry_url) as ws:
        req = make_envelope("resolve", payload={"nodeId": node_id, "registryToken": registry_token})
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "resolve error"))
            if env.type == "resolve_ok":
                endpoint = (env.payload or {}).get("endpointUrl", "")
                if not endpoint:
                    raise RuntimeError("endpoint not available")
                return endpoint


def _score_node(node: PublicNode) -> float:
    total = node.success_count + node.fail_count
    # Laplace smoothing to avoid zero-division and cold-start bias.
    success_rate = (node.success_count + 1) / (total + 2)
    avg_latency = node.avg_latency_ms or 1000
    price = max(1, int(node.price_points or 1))
    # Higher is better.
    return (success_rate * 100.0) - (avg_latency / 1000.0 * 10.0) - (price * 2.0)


def pick_node(nodes: list[PublicNode], *, max_price_points: int | None = None) -> PublicNode:
    candidates = nodes
    if max_price_points is not None:
        candidates = [n for n in nodes if n.price_points <= max_price_points]
    if not candidates:
        raise RuntimeError("no eligible nodes found")
    scored = sorted(candidates, key=_score_node, reverse=True)
    top_score = _score_node(scored[0])
    top = [n for n in scored if _score_node(n) >= top_score - 0.5]
    return random.choice(top)


async def call_node(
    *,
    endpoint_url: str,
    kind: str,
    prompt: str,
    service_token: str = "",
    client_id: str | None = None,
    timeout_s: float = 120.0,
) -> str:
    async with websockets.connect(endpoint_url) as ws:
        req = make_envelope(
            "task_run",
            payload={
                "kind": kind,
                "prompt": prompt,
                "serviceToken": service_token,
                "clientId": client_id,
            },
        )
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "task_result":
                return (env.payload or {}).get("content", "")
            if env.type in {"task_error", "error"}:
                raise RuntimeError((env.payload or {}).get("message", "task failed"))


async def call_via_relay(
    *,
    relay_url: str,
    node_id: str,
    kind: str,
    prompt: str,
    service_token: str = "",
    client_id: str | None = None,
    relay_token: str = "",
    timeout_s: float = 120.0,
) -> str:
    async with websockets.connect(relay_url) as ws:
        req = make_envelope(
            "relay_request",
            payload={
                "nodeId": node_id,
                "kind": kind,
                "prompt": prompt,
                "serviceToken": service_token,
                "clientId": client_id,
                "relayToken": relay_token,
            },
        )
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "relay_response":
                ok = (env.payload or {}).get("ok")
                if ok:
                    return (env.payload or {}).get("content", "")
                raise RuntimeError((env.payload or {}).get("message", "relay task failed"))
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "relay error"))


async def reserve_points(
    *,
    registry_url: str,
    registry_token: str,
    payer_node_id: str,
    provider_node_id: str,
    points: int,
) -> str:
    async with websockets.connect(registry_url) as ws:
        req = make_envelope(
            "reserve",
            payload={
                "nodeId": provider_node_id,
                "payerNode": payer_node_id,
                "points": int(points),
                "registryToken": registry_token,
            },
        )
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "reserve_ok":
                return (env.payload or {}).get("reservationId", "")
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "reserve failed"))


async def commit_reservation(
    *,
    registry_url: str,
    registry_token: str,
    reservation_id: str,
) -> None:
    async with websockets.connect(registry_url) as ws:
        req = make_envelope(
            "commit",
            payload={"reservationId": reservation_id, "registryToken": registry_token},
        )
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "commit_ok":
                return
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "commit failed"))


async def cancel_reservation(
    *,
    registry_url: str,
    registry_token: str,
    reservation_id: str,
) -> None:
    async with websockets.connect(registry_url) as ws:
        req = make_envelope(
            "cancel",
            payload={"reservationId": reservation_id, "registryToken": registry_token},
        )
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "cancel_ok":
                return
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "cancel failed"))


async def report_task(
    *,
    registry_url: str,
    registry_token: str,
    node_id: str,
    ok: bool,
    latency_ms: int,
) -> None:
    async with websockets.connect(registry_url) as ws:
        req = make_envelope(
            "report",
            payload={
                "nodeId": node_id,
                "ok": bool(ok),
                "latencyMs": int(latency_ms),
                "registryToken": registry_token,
            },
        )
        await ws.send(req.to_json())
        while True:
            env = Envelope.from_json(await ws.recv())
            if env.id != req.id:
                continue
            if env.type == "report_ok":
                return
            if env.type == "error":
                raise RuntimeError((env.payload or {}).get("message", "report failed"))

async def delegate_task(
    *,
    registry_url: str,
    kind: str,
    prompt: str,
    require_capability: str | None = None,
    to_node_id: str | None = None,
    service_token: str = "",
    max_price_points: int | None = None,
    client_id: str | None = None,
    registry_token: str | None = None,
    relay_url: str | None = None,
    relay_token: str = "",
    relay_only: bool = False,
    preauth_enabled: bool = True,
    preauth_required: bool = False,
) -> tuple[PublicNode, str]:
    # Avoid delegating to self by default.
    if client_id:
        client_id = str(client_id)
    cap = require_capability or kind
    nodes = await list_public_nodes(
        registry_url=registry_url,
        registry_token=registry_token,
        require_capabilities=[cap] if cap else [],
    )
    if client_id:
        nodes = [n for n in nodes if n.node_id != client_id]
    if not nodes:
        raise RuntimeError("no eligible nodes found")
    if to_node_id:
        for n in nodes:
            if n.node_id == to_node_id:
                reservation_id = ""
                start = time.monotonic()
                if preauth_enabled:
                    if not registry_token or not client_id:
                        if preauth_required:
                            raise RuntimeError("preauth requires registry_token and client_id")
                    else:
                        reservation_id = await reserve_points(
                            registry_url=registry_url,
                            registry_token=registry_token,
                            payer_node_id=str(client_id),
                            provider_node_id=n.node_id,
                            points=int(n.price_points or 1),
                        )
                if relay_url:
                    try:
                        out = await call_via_relay(
                            relay_url=relay_url,
                            node_id=n.node_id,
                            kind=kind,
                            prompt=prompt,
                            service_token=service_token,
                            client_id=client_id,
                            relay_token=relay_token,
                        )
                        if reservation_id:
                            await commit_reservation(
                                registry_url=registry_url,
                                registry_token=registry_token or "",
                                reservation_id=reservation_id,
                            )
                        if registry_token:
                            try:
                                await report_task(
                                    registry_url=registry_url,
                                    registry_token=registry_token,
                                    node_id=n.node_id,
                                    ok=True,
                                    latency_ms=int((time.monotonic() - start) * 1000),
                                )
                            except Exception:
                                pass
                        return n, out
                    except Exception:
                        if registry_token:
                            try:
                                await report_task(
                                    registry_url=registry_url,
                                    registry_token=registry_token,
                                    node_id=n.node_id,
                                    ok=False,
                                    latency_ms=int((time.monotonic() - start) * 1000),
                                )
                            except Exception:
                                pass
                        if reservation_id:
                            try:
                                await cancel_reservation(
                                    registry_url=registry_url,
                                    registry_token=registry_token or "",
                                    reservation_id=reservation_id,
                                )
                            except Exception:
                                pass
                        if relay_only:
                            raise
                if relay_only:
                    raise RuntimeError("relay_only enabled but relay_url is not configured")
                endpoint = n.endpoint_url or await resolve_endpoint(
                    registry_url=registry_url,
                    node_id=n.node_id,
                    registry_token=registry_token or "",
                )
                try:
                    out = await call_node(
                        endpoint_url=endpoint,
                        kind=kind,
                        prompt=prompt,
                        service_token=service_token,
                        client_id=client_id,
                    )
                    if reservation_id:
                        await commit_reservation(
                            registry_url=registry_url,
                            registry_token=registry_token or "",
                            reservation_id=reservation_id,
                        )
                    if registry_token:
                        try:
                            await report_task(
                                registry_url=registry_url,
                                registry_token=registry_token,
                                node_id=n.node_id,
                                ok=True,
                                latency_ms=int((time.monotonic() - start) * 1000),
                            )
                        except Exception:
                            pass
                    return n, out
                except Exception:
                    if registry_token:
                        try:
                            await report_task(
                                registry_url=registry_url,
                                registry_token=registry_token,
                                node_id=n.node_id,
                                ok=False,
                                latency_ms=int((time.monotonic() - start) * 1000),
                            )
                        except Exception:
                            pass
                    if reservation_id:
                        try:
                            await cancel_reservation(
                                registry_url=registry_url,
                                registry_token=registry_token or "",
                                reservation_id=reservation_id,
                            )
                        except Exception:
                            pass
                    raise
        raise RuntimeError(f"node not found/online: {to_node_id}")

    node = pick_node(nodes, max_price_points=max_price_points)
    reservation_id = ""
    start = time.monotonic()
    if preauth_enabled:
        if not registry_token or not client_id:
            if preauth_required:
                raise RuntimeError("preauth requires registry_token and client_id")
        else:
            reservation_id = await reserve_points(
                registry_url=registry_url,
                registry_token=registry_token,
                payer_node_id=str(client_id),
                provider_node_id=node.node_id,
                points=int(node.price_points or 1),
            )
    if relay_url:
        try:
            out = await call_via_relay(
                relay_url=relay_url,
                node_id=node.node_id,
                kind=kind,
                prompt=prompt,
                service_token=service_token,
                client_id=client_id,
                relay_token=relay_token,
            )
            if reservation_id:
                await commit_reservation(
                    registry_url=registry_url,
                    registry_token=registry_token or "",
                    reservation_id=reservation_id,
                )
            if registry_token:
                try:
                    await report_task(
                        registry_url=registry_url,
                        registry_token=registry_token,
                        node_id=node.node_id,
                        ok=True,
                        latency_ms=int((time.monotonic() - start) * 1000),
                    )
                except Exception:
                    pass
            return node, out
        except Exception:
            if registry_token:
                try:
                    await report_task(
                        registry_url=registry_url,
                        registry_token=registry_token,
                        node_id=node.node_id,
                        ok=False,
                        latency_ms=int((time.monotonic() - start) * 1000),
                    )
                except Exception:
                    pass
            if reservation_id:
                try:
                    await cancel_reservation(
                        registry_url=registry_url,
                        registry_token=registry_token or "",
                        reservation_id=reservation_id,
                    )
                except Exception:
                    pass
            if relay_only:
                raise
    if relay_only:
        raise RuntimeError("relay_only enabled but relay_url is not configured")
    endpoint = node.endpoint_url or await resolve_endpoint(
        registry_url=registry_url,
        node_id=node.node_id,
        registry_token=registry_token or "",
    )
    try:
        out = await call_node(
            endpoint_url=endpoint,
            kind=kind,
            prompt=prompt,
            service_token=service_token,
            client_id=client_id,
        )
        if reservation_id:
            await commit_reservation(
                registry_url=registry_url,
                registry_token=registry_token or "",
                reservation_id=reservation_id,
            )
        if registry_token:
            try:
                await report_task(
                    registry_url=registry_url,
                    registry_token=registry_token,
                    node_id=node.node_id,
                    ok=True,
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception:
                pass
        return node, out
    except Exception:
        if registry_token:
            try:
                await report_task(
                    registry_url=registry_url,
                    registry_token=registry_token,
                    node_id=node.node_id,
                    ok=False,
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception:
                pass
        if reservation_id:
            try:
                await cancel_reservation(
                    registry_url=registry_url,
                    registry_token=registry_token or "",
                    reservation_id=reservation_id,
                )
            except Exception:
                pass
        raise
