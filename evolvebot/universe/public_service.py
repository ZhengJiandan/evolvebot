from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import hashlib
import time
from uuid import uuid4
from pathlib import Path
from typing import Any

import httpx
import websockets

from loguru import logger

from evolvebot.config.schema import Config
from evolvebot.universe.node_server import NodeServer, NodeServerConfig
from evolvebot.universe.relay_client import RelayNodeClient, RelayNodeClientConfig
from evolvebot.universe.protocol import Envelope, make_envelope
from evolvebot.universe.public_client import knowledge_publish


def _load_knowledge_pack_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name", "")).strip()
    kind = str(data.get("kind", "")).strip()
    content = str(data.get("content", ""))
    if not name or not kind or not content:
        return None
    return data


def _compute_pack_id(name: str, kind: str, content: str) -> str:
    raw = f"{name}\n{kind}\n{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _publish_state_path(base: Path) -> Path:
    return base / ".publish_state.json"


def _load_publish_state(base: Path) -> dict[str, Any]:
    path = _publish_state_path(base)
    if not path.exists():
        return {"failures": {}}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data.setdefault("failures", {})
            return data
    except Exception:
        pass
    return {"failures": {}}


def _save_publish_state(base: Path, state: dict[str, Any]) -> None:
    path = _publish_state_path(base)
    try:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception:
        return


def _move_pack_file(src: Path, dest_dir: Path) -> Path | None:
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / src.name
        if target.exists():
            target = dest_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
        src.replace(target)
        return target
    except Exception:
        return None


async def _knowledge_publish_loop(cfg: Config, *, log_prefix: str = "universe") -> None:
    uc = cfg.universe
    if not uc.public_knowledge_publish_dir:
        return

    base = Path(uc.public_knowledge_publish_dir).expanduser()
    interval = max(10, int(uc.public_knowledge_publish_interval_s or 300))
    max_retries = max(1, int(getattr(uc, "knowledge_review_publish_max_retries", 3) or 3))

    while True:
        try:
            if base.exists():
                state = _load_publish_state(base)
                failures: dict[str, Any] = state.get("failures", {}) or {}
                state_path = _publish_state_path(base)
                for path in sorted(base.glob("*.json")):
                    if path == state_path:
                        continue
                    data = _load_knowledge_pack_file(path)
                    if not data:
                        info = failures.get(path.name, {"count": 0})
                        info["count"] = int(info.get("count", 0) or 0) + 1
                        info["last_error"] = "invalid pack file"
                        info["updated_ts"] = time.time()
                        if info["count"] >= max_retries:
                            failed_dir = base / "failed"
                            moved = _move_pack_file(path, failed_dir)
                            if moved:
                                (moved.parent / f"{moved.name}.error.txt").write_text(info["last_error"])
                            failures.pop(path.name, None)
                            logger.warning(f"{log_prefix}: moved invalid pack to failed: {path.name}")
                        else:
                            failures[path.name] = info
                        continue
                    name = str(data.get("name", "")).strip()
                    kind = str(data.get("kind", "")).strip()
                    content = str(data.get("content", ""))
                    pack_id = str(data.get("id") or "").strip() or _compute_pack_id(name, kind, content)
                    try:
                        await knowledge_publish(
                            registry_url=uc.public_registry_url,
                            registry_token=uc.public_registry_token,
                            name=name,
                            kind=kind,
                            content=content,
                            summary=str(data.get("summary", "")).strip(),
                            tags=data.get("tags", []) or [],
                            version=str(data.get("version", "1.0")).strip() or "1.0",
                            pack_id=pack_id,
                            owner_node=uc.node_id,
                            allow_update=True,
                        )
                        failures.pop(path.name, None)
                        published_dir = base / "published"
                        moved = _move_pack_file(path, published_dir)
                        if moved:
                            logger.info(f"{log_prefix}: published knowledge pack {pack_id}")
                        else:
                            logger.warning(f"{log_prefix}: publish succeeded but move failed for {path.name}")
                    except Exception as e:
                        info = failures.get(path.name, {"count": 0})
                        info["count"] = int(info.get("count", 0) or 0) + 1
                        info["last_error"] = str(e)
                        info["updated_ts"] = time.time()
                        if info["count"] >= max_retries:
                            failed_dir = base / "failed"
                            moved = _move_pack_file(path, failed_dir)
                            if moved:
                                (moved.parent / f"{moved.name}.error.txt").write_text(info["last_error"])
                            failures.pop(path.name, None)
                            logger.warning(f"{log_prefix}: moved failed pack to failed: {path.name}")
                        else:
                            failures[path.name] = info
                state["failures"] = failures
                _save_publish_state(base, state)
        except Exception as e:
            logger.warning(f"{log_prefix}: knowledge auto publish failed: {e}")
        await asyncio.sleep(interval)


@dataclass
class PublicServiceHandle:
    server: NodeServer | None
    register_task: asyncio.Task[None] | None
    relay_task: asyncio.Task[None] | None = None
    publish_task: asyncio.Task[None] | None = None


def build_public_capabilities(cfg: Config) -> dict[str, Any]:
    caps = dict(cfg.universe.public_capabilities or {"llm.chat": True})
    if cfg.universe.knowledge_review_publish:
        caps["knowledge.pack"] = True
    if cfg.universe.public_allow_agent_tasks:
        caps["evolvebot.agent"] = True
        allow = set(cfg.universe.public_agent_tool_allowlist or [])
        if "web_search" in allow:
            caps["web_search"] = True
        if "web_fetch" in allow:
            caps["web_fetch"] = True
    else:
        caps.pop("evolvebot.agent", None)
    return caps


def build_capability_card(cfg: Config, endpoint_url: str) -> dict[str, Any]:
    caps = dict(cfg.universe.public_capabilities or {})
    card = {
        "schemaVersion": "1.0",
        "nodeId": cfg.universe.node_id,
        "nodeName": cfg.universe.node_name or "",
        "capabilities": caps,
        "tools": list(cfg.universe.public_tools or []),
        "models": list(cfg.universe.public_models or []),
        "languages": list(cfg.universe.public_languages or []),
        "pricing": {"unit": "point", "perRequest": int(cfg.universe.public_price_points or 1)},
        "limits": {
            "maxTokens": int(cfg.universe.public_max_tokens or 1024),
            "rateLimitPerMin": int(cfg.universe.public_rate_limit_per_min or 60),
            "rateLimitPerMinByNode": int(cfg.universe.public_rate_limit_per_min_by_node or 60),
        },
    }
    if cfg.universe.public_allow_agent_tasks:
        caps["evolvebot.agent"] = True
        allow = set(cfg.universe.public_agent_tool_allowlist or [])
        if "web_search" in allow:
            caps["web_search"] = True
        if "web_fetch" in allow:
            caps["web_fetch"] = True
    if cfg.universe.knowledge_review_publish:
        caps["knowledge.pack"] = True
    if not card.get("summary"):
        card["summary"] = cfg.universe.node_name or "evolvebot node"
    if not card.get("skills"):
        card["skills"] = list(caps.keys()) if caps else []
    if cfg.universe.public_capability_card:
        card.update(cfg.universe.public_capability_card)
    return card


async def _detect_public_ip(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            ip = (r.text or "").strip()
            return ip or None
    except Exception:
        return None


async def resolve_advertise_url(cfg: Config, server_port: int) -> str:
    uc = cfg.universe
    if uc.public_advertise_url:
        return uc.public_advertise_url

    host = uc.public_advertise_host
    port = uc.public_advertise_port or server_port
    if host:
        return f"ws://{host}:{port}"

    if uc.public_detect_public_ip:
        ip = await _detect_public_ip(uc.public_detect_ip_service)
        if ip:
            return f"ws://{ip}:{port}"

    if uc.public_service_host and uc.public_service_host != "0.0.0.0":
        return f"ws://{uc.public_service_host}:{port}"

    # Fallback: localhost (only useful for same-machine testing).
    return f"ws://127.0.0.1:{port}"


async def _self_check(endpoint_url: str, timeout_s: float) -> bool:
    try:
        async with websockets.connect(endpoint_url, open_timeout=timeout_s, close_timeout=timeout_s) as ws:
            ping = make_envelope("ping")
            await ws.send(ping.to_json())
            resp = Envelope.from_json(await ws.recv())
            return resp.type == "pong"
    except Exception:
        return False


async def _register_loop(
    *,
    registry_url: str,
    registry_token: str,
    node_id: str,
    node_name: str,
    endpoint_url: str,
    capabilities: dict[str, Any],
    price_points: int,
    capability_card: dict[str, Any] | None = None,
    log_prefix: str = "universe",
) -> None:
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(registry_url) as ws:
                env = make_envelope(
                    "register",
                    from_node=node_id,
                    payload={
                        "nodeId": node_id,
                        "nodeName": node_name,
                        "endpointUrl": endpoint_url,
                        "capabilities": capabilities,
                        "capabilityCard": capability_card or {},
                        "pricePoints": price_points,
                        "registryToken": registry_token,
                    },
                )
                await ws.send(env.to_json())
                resp = Envelope.from_json(await ws.recv())
                if resp.type != "register_ok":
                    raise RuntimeError((resp.payload or {}).get("message", "register failed"))

                logger.info(f"{log_prefix}: registered in registry {registry_url} as {node_id}")
                backoff = 1.0
                while True:
                    await asyncio.sleep(30)
                    upd = make_envelope(
                        "update",
                        from_node=node_id,
                        payload={
                            "nodeId": node_id,
                            "nodeName": node_name,
                            "endpointUrl": endpoint_url,
                            "capabilities": capabilities,
                            "capabilityCard": capability_card or {},
                            "pricePoints": price_points,
                            "registryToken": registry_token,
                        },
                    )
                    await ws.send(upd.to_json())
                    resp = Envelope.from_json(await ws.recv())
                    if resp.type != "update_ok":
                        raise RuntimeError((resp.payload or {}).get("message", "update failed"))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"{log_prefix}: registry connection failed ({e}); retrying in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def start_public_service(cfg: Config, *, log_prefix: str = "universe") -> PublicServiceHandle:
    uc = cfg.universe
    if not uc.node_id:
        uc.node_id = str(uuid4())
        logger.warning(f"{log_prefix}: node_id missing; generated a new one (persist it in config for stability)")
    relay_only = bool(uc.public_relay_only)
    if relay_only and not uc.public_relay_url:
        logger.warning(f"{log_prefix}: public_relay_only is set but public_relay_url is empty; falling back to direct mode")
        relay_only = False

    server: NodeServer | None = None
    endpoint_url = ""
    if not relay_only:
        server = NodeServer(
            NodeServerConfig(
                host=uc.public_service_host,
                port=uc.public_service_port,
                service_token=uc.public_service_token,
                rate_limit_per_min=uc.public_rate_limit_per_min,
                rate_limit_burst=uc.public_rate_limit_burst,
                rate_limit_per_min_by_node=uc.public_rate_limit_per_min_by_node,
                rate_limit_burst_by_node=uc.public_rate_limit_burst_by_node,
            )
        )
        await server.start()

        endpoint_url = await resolve_advertise_url(cfg, server.bound_port)
        if endpoint_url.startswith("ws://127.0.0.1"):
            logger.warning(f"{log_prefix}: advertise_url is localhost; other machines cannot reach this node")

    caps = build_public_capabilities(cfg)
    card = build_capability_card(cfg, endpoint_url)
    node_id = uc.node_id
    node_name = uc.node_name or ""

    if endpoint_url and uc.public_self_check_enabled:
        ok = await _self_check(endpoint_url, float(uc.public_self_check_timeout_s or 3.0))
        if not ok:
            logger.warning(f"{log_prefix}: self-check failed for {endpoint_url} (NAT or firewall may block)")
    reg_task = asyncio.create_task(
        _register_loop(
            registry_url=uc.public_registry_url,
            registry_token=uc.public_registry_token,
            node_id=node_id,
            node_name=node_name,
            endpoint_url=endpoint_url,
            capabilities=caps,
            price_points=int(uc.public_price_points or 1),
            capability_card=card,
            log_prefix=log_prefix,
        )
    )

    relay_task: asyncio.Task[None] | None = None
    if uc.public_relay_url:
        relay = RelayNodeClient(
            RelayNodeClientConfig(
                relay_url=uc.public_relay_url,
                node_id=node_id,
                relay_token=uc.public_relay_token or "",
                service_token=uc.public_service_token or "",
                rate_limit_per_min=uc.public_rate_limit_per_min,
                rate_limit_burst=uc.public_rate_limit_burst,
                rate_limit_per_min_by_node=uc.public_rate_limit_per_min_by_node,
                rate_limit_burst_by_node=uc.public_rate_limit_burst_by_node,
            )
        )
        relay_task = asyncio.create_task(relay.run_forever())
        logger.info(f"{log_prefix}: relay client started ({uc.public_relay_url})")

    return PublicServiceHandle(server=server, register_task=reg_task, relay_task=relay_task)


async def stop_public_service(handle: PublicServiceHandle | None) -> None:
    if not handle:
        return
    if handle.register_task:
        handle.register_task.cancel()
        try:
            await handle.register_task
        except Exception:
            pass
    if handle.relay_task:
        handle.relay_task.cancel()
        try:
            await handle.relay_task
        except Exception:
            pass
    if handle.publish_task:
        handle.publish_task.cancel()
        try:
            await handle.publish_task
        except Exception:
            pass
    if handle.server:
        await handle.server.stop()


async def maybe_start_public_service(cfg: Config, *, log_prefix: str = "universe") -> PublicServiceHandle | None:
    uc = cfg.universe
    if not uc.public_enabled:
        return None
    publish_task: asyncio.Task[None] | None = None
    if uc.knowledge_review_publish and uc.public_knowledge_publish_dir:
        publish_task = asyncio.create_task(_knowledge_publish_loop(cfg, log_prefix=log_prefix))

    if not uc.public_provide_service or not uc.public_auto_register:
        if publish_task:
            return PublicServiceHandle(server=None, register_task=None, publish_task=publish_task)
        return None

    handle = await start_public_service(cfg, log_prefix=log_prefix)
    handle.publish_task = publish_task
    return handle
