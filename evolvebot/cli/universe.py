"""Universe CLI commands (client-side).

These commands manage org memberships, friends, and basic messaging.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from evolvebot.config.loader import load_config, save_config
from evolvebot.config.schema import UniverseMembership
from evolvebot.universe.client import UniverseClient

app = typer.Typer(help="Universe: org-scoped evolvebot network")
console = Console()


def _ensure_node_id() -> str:
    cfg = load_config()
    if not cfg.universe.node_id:
        cfg.universe.node_id = str(uuid4())
        cfg.universe.enabled = True
        save_config(cfg)
    return cfg.universe.node_id


def _get_membership(org_id: str | None) -> UniverseMembership:
    cfg = load_config()
    memberships = [m for m in cfg.universe.memberships if m.enabled]
    if not memberships:
        raise typer.BadParameter("No universe memberships. Run: evolvebot universe join ...")

    if org_id:
        for m in memberships:
            if m.org_id == org_id:
                return m
        raise typer.BadParameter(f"Unknown org_id: {org_id}")

    if len(memberships) == 1:
        return memberships[0]

    raise typer.BadParameter("Multiple memberships found; specify --org ORG_ID")


async def _client_from_config(org_id: str | None) -> UniverseClient:
    cfg = load_config()
    node_id = _ensure_node_id()
    m = _get_membership(org_id)
    node_name = cfg.universe.node_name or ""
    return UniverseClient(
        hub_url=m.hub_url,
        org_id=m.org_id,
        join_secret=m.join_secret,
        node_id=node_id,
        node_name=node_name,
        capabilities={"dm": True, "friend": True},
    )


@app.command("join")
def join(
    hub: str = typer.Option(..., "--hub", help="Hub WebSocket URL, e.g. ws://host:18888"),
    org: str = typer.Option(..., "--org", help="Org ID"),
    join_secret: str = typer.Option(..., "--join-secret", help="Org join secret/invite token"),
    label: str = typer.Option("", "--label", help="Optional label for this membership"),
    test: bool = typer.Option(True, "--test/--no-test", help="Test connection after saving"),
):
    """Join an org-scoped Universe hub."""
    cfg = load_config()
    _ensure_node_id()
    cfg.universe.enabled = True

    # Upsert membership
    existing = None
    for m in cfg.universe.memberships:
        if m.org_id == org:
            existing = m
            break
    if existing is None:
        cfg.universe.memberships.append(
            UniverseMembership(org_id=org, hub_url=hub, join_secret=join_secret, label=label, enabled=True)
        )
    else:
        existing.hub_url = hub
        existing.join_secret = join_secret
        existing.label = label or existing.label
        existing.enabled = True

    save_config(cfg)
    console.print(f"[green]✓[/green] Joined org [cyan]{org}[/cyan] at {hub}")

    if test:
        async def _test():
            client = await _client_from_config(org)
            async with client:
                nodes = await client.list_nodes()
                return nodes

        try:
            nodes = asyncio.run(_test())
            console.print(f"[green]✓[/green] Connected. Online nodes: {len(nodes)}")
        except Exception as e:
            console.print(f"[yellow]![/yellow] Saved, but connection test failed: {e}")


@app.command("status")
def status():
    """Show local Universe config + joined networks."""
    cfg = load_config()
    node_id = cfg.universe.node_id or "[dim]not set (run join)[/dim]"
    console.print(f"Node ID: [cyan]{node_id}[/cyan]")
    if cfg.universe.node_name:
        console.print(f"Node Name: {cfg.universe.node_name}")

    table = Table(title="Memberships")
    table.add_column("Org", style="cyan")
    table.add_column("Enabled")
    table.add_column("Hub URL")
    table.add_column("Label")
    for m in cfg.universe.memberships:
        table.add_row(m.org_id or "-", "✓" if m.enabled else "✗", m.hub_url or "-", m.label or "-")
    console.print(table)


@app.command("nodes")
def nodes(org: str = typer.Option(None, "--org", help="Org ID (if multiple memberships)")):
    """List online nodes in the org."""
    async def _run():
        client = await _client_from_config(org)
        async with client:
            return await client.list_nodes()

    nodes = asyncio.run(_run())
    table = Table(title="Online Nodes")
    table.add_column("Node ID", style="cyan")
    table.add_column("Name")
    table.add_column("Capabilities")
    for n in nodes:
        table.add_row(n.get("nodeId", ""), n.get("nodeName", ""), str(n.get("capabilities", {})))
    console.print(table)


friend_app = typer.Typer(help="Manage friends (explicit mutual add)")
app.add_typer(friend_app, name="friend")


@friend_app.command("request")
def friend_request(
    to: str = typer.Option(..., "--to", help="Target node_id"),
    message: str = typer.Option("", "--message", "-m", help="Optional message"),
    org: str = typer.Option(None, "--org", help="Org ID (if multiple memberships)"),
):
    """Send a friend request to another node."""
    async def _run():
        client = await _client_from_config(org)
        async with client:
            return await client.send_friend_request(to, message=message)

    req_id = asyncio.run(_run())
    console.print(f"[green]✓[/green] Friend request sent. requestId={req_id}")


@friend_app.command("pending")
def friend_pending(org: str = typer.Option(None, "--org", help="Org ID (if multiple memberships)")):
    """List pending friend requests addressed to you."""
    async def _run():
        client = await _client_from_config(org)
        async with client:
            return await client.get_pending_friend_requests()

    reqs = asyncio.run(_run())
    table = Table(title="Pending Friend Requests")
    table.add_column("Request ID", style="cyan")
    table.add_column("From")
    table.add_column("Message")
    for r in reqs:
        table.add_row(r.get("requestId", ""), r.get("fromNode", ""), r.get("message", ""))
    console.print(table)


@friend_app.command("accept")
def friend_accept(
    request_id: str = typer.Argument(..., help="Request ID"),
    org: str = typer.Option(None, "--org", help="Org ID (if multiple memberships)"),
):
    """Accept a pending friend request."""
    async def _run():
        client = await _client_from_config(org)
        async with client:
            await client.accept_friend_request(request_id)

    asyncio.run(_run())
    console.print("[green]✓[/green] Friend request accepted.")


@friend_app.command("list")
def friend_list(org: str = typer.Option(None, "--org", help="Org ID (if multiple memberships)")):
    """List your friends (node_ids)."""
    async def _run():
        client = await _client_from_config(org)
        async with client:
            return await client.get_friends()

    friends = asyncio.run(_run())
    for f in friends:
        console.print(f"- {f}")


@app.command("dm")
def dm(
    to: str = typer.Option(..., "--to", help="Target node_id (must be a friend)"),
    content: str = typer.Option(..., "--content", "-m", help="Message content"),
    org: str = typer.Option(None, "--org", help="Org ID (if multiple memberships)"),
):
    """Send a DM to a friend."""
    async def _run():
        client = await _client_from_config(org)
        async with client:
            await client.dm(to, content)

    asyncio.run(_run())
    console.print("[green]✓[/green] Sent.")


@app.command("listen")
def listen(org: str = typer.Option(None, "--org", help="Org ID (if multiple memberships)")):
    """Listen for inbound Universe events (DMs, friend requests, presence)."""

    async def _run():
        client = await _client_from_config(org)
        async with client:
            async def _handler(env):
                t = env.type
                if t == "dm_deliver":
                    console.print(f"[cyan]DM[/cyan] from {env.payload.get('fromNode')}: {env.payload.get('content')}")
                elif t == "friend_request_deliver":
                    console.print(
                        f"[cyan]Friend Request[/cyan] from {env.payload.get('fromNode')}: {env.payload.get('message')}\n"
                        f"  requestId={env.payload.get('requestId')}"
                    )
                elif t == "friend_added":
                    console.print(f"[green]Friend added[/green]: {env.payload.get('friendNode')}")
                elif t == "presence":
                    node = env.payload.get("nodeId")
                    online = env.payload.get("online")
                    console.print(f"[dim]presence[/dim] {node} -> {'online' if online else 'offline'}")
                else:
                    console.print(f"[dim]{t}[/dim] {env.payload}")

            console.print("[dim]Listening... Ctrl+C to stop[/dim]")
            await client.listen(_handler)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\nStopped.")


# ============================================================================
# Public Universe (central registry + direct calls)
# ============================================================================

public_app = typer.Typer(help="Public universe: opt-in registry discovery + direct task calls")
app.add_typer(public_app, name="public")

knowledge_app = typer.Typer(help="Public knowledge packs (free sharing)")
public_app.add_typer(knowledge_app, name="knowledge")


@public_app.command("enable")
def public_enable(
    registry: str = typer.Option(None, "--registry", help="Registry URL, e.g. ws://host:18999"),
    provide: bool = typer.Option(None, "--provide/--no-provide", help="Whether to provide public service"),
    token: str = typer.Option(None, "--service-token", help="Optional service token required by this node"),
    allow_agent_tasks: bool = typer.Option(
        None, "--allow-agent-tasks/--no-allow-agent-tasks", help="Allow this node to serve kind=evolvebot.agent"
    ),
    auto_delegate: bool = typer.Option(
        None, "--auto-delegate/--no-auto-delegate", help="Auto-delegate to universe when local run is blocked"
    ),
    auto_delegate_debug: bool = typer.Option(
        None, "--auto-delegate-debug/--no-auto-delegate-debug", help="Show debug info when auto-delegating"
    ),
    auto_register: bool = typer.Option(
        None, "--auto-register/--no-auto-register", help="Auto-register to registry at startup"
    ),
    advertise_url: str = typer.Option(
        None, "--advertise-url", help="Advertise URL override, e.g. ws://example.com:18998"
    ),
    advertise_host: str = typer.Option(
        None, "--advertise-host", help="Advertise host override (uses service port)"
    ),
    advertise_port: int = typer.Option(
        None, "--advertise-port", help="Advertise port override"
    ),
    detect_public_ip: bool = typer.Option(
        None, "--detect-public-ip/--no-detect-public-ip", help="Detect public IP for advertise URL"
    ),
    relay_url: str = typer.Option(None, "--relay-url", help="Relay URL, e.g. ws://relay-host:19001"),
    relay_token: str = typer.Option(None, "--relay-token", help="Relay token (if required)"),
    relay_only: bool = typer.Option(
        None, "--relay-only/--no-relay-only", help="Use relay only (disable direct endpoint exposure)"
    ),
):
    """Enable public universe mode (opt-in)."""
    cfg = load_config()
    _ensure_node_id()
    cfg.universe.public_enabled = True
    if registry is not None:
        cfg.universe.public_registry_url = registry
    if provide is not None:
        cfg.universe.public_provide_service = provide
    if token is not None:
        cfg.universe.public_service_token = token
    if allow_agent_tasks is not None:
        cfg.universe.public_allow_agent_tasks = allow_agent_tasks
    if auto_delegate is not None:
        cfg.universe.public_auto_delegate_enabled = auto_delegate
    if auto_delegate_debug is not None:
        cfg.universe.public_auto_delegate_debug = auto_delegate_debug
    if auto_register is not None:
        cfg.universe.public_auto_register = auto_register
    if advertise_url is not None:
        cfg.universe.public_advertise_url = advertise_url
    if advertise_host is not None:
        cfg.universe.public_advertise_host = advertise_host
    if advertise_port is not None:
        cfg.universe.public_advertise_port = advertise_port
    if detect_public_ip is not None:
        cfg.universe.public_detect_public_ip = detect_public_ip
    if relay_url is not None:
        cfg.universe.public_relay_url = relay_url
    if relay_token is not None:
        cfg.universe.public_relay_token = relay_token
    if relay_only is not None:
        cfg.universe.public_relay_only = relay_only
    save_config(cfg)
    console.print("[green]✓[/green] Public universe enabled.")


@public_app.command("disable")
def public_disable():
    """Disable public universe mode."""
    cfg = load_config()
    cfg.universe.public_enabled = False
    cfg.universe.public_provide_service = False
    save_config(cfg)
    console.print("[green]✓[/green] Public universe disabled.")


@public_app.command("register")
def public_register(
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    host: str = typer.Option(None, "--host", help="Bind host for node service"),
    port: int = typer.Option(None, "--port", help="Bind port for node service"),
    advertise: str = typer.Option(None, "--advertise", help="Advertised endpoint URL, e.g. ws://YOUR_IP:18998"),
    name: str = typer.Option(None, "--name", help="Node display name override"),
    price: int = typer.Option(None, "--price", help="Price points per task"),
    registry_token: str = typer.Option(None, "--registry-token", help="Registry token (if required)"),
):
    """Register this node and serve it publicly (same as public serve)."""
    cfg = load_config()
    if not cfg.universe.node_id:
        cfg.universe.node_id = str(uuid4())
    cfg.universe.public_enabled = True
    cfg.universe.public_provide_service = True
    cfg.universe.public_auto_register = True
    save_config(cfg)
    public_serve(
        registry=registry,
        host=host,
        port=port,
        advertise=advertise,
        name=name,
        price=price,
        registry_token=registry_token,
    )


@public_app.command("unregister")
def public_unregister(
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    registry_token: str = typer.Option(None, "--registry-token", help="Registry token (if required)"),
    node_id: str = typer.Option(None, "--node-id", help="Node ID override"),
):
    """Unregister this node from the public registry."""
    from evolvebot.universe.protocol import Envelope, make_envelope
    import websockets

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url
    token = registry_token or cfg.universe.public_registry_token or ""
    node = node_id or cfg.universe.node_id
    if not node:
        console.print("[red]Error: node_id is missing. Run `evolvebot onboard` or set it in config.[/red]")
        raise typer.Exit(1)

    async def _run():
        async with websockets.connect(reg) as ws:
            req = make_envelope(
                "unregister",
                payload={"nodeId": node, "registryToken": token},
            )
            await ws.send(req.to_json())
            return Envelope.from_json(await ws.recv())

    env = asyncio.run(_run())
    if env.type == "error":
        message = (env.payload or {}).get("message", "unregister failed")
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Unregistered {node} from {reg}")
    console.print("[dim]If a public service is still running, it may re-register. Stop the service to stay offline.[/dim]")


@public_app.command("list")
def public_list(
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    require_cap: str = typer.Option("llm.chat", "--require-cap", help="Required capability key"),
    all: bool = typer.Option(False, "--all", help="Include offline nodes"),
    show_card: bool = typer.Option(False, "--show-card", help="Show capabilityCard details"),
):
    """List nodes in the public registry."""
    from evolvebot.universe.protocol import Envelope, make_envelope
    import websockets
    import json

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url
    display_reg = reg
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(reg)
        if parts.hostname == "82.157.31.6":
            userinfo = ""
            if parts.username:
                userinfo = parts.username
                if parts.password:
                    userinfo += f":{parts.password}"
                userinfo += "@"
            netloc = f"{userinfo}evolvebot.work"
            if parts.port:
                netloc = f"{netloc}:{parts.port}"
            display_reg = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        display_reg = reg

    async def _run():
        async with websockets.connect(reg) as ws:
            req = make_envelope(
                "list",
                payload={
                    "onlineOnly": not all,
                    "requireCapabilities": [require_cap] if require_cap else [],
                },
            )
            await ws.send(req.to_json())
            # ack + result
            while True:
                env = Envelope.from_json(await ws.recv())
                if env.id == req.id and env.type in {"list_result", "error"}:
                    return env

    env = asyncio.run(_run())
    if env.type == "error":
        raise typer.Exit(code=1)

    table = Table(title=f"Public Nodes (registry={display_reg})")
    table.add_column("Node ID", style="cyan")
    table.add_column("Name")
    table.add_column("Price")
    table.add_column("Done")
    table.add_column("Points")
    for n in (env.payload or {}).get("nodes", []):
        table.add_row(
            n.get("nodeId", ""),
            n.get("nodeName", ""),
            str(n.get("pricePoints", "")),
            str(n.get("completedTasks", "")),
            str(n.get("earnedPoints", "")),
        )
    console.print(table)
    if show_card:
        for n in (env.payload or {}).get("nodes", []):
            node_id = n.get("nodeId", "")
            card = dict(n.get("capabilityCard", {}) or {})
            # Redact any endpoint URL to avoid leaking IPs in CLI output.
            card.pop("endpointUrl", None)
            if not card:
                continue
            console.print(f"[bold]capabilityCard[/bold] {node_id}")
            console.print(json.dumps(card, ensure_ascii=False, indent=2))


@knowledge_app.command("list")
def knowledge_list(
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    kind: str = typer.Option(None, "--kind", help="Filter by kind"),
    tag: str = typer.Option(None, "--tag", help="Filter by tag"),
    owner: str = typer.Option(None, "--owner", help="Filter by owner node_id"),
    limit: int = typer.Option(50, "--limit", help="Max entries"),
    include_pending: bool = typer.Option(False, "--include-pending", help="Include pending (requires token)"),
):
    """List public knowledge packs (free)."""
    from evolvebot.universe.public_client import knowledge_list as _knowledge_list

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url
    token = cfg.universe.public_registry_token or None

    async def _run():
        return await _knowledge_list(
            registry_url=reg,
            registry_token=token,
            kind=kind,
            tag=tag,
            owner_node=owner,
            limit=limit,
            include_pending=include_pending,
        )

    packs = asyncio.run(_run())
    table = Table(title=f"Knowledge Packs (registry={reg})")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Score")
    table.add_column("Version")
    table.add_column("Uses")
    table.add_column("Size")
    table.add_column("Owner")
    table.add_column("Tags")
    for p in packs:
        table.add_row(
            p.pack_id,
            p.name,
            p.kind,
            p.status,
            f"{p.score:.2f}",
            p.version,
            str(p.uses),
            str(p.size_bytes),
            p.owner_node,
            ",".join(p.tags or []),
        )
    console.print(table)


@knowledge_app.command("publish")
def knowledge_publish(
    file: str = typer.Option(None, "--file", help="JSON file with knowledge pack"),
    name: str = typer.Option(None, "--name", help="Pack name"),
    kind: str = typer.Option(None, "--kind", help="Pack kind (prompt|skill|workflow)"),
    content: str = typer.Option(None, "--content", help="Pack content (string)"),
    summary: str = typer.Option("", "--summary", help="Short summary"),
    tag: list[str] = typer.Option(None, "--tag", help="Tag (repeatable)"),
    version: str = typer.Option("1.0", "--version", help="Version"),
    pack_id: str = typer.Option(None, "--id", help="Optional pack ID"),
    allow_update: bool = typer.Option(False, "--allow-update", help="Allow update if ID exists"),
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    registry_token: str = typer.Option(None, "--registry-token", help="Registry token (if required)"),
):
    """Publish a knowledge pack (free, no points)."""
    import json as _json
    from pathlib import Path
    from evolvebot.universe.public_client import knowledge_publish as _knowledge_publish

    cfg = load_config()
    node_id = _ensure_node_id()
    reg = registry or cfg.universe.public_registry_url
    token = registry_token or cfg.universe.public_registry_token or ""

    payload: dict = {}
    if file:
        path = Path(file).expanduser()
        if not path.exists():
            console.print(f"[red]File not found:[/red] {path}")
            raise typer.Exit(1)
        payload = _json.loads(path.read_text())

    name = name or payload.get("name")
    kind = kind or payload.get("kind")
    content = content or payload.get("content")
    summary = summary or payload.get("summary", "")
    version = version or payload.get("version", "1.0")
    tags = tag or payload.get("tags", []) or []
    pack_id = pack_id or payload.get("id")

    if not name or not kind or not content:
        console.print("[red]Missing required fields: name/kind/content[/red]")
        raise typer.Exit(1)

    async def _run():
        return await _knowledge_publish(
            registry_url=reg,
            registry_token=token,
            name=name,
            kind=kind,
            content=content,
            summary=summary,
            tags=tags,
            version=version,
            pack_id=pack_id,
            owner_node=node_id,
            allow_update=allow_update,
        )

    resp = asyncio.run(_run())
    console.print(f"[green]✓[/green] Published knowledge pack: {resp.get('id')}")
    console.print(f"sizeBytes={resp.get('sizeBytes')} hash={resp.get('contentHash')}")


@knowledge_app.command("fetch")
def knowledge_fetch(
    pack_id: str = typer.Option(..., "--id", help="Knowledge pack ID"),
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    registry_token: str = typer.Option(None, "--registry-token", help="Registry token (if required)"),
    save_dir: str = typer.Option(None, "--save-dir", help="Save directory (default: ~/.evolvebot/universe_inbox)"),
):
    """Fetch a knowledge pack into local inbox (no auto-exec)."""
    from evolvebot.universe.public_client import knowledge_get as _knowledge_get
    from evolvebot.universe.knowledge_store import save_pack

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url
    token = registry_token or cfg.universe.public_registry_token or None
    inbox_dir = save_dir or (cfg.universe.public_knowledge_inbox_dir or None)

    async def _run():
        return await _knowledge_get(
            registry_url=reg,
            pack_id=pack_id,
            registry_token=token,
        )

    pack = asyncio.run(_run())
    path = save_pack(pack, inbox_dir=inbox_dir)
    console.print(f"[green]✓[/green] Saved to {path}")
    console.print("[dim]Review this pack before enabling or copying into skills/prompts.[/dim]")


@knowledge_app.command("apply")
def knowledge_apply(
    pack_id: str = typer.Option(..., "--id", help="Knowledge pack ID"),
    name: str = typer.Option(None, "--name", help="Override local skill name"),
    always: bool = typer.Option(True, "--always/--no-always", help="Mark skill as always loaded"),
    inbox_dir: str = typer.Option(None, "--inbox-dir", help="Inbox dir override"),
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    registry_token: str = typer.Option(None, "--registry-token", help="Registry token (if required)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing skill"),
):
    """Apply a knowledge pack as a local skill (safe; no auto-exec)."""
    from evolvebot.universe.public_client import knowledge_get as _knowledge_get
    from evolvebot.universe.knowledge_store import find_pack_in_inbox, save_pack
    from evolvebot.utils.helpers import safe_filename

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url
    token = registry_token or cfg.universe.public_registry_token or None
    effective_inbox = inbox_dir or (cfg.universe.public_knowledge_inbox_dir or None)

    pack = None
    try:
        pack = find_pack_in_inbox(pack_id, inbox_dir=effective_inbox)
    except Exception:
        pack = None

    if not pack:
        async def _run():
            return await _knowledge_get(registry_url=reg, pack_id=pack_id, registry_token=token)

        try:
            pack = asyncio.run(_run())
            save_pack(pack, inbox_dir=effective_inbox)
        except Exception as e:
            console.print(f"[red]Failed to load pack {pack_id}: {e}[/red]")
            raise typer.Exit(code=1)

    if not pack:
        console.print(f"[red]Pack not found: {pack_id}[/red]")
        raise typer.Exit(code=1)

    workspace = cfg.workspace_path
    skills_dir = workspace / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    skill_name = safe_filename(name or pack.name or f"knowledge_{pack.pack_id[:8]}")
    if not skill_name:
        skill_name = f"knowledge_{pack.pack_id[:8]}"
    skill_dir = skills_dir / skill_name
    skill_file = skill_dir / "SKILL.md"

    if skill_file.exists() and not overwrite:
        console.print(f"[red]Skill already exists: {skill_dir} (use --overwrite)[/red]")
        raise typer.Exit(code=1)
    skill_dir.mkdir(parents=True, exist_ok=True)

    desc = pack.summary or pack.name or "Knowledge pack"
    desc = desc.replace('"', '\\"')
    tags = ",".join(pack.tags or [])
    frontmatter = [
        "---",
        f"description: \"{desc}\"",
        f"source: \"public_universe\"",
        f"pack_id: \"{pack.pack_id}\"",
        f"kind: \"{pack.kind}\"",
        f"owner: \"{pack.owner_node}\"",
        f"version: \"{pack.version}\"",
        f"content_hash: \"{pack.content_hash}\"",
        f"size_bytes: \"{pack.size_bytes}\"",
        f"tags: \"{tags}\"",
        f"always: \"{str(always).lower()}\"",
        "---",
        "",
    ]
    body = pack.content.strip() + "\n"
    skill_file.write_text("\n".join(frontmatter) + body, encoding="utf-8")

    console.print(f"[green]✓[/green] Applied pack as skill: {skill_dir}")
    console.print("[dim]Restart evolvebot agent or start a new session to load the new skill.[/dim]")


@knowledge_app.command("rate")
def knowledge_rate(
    pack_id: str = typer.Option(..., "--id", help="Knowledge pack ID"),
    up: bool = typer.Option(False, "--up", help="Upvote"),
    down: bool = typer.Option(False, "--down", help="Downvote"),
    delta: int = typer.Option(0, "--delta", help="Custom delta (+1 or -1)"),
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    registry_token: str = typer.Option(None, "--registry-token", help="Registry token (if required)"),
):
    """Rate a knowledge pack (up/down vote)."""
    from evolvebot.universe.public_client import knowledge_rate as _knowledge_rate

    if delta == 0:
        if up and down:
            console.print("[red]Choose only one of --up or --down[/red]")
            raise typer.Exit(1)
        if not up and not down:
            console.print("[red]Specify --up or --down (or --delta)[/red]")
            raise typer.Exit(1)
        delta = 1 if up else -1

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url
    token = registry_token or cfg.universe.public_registry_token or None

    async def _run():
        return await _knowledge_rate(
            registry_url=reg,
            pack_id=pack_id,
            delta=delta,
            registry_token=token,
        )

    resp = asyncio.run(_run())
    console.print(f"[green]✓[/green] Rated pack {resp.get('id')}")
    console.print(f"score={resp.get('score')} status={resp.get('status')} up={resp.get('upvotes')} down={resp.get('downvotes')}")


@public_app.command("leaderboard")
def public_leaderboard(
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    limit: int = typer.Option(20, "--limit", help="Max entries"),
    sort_by: str = typer.Option("earnedPoints", "--sort-by", help="earnedPoints | balance | completedTasks"),
):
    """Show public universe points leaderboard."""
    from evolvebot.universe.protocol import Envelope, make_envelope
    import websockets

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url

    async def _run():
        async with websockets.connect(reg) as ws:
            req = make_envelope("leaderboard", payload={"limit": limit, "sortBy": sort_by})
            await ws.send(req.to_json())
            while True:
                env = Envelope.from_json(await ws.recv())
                if env.id == req.id and env.type in {"leaderboard_result", "error"}:
                    return env

    env = asyncio.run(_run())
    if env.type == "error":
        raise typer.Exit(code=1)

    table = Table(title=f"Public Leaderboard (registry={reg})")
    table.add_column("Node ID", style="cyan")
    table.add_column("Name")
    table.add_column("Balance")
    table.add_column("Earned")
    table.add_column("Spent")
    table.add_column("Done")
    for n in (env.payload or {}).get("nodes", []):
        table.add_row(
            n.get("nodeId", ""),
            n.get("nodeName", ""),
            str(n.get("balance", "")),
            str(n.get("earnedPoints", "")),
            str(n.get("spentPoints", "")),
            str(n.get("completedTasks", "")),
        )
    console.print(table)


@public_app.command("serve")
def public_serve(
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    host: str = typer.Option(None, "--host", help="Bind host for node service"),
    port: int = typer.Option(None, "--port", help="Bind port for node service"),
    advertise: str = typer.Option(None, "--advertise", help="Advertised endpoint URL, e.g. ws://YOUR_IP:18998"),
    name: str = typer.Option(None, "--name", help="Node display name override"),
    price: int = typer.Option(None, "--price", help="Price points per task"),
    registry_token: str = typer.Option(None, "--registry-token", help="Registry token (if required)"),
):
    """Run a public service node and register it in the registry."""
    from evolvebot.universe.public_service import start_public_service, stop_public_service

    cfg = load_config()
    node_id = _ensure_node_id()
    if not cfg.universe.public_enabled:
        console.print("[yellow]![/yellow] Public universe is disabled. Run: evolvebot universe public enable")
        raise typer.Exit(1)
    if not cfg.universe.public_provide_service:
        console.print("[yellow]![/yellow] This node is not set to provide service. Re-run enable with --provide")
        raise typer.Exit(1)

    async def _run():
        # Apply overrides on a local copy (do not persist).
        cfg_local = cfg.model_copy(deep=True)
        if registry is not None:
            cfg_local.universe.public_registry_url = registry
        if host is not None:
            cfg_local.universe.public_service_host = host
        if port is not None:
            cfg_local.universe.public_service_port = port
        if advertise is not None:
            cfg_local.universe.public_advertise_url = advertise
        if name is not None:
            cfg_local.universe.node_name = name
        if price is not None:
            cfg_local.universe.public_price_points = int(price)
        if registry_token is not None:
            cfg_local.universe.public_registry_token = registry_token

        handle = await start_public_service(cfg_local, log_prefix="universe")
        try:
            console.print("[dim]Serving... Ctrl+C to stop[/dim]")
            await asyncio.Future()
        finally:
            await stop_public_service(handle)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\nStopped.")


@public_app.command("call")
def public_call(
    prompt: str = typer.Option(..., "--prompt", "-m", help="Task prompt"),
    registry: str = typer.Option(None, "--registry", help="Registry URL override"),
    kind: str = typer.Option("llm.chat", "--kind", help="Task kind: llm.chat | echo"),
    require_cap: str = typer.Option("", "--require-cap", help="Required capability key (default: kind)"),
    to: str = typer.Option(None, "--to", help="Directly call a specific node_id (skip selection)"),
    service_token: str = typer.Option("", "--service-token", help="Service token if required by provider node"),
    client_id: str = typer.Option("", "--client-id", help="Client node_id for rate limiting (optional)"),
    award_points: int = typer.Option(0, "--award", help="Award points to the provider in registry (MVP bookkeeping)"),
    registry_token: str = typer.Option("", "--registry-token", help="Registry token (if award requires it)"),
    relay: str = typer.Option(None, "--relay", help="Relay URL override (ws://...)"),
    relay_token: str = typer.Option("", "--relay-token", help="Relay token (if required)"),
    direct: bool = typer.Option(False, "--direct", help="Force direct call (skip relay)"),
):
    """Call a public node to execute a task (direct node-to-node)."""
    import websockets

    from evolvebot.universe.protocol import Envelope, make_envelope
    from evolvebot.universe.public_client import (
        reserve_points,
        commit_reservation,
        cancel_reservation,
        report_task,
        knowledge_list,
        knowledge_get,
    )
    from evolvebot.universe.knowledge_store import save_pack
    import time

    cfg = load_config()
    reg = registry or cfg.universe.public_registry_url
    cap = require_cap or kind
    effective_client_id = client_id or cfg.universe.node_id or None
    effective_registry_token = registry_token or cfg.universe.public_registry_token or ""
    relay_url = None if direct else (relay or cfg.universe.public_relay_url or None)
    effective_relay_token = relay_token or cfg.universe.public_relay_token or ""
    relay_only = bool(cfg.universe.public_relay_only) and not direct
    preauth_enabled = bool(getattr(cfg.universe, "public_preauth_enabled", True))
    preauth_required = bool(getattr(cfg.universe, "public_preauth_required", False))

    async def _pick_target() -> dict:
        async with websockets.connect(reg) as ws:
            req = make_envelope(
                "list",
                payload={"onlineOnly": True, "requireCapabilities": [cap] if cap else []},
            )
            await ws.send(req.to_json())
            while True:
                env = Envelope.from_json(await ws.recv())
                if env.id == req.id and env.type == "list_result":
                    nodes = (env.payload or {}).get("nodes", [])
                    if to:
                        for n in nodes:
                            if n.get("nodeId") == to:
                                return n
                        raise RuntimeError(f"node not found/online: {to}")
                    if not nodes:
                        raise RuntimeError("no online nodes match requirement")
                    # MVP selection: cheapest first, then arbitrary.
                    nodes.sort(key=lambda n: int(n.get("pricePoints", 1) or 1))
                    return nodes[0]
        raise RuntimeError("registry did not respond")

    async def _resolve_endpoint(node_id: str) -> str:
        if not effective_registry_token:
            raise RuntimeError("registry token required to resolve endpoint")
        async with websockets.connect(reg) as ws:
            req = make_envelope("resolve", payload={"nodeId": node_id, "registryToken": effective_registry_token})
            await ws.send(req.to_json())
            while True:
                env = Envelope.from_json(await ws.recv())
                if env.id != req.id:
                    continue
                if env.type == "error":
                    raise RuntimeError((env.payload or {}).get("message", "resolve failed"))
                if env.type == "resolve_ok":
                    endpoint = (env.payload or {}).get("endpointUrl", "")
                    if not endpoint:
                        raise RuntimeError("endpoint not available")
                    return endpoint

    async def _call(endpoint_url: str, node_id: str) -> str:
        async with websockets.connect(endpoint_url) as ws:
            req = make_envelope(
                "task_run",
                payload={"kind": kind, "prompt": prompt, "serviceToken": service_token, "clientId": effective_client_id},
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
        raise RuntimeError("provider disconnected")

    async def _award(node_id: str):
        if award_points <= 0:
            return
        async with websockets.connect(reg) as ws:
            req = make_envelope(
                "award",
                payload={
                    "nodeId": node_id,
                    "points": award_points,
                    "registryToken": effective_registry_token,
                    "payerNode": effective_client_id or "",
                },
            )
            await ws.send(req.to_json())
            env = Envelope.from_json(await ws.recv())
            if env.type != "award_ok":
                raise RuntimeError((env.payload or {}).get("message", "award failed"))

    async def _call_relay(node_id: str) -> str:
        async with websockets.connect(relay_url) as ws:
            req = make_envelope(
                "relay_request",
                payload={
                    "nodeId": node_id,
                    "kind": kind,
                    "prompt": prompt,
                    "serviceToken": service_token,
                    "clientId": effective_client_id,
                    "relayToken": effective_relay_token,
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

    async def _run():
        target = await _pick_target()
        node_id = target.get("nodeId", "")
        if not node_id:
            raise RuntimeError("target missing nodeId")
        points_to_charge = int(award_points or (target.get("pricePoints", 1) or 1))
        reservation_id = ""
        start = time.monotonic()
        if preauth_enabled:
            if not effective_registry_token or not effective_client_id:
                if preauth_required:
                    raise RuntimeError("preauth requires registry_token and client_id")
            else:
                reservation_id = await reserve_points(
                    registry_url=reg,
                    registry_token=effective_registry_token,
                    payer_node_id=str(effective_client_id),
                    provider_node_id=node_id,
                    points=points_to_charge,
                )
        if relay_url:
            try:
                out = await _call_relay(node_id)
            except Exception:
                if effective_registry_token:
                    try:
                        await report_task(
                            registry_url=reg,
                            registry_token=effective_registry_token,
                            node_id=node_id,
                            ok=False,
                            latency_ms=int((time.monotonic() - start) * 1000),
                        )
                    except Exception:
                        pass
                if reservation_id:
                    try:
                        await cancel_reservation(
                            registry_url=reg,
                            registry_token=effective_registry_token,
                            reservation_id=reservation_id,
                        )
                    except Exception:
                        pass
                raise
        else:
            if relay_only:
                raise RuntimeError("relay_only enabled but relay URL is not configured")
            try:
                endpoint = await _resolve_endpoint(node_id)
                out = await _call(endpoint, node_id)
            except Exception:
                if effective_registry_token:
                    try:
                        await report_task(
                            registry_url=reg,
                            registry_token=effective_registry_token,
                            node_id=node_id,
                            ok=False,
                            latency_ms=int((time.monotonic() - start) * 1000),
                        )
                    except Exception:
                        pass
                if reservation_id:
                    try:
                        await cancel_reservation(
                            registry_url=reg,
                            registry_token=effective_registry_token,
                            reservation_id=reservation_id,
                        )
                    except Exception:
                        pass
                raise
        if reservation_id:
            await commit_reservation(
                registry_url=reg,
                registry_token=effective_registry_token,
                reservation_id=reservation_id,
            )
        else:
            await _award(node_id)
        if effective_registry_token:
            try:
                await report_task(
                    registry_url=reg,
                    registry_token=effective_registry_token,
                    node_id=node_id,
                    ok=True,
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception:
                pass
        if getattr(cfg.universe, "public_knowledge_auto_pull", False):
            try:
                limit = max(1, int(getattr(cfg.universe, "public_knowledge_auto_pull_limit", 1) or 1))
                tagged_only = bool(getattr(cfg.universe, "public_knowledge_auto_pull_tagged_only", True))
                inbox_dir = getattr(cfg.universe, "public_knowledge_inbox_dir", "") or None
                packs = await knowledge_list(
                    registry_url=reg,
                    registry_token=effective_registry_token or None,
                    owner_node=node_id,
                    tag=cap if tagged_only and cap else None,
                    limit=limit,
                )
                if not packs and tagged_only:
                    packs = await knowledge_list(
                        registry_url=reg,
                        registry_token=effective_registry_token or None,
                        owner_node=node_id,
                        tag=None,
                        limit=limit,
                    )
                for meta in packs[:limit]:
                    pack = await knowledge_get(
                        registry_url=reg,
                        registry_token=effective_registry_token or None,
                        pack_id=meta.pack_id,
                    )
                    save_pack(pack, inbox_dir=inbox_dir)
            except Exception:
                pass
        return node_id, out

    node_id, out = asyncio.run(_run())
    console.print(f"[dim]provider[/dim] {node_id}")
    console.print(out)
