"""Console entrypoint for the Universe Hub (independent process)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import typer
from rich.console import Console

from evolvebot.universe.hub_server import HubServer, HubServerConfig
from evolvebot.universe.hub_state import HubState

app = typer.Typer(name="evolvebot-universe-hub", help="Universe hub (org-scoped evolvebot network)")
console = Console()


def _parse_org_secret(spec: str) -> tuple[str, str]:
    # Accept "org_id:secret" or "org_id=secret"
    if ":" in spec:
        org_id, secret = spec.split(":", 1)
    elif "=" in spec:
        org_id, secret = spec.split("=", 1)
    else:
        raise typer.BadParameter("ORG must be in format org_id:join_secret")
    org_id = org_id.strip()
    secret = secret.strip()
    if not org_id or not secret:
        raise typer.BadParameter("ORG and join_secret cannot be empty")
    return org_id, secret


@app.command("run")
def run(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(18888, "--port", help="Bind port"),
    org: list[str] = typer.Option(
        ...,
        "--org",
        help="Org join secret mapping, repeatable: --org acme:secret --org lab:secret",
    ),
):
    """Run the Universe hub server."""
    org_secrets: dict[str, str] = {}
    for spec in org:
        org_id, secret = _parse_org_secret(spec)
        org_secrets[org_id] = secret

    state = HubState(org_join_secrets=org_secrets)
    server = HubServer(state, HubServerConfig(host=host, port=port))

    async def _main():
        await server.start()
        # Run forever
        await asyncio.Future()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\nStopping hub...")


if __name__ == "__main__":
    app()
