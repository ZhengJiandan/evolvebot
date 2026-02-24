"""Console entrypoint for the Public Universe Registry (independent process)."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import typer
from rich.console import Console

from evolvebot.config.loader import get_data_dir
from evolvebot.universe.registry_server import RegistryServer, RegistryServerConfig

app = typer.Typer(name="evolvebot-universe-registry", help="Public Universe registry (central directory)")
console = Console()


@app.command("run")
def run(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(18999, "--port", help="Bind port"),
    token: str = typer.Option("", "--token", help="Optional registry token (required for register/update/award)"),
    state_file: str = typer.Option("", "--state-file", help="Path to registry state file"),
    ttl: int = typer.Option(120, "--ttl", help="Offline TTL for nodes (seconds)"),
    metrics_port: int = typer.Option(None, "--metrics-port", help="Enable /health and /metrics on this port"),
    rate_limit: int = typer.Option(120, "--rate-limit", help="Rate limit per client IP (per minute)"),
    rate_burst: int = typer.Option(120, "--rate-burst", help="Rate limit burst per client IP"),
    initial_points: int = typer.Option(10, "--initial-points", help="Initial points granted per new node"),
    preauth_ttl: int = typer.Option(300, "--preauth-ttl", help="Preauth reservation TTL (seconds)"),
    knowledge_max_bytes: int = typer.Option(50000, "--knowledge-max-bytes", help="Max knowledge pack size in bytes"),
    foreground: bool = typer.Option(False, "--foreground/--background", help="Run in foreground"),
):
    """Run the registry server."""
    if not foreground:
        log_path = get_data_dir() / "registry.log"
        cmd = [
            sys.executable,
            "-m",
            "evolvebot.universe.registry_cli",
            "--foreground",
            "--host",
            host,
            "--port",
            str(port),
            "--ttl",
            str(ttl),
            "--rate-limit",
            str(rate_limit),
            "--rate-burst",
            str(rate_burst),
            "--initial-points",
            str(initial_points),
            "--preauth-ttl",
            str(preauth_ttl),
            "--knowledge-max-bytes",
            str(knowledge_max_bytes),
        ]
        if token:
            cmd += ["--token", token]
        if state_file:
            cmd += ["--state-file", state_file]
        if metrics_port is not None:
            cmd += ["--metrics-port", str(metrics_port)]

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", buffering=1) as log_file:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
                close_fds=True,
            )
        console.print(f"[green]✓[/green] Registry started in background (pid={proc.pid})")
        console.print(f"[dim]Logs: {log_path}[/dim]")
        return

    server = RegistryServer(
        cfg=RegistryServerConfig(
            host=host,
            port=port,
            registry_token=token,
            state_file=state_file,
            ttl_seconds=ttl,
            metrics_port=metrics_port,
            rate_limit_per_min=rate_limit,
            rate_limit_burst=rate_burst,
            initial_points=initial_points,
            preauth_ttl_seconds=preauth_ttl,
            knowledge_max_bytes=knowledge_max_bytes,
        )
    )

    async def _main():
        await server.start()
        await asyncio.Future()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\nStopping registry...")


if __name__ == "__main__":
    app()
