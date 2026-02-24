"""Console entrypoint for the Public Universe Relay (independent process)."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import typer
from rich.console import Console

from evolvebot.config.loader import get_data_dir
from evolvebot.universe.relay_server import RelayServer, RelayServerConfig

app = typer.Typer(name="evolvebot-universe-relay", help="Public Universe relay (task forwarding)")
console = Console()


@app.command("run")
def run(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(19001, "--port", help="Bind port"),
    token: str = typer.Option("", "--token", help="Optional relay token (shared secret)"),
    foreground: bool = typer.Option(False, "--foreground/--background", help="Run in foreground"),
):
    """Run the relay server."""
    if not foreground:
        log_path = get_data_dir() / "relay.log"
        cmd = [
            sys.executable,
            "-m",
            "evolvebot.universe.relay_cli",
            "--foreground",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if token:
            cmd += ["--token", token]

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
        console.print(f"[green]✓[/green] Relay started in background (pid={proc.pid})")
        console.print(f"[dim]Logs: {log_path}[/dim]")
        return

    server = RelayServer(cfg=RelayServerConfig(host=host, port=port, relay_token=token))

    async def _main():
        await server.start()
        await asyncio.Future()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\nStopping relay...")


if __name__ == "__main__":
    app()
