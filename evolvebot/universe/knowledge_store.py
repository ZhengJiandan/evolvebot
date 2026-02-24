"""Local storage helpers for knowledge packs (inbox/outbox)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evolvebot.utils.helpers import get_data_path, ensure_dir, safe_filename
from evolvebot.universe.public_client import KnowledgePack


def get_inbox_path(base: str | None = None) -> Path:
    if base:
        return ensure_dir(Path(base).expanduser())
    return ensure_dir(get_data_path() / "universe_inbox")


def _manifest_path(inbox: Path) -> Path:
    return inbox / "manifest.json"


def _load_manifest(inbox: Path) -> dict[str, Any]:
    path = _manifest_path(inbox)
    if not path.exists():
        return {"packs": []}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"packs": []}


def _save_manifest(inbox: Path, data: dict[str, Any]) -> None:
    path = _manifest_path(inbox)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def save_pack(pack: KnowledgePack, *, inbox_dir: str | None = None) -> Path:
    inbox = get_inbox_path(inbox_dir)
    manifest = _load_manifest(inbox)
    known = {(p.get("id"), p.get("contentHash")) for p in manifest.get("packs", []) if isinstance(p, dict)}
    key = (pack.pack_id, pack.content_hash)
    if key in known:
        return inbox / f"{pack.pack_id}_{safe_filename(pack.name)}.json"

    fname = f"{pack.pack_id}_{safe_filename(pack.name)}.json"
    path = inbox / fname
    data = asdict(pack)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    manifest.setdefault("packs", []).append(
        {
            "id": pack.pack_id,
            "name": pack.name,
            "kind": pack.kind,
            "version": pack.version,
            "contentHash": pack.content_hash,
            "savedAt": __import__("time").time(),
            "file": str(path.name),
        }
    )
    _save_manifest(inbox, manifest)
    return path


def load_pack_from_dict(data: dict[str, Any]) -> KnowledgePack:
    def _get(key: str, alt: str | None = None, default: Any = "") -> Any:
        if key in data:
            return data.get(key)
        if alt and alt in data:
            return data.get(alt)
        return default

    return KnowledgePack(
        pack_id=str(_get("pack_id", "id", "")),
        name=str(_get("name", None, "")),
        kind=str(_get("kind", None, "")),
        summary=str(_get("summary", None, "")),
        tags=list(_get("tags", None, []) or []),
        version=str(_get("version", None, "1.0")),
        owner_node=str(_get("owner_node", "ownerNode", "")),
        status=str(_get("status", None, "pending")),
        score=float(_get("score", None, 0) or 0),
        quality_score=int(_get("quality_score", "qualityScore", 0) or 0),
        uses=int(_get("uses", None, 0) or 0),
        upvotes=int(_get("upvotes", None, 0) or 0),
        downvotes=int(_get("downvotes", None, 0) or 0),
        created_ts=float(_get("created_ts", "createdTs", 0) or 0),
        updated_ts=float(_get("updated_ts", "updatedTs", 0) or 0),
        content_hash=str(_get("content_hash", "contentHash", "")),
        size_bytes=int(_get("size_bytes", "sizeBytes", 0) or 0),
        content=str(_get("content", None, "")),
    )


def load_pack_file(path: Path) -> KnowledgePack:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("invalid knowledge pack file")
    return load_pack_from_dict(data)


def find_pack_in_inbox(pack_id: str, *, inbox_dir: str | None = None) -> KnowledgePack | None:
    inbox = get_inbox_path(inbox_dir)
    manifest = _load_manifest(inbox)
    for entry in manifest.get("packs", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("id") != pack_id:
            continue
        fname = entry.get("file")
        if not fname:
            continue
        path = inbox / fname
        if path.exists():
            try:
                return load_pack_file(path)
            except Exception:
                continue
    # Fallback: scan files if manifest is missing/stale.
    for path in inbox.glob("*.json"):
        try:
            pack = load_pack_file(path)
        except Exception:
            continue
        if pack.pack_id == pack_id:
            return pack
    return None
