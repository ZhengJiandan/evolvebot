"""Universe wire protocol helpers (JSON over WebSocket)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Envelope(BaseModel):
    """Top-level message wrapper.

    We keep payload flexible in the MVP to iterate quickly, while still
    validating the outer envelope consistently.
    """

    v: int = Field(default=PROTOCOL_VERSION)
    type: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    ts: str = Field(default_factory=utc_now_iso)

    org_id: str | None = None
    from_node: str | None = None
    to_node: str | None = None

    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def from_json(data: str) -> "Envelope":
        return Envelope.model_validate(json.loads(data))


def make_envelope(msg_type: str, **kwargs: Any) -> Envelope:
    return Envelope(type=msg_type, **kwargs)

