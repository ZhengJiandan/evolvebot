"""evolvebot Universe (multi-node) networking.

This module implements an MVP org-scoped "universe hub" plus a thin client.
Security note: the MVP focuses on plumbing (presence/friends/dm). Strong E2EE and
public-key identities can be layered on later without changing the high-level API.
"""

from .protocol import PROTOCOL_VERSION

__all__ = ["PROTOCOL_VERSION"]

