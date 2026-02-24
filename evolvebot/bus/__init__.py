"""Message bus module for decoupled channel-agent communication."""

from evolvebot.bus.events import InboundMessage, OutboundMessage
from evolvebot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
