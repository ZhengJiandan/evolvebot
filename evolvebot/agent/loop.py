"""Agent loop: the core processing engine."""

import asyncio
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
from typing import Any

from loguru import logger

from evolvebot.bus.events import InboundMessage, OutboundMessage
from evolvebot.bus.queue import MessageBus
from evolvebot.providers.base import LLMProvider
from evolvebot.agent.context import ContextBuilder
from evolvebot.agent.tools.registry import ToolRegistry
from evolvebot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from evolvebot.agent.tools.shell import ExecTool
from evolvebot.agent.tools.web import WebSearchTool, WebFetchTool
from evolvebot.agent.tools.message import MessageTool
from evolvebot.agent.tools.spawn import SpawnTool
from evolvebot.agent.tools.cron import CronTool
from evolvebot.agent.tools.universe import UniverseHelpTool
from evolvebot.agent.memory import MemoryStore
from evolvebot.agent.subagent import SubagentManager
from evolvebot.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        universe_config: "UniverseConfig | None" = None,
    ):
        from evolvebot.config.schema import ExecToolConfig
        from evolvebot.config.schema import UniverseConfig
        from evolvebot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.universe_config = universe_config or UniverseConfig()

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.learning = None
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._register_default_tools()
        self._init_learning_manager()

    def _init_learning_manager(self) -> None:
        """Initialize learning manager (optional)."""
        try:
            from evolvebot.universe.learning import LearningManager
            from evolvebot.config.schema import Config

            self.learning = LearningManager(
                Config.model_validate({"universe": self.universe_config.model_dump()}),
                provider=self.provider,
                model=self.model,
            )
        except Exception as e:
            logger.warning(f"learning manager init failed: {e}")
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Universe delegation (public network)
        self.tools.register(UniverseHelpTool())
    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from evolvebot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str], list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        tool_errors: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    if isinstance(result, str) and result.startswith("Error:"):
                        # Keep a small sample for auto-delegation heuristics.
                        if len(tool_errors) < 3:
                            tool_errors.append(result[:200])
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                final_content = response.content
                break

        return final_content, tools_used, tool_errors

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 evolvebot commands:\n/new — Start a new conversation\n/help — Show available commands")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used, tool_errors = await self._run_agent_loop(initial_messages)

        # Auto-delegate to public universe when locally blocked (opt-in).
        final_content = await self._maybe_delegate_public(msg.content, final_content, tool_errors)
        if self.learning:
            asyncio.create_task(self.learning.maybe_task_driven_learn(
                task_prompt=msg.content,
                tool_errors=tool_errors,
            ))
            self.learning.record_task_summary(
                prompt=msg.content,
                answer=final_content,
                tool_errors=tool_errors,
            )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _, tool_errors = await self._run_agent_loop(initial_messages)
        final_content = await self._maybe_delegate_public(msg.content, final_content, tool_errors)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def _maybe_delegate_public(self, user_prompt: str, local_answer: str | None, tool_errors: list[str]) -> str | None:
        """Optionally auto-delegate to public universe nodes when local run looks blocked.

        This is intentionally conservative and requires explicit opt-in via config.
        """
        uc = self.universe_config
        debug = getattr(uc, "public_auto_delegate_debug", False)
        if not getattr(uc, "public_enabled", False):
            return local_answer
        if not getattr(uc, "public_auto_delegate_enabled", False):
            return local_answer
        if debug:
            relay_url = getattr(uc, "public_relay_url", "") or ""
            relay_only = bool(getattr(uc, "public_relay_only", False))
            registry_url = getattr(uc, "public_registry_url", "") or ""
            registry_token = getattr(uc, "public_registry_token", "") or ""
            logger.info(
                "[universe-debug] config relay_url=%s relay_only=%s registry_url=%s registry_token=%s",
                relay_url,
                relay_only,
                registry_url,
                bool(registry_token),
            )

        # Heuristics: tool errors (missing keys, blocked network, etc.) or explicit "can't" responses.
        blocked_phrases = (
            "i can't",
            "i cannot",
            "can't access",
            "unable to",
            "无法",
            "不能",
            "做不到",
            "Error: BRAVE_API_KEY not configured".lower(),
        )
        ans = (local_answer or "").strip()
        looks_blocked = False
        if getattr(uc, "public_auto_delegate_on_tool_error", True) and tool_errors:
            looks_blocked = True
        if ans:
            low = ans.lower()
            if any(p in low for p in blocked_phrases):
                looks_blocked = True

        if not looks_blocked:
            return local_answer

        from evolvebot.universe.public_client import delegate_task

        required_caps = await self._infer_required_caps_async(user_prompt, local_answer, tool_errors)

        # Provide a tiny bit of context; avoid leaking tool traces by default.
        prompt = user_prompt
        if tool_errors:
            prompt = f"{user_prompt}\n\n(Notes: local run hit tool errors: {', '.join(tool_errors)})"

        last_err: Exception | None = None
        for cap in required_caps:
            try:
                node, out = await delegate_task(
                    registry_url=uc.public_registry_url,
                    kind=getattr(uc, "public_auto_delegate_kind", "evolvebot.agent"),
                    prompt=prompt,
                    require_capability=cap,
                    max_price_points=getattr(uc, "public_auto_delegate_max_price_points", None),
                    client_id=uc.node_id or None,
                    registry_token=uc.public_registry_token or None,
                    relay_url=getattr(uc, "public_relay_url", "") or None,
                    relay_token=getattr(uc, "public_relay_token", "") or "",
                    relay_only=bool(getattr(uc, "public_relay_only", False)),
                    preauth_enabled=bool(getattr(uc, "public_preauth_enabled", True)),
                    preauth_required=bool(getattr(uc, "public_preauth_required", False)),
                )
                if getattr(uc, "public_knowledge_auto_pull", False):
                    asyncio.create_task(self._auto_pull_knowledge(node.node_id, cap))
                if debug:
                    return f"[universe-debug] delegated to {node.node_id} (cap={cap})\n[universe:{node.node_id}] {out}"
                return f"[universe:{node.node_id}] {out}"
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "no eligible nodes" in msg or "no online nodes" in msg or "node not found" in msg:
                    continue
                # Other errors: continue to next capability as a fallback.
                continue

        # If delegation fails, fall back to the local answer.
        if last_err:
            logger.warning(f"public universe auto-delegation failed: {last_err}")
            if debug:
                prefix = f"[universe-debug] delegation failed: {last_err}"
                if local_answer:
                    return f"{prefix}\n{local_answer}"
                return prefix
        return local_answer

    def _infer_required_caps(self, user_prompt: str, local_answer: str | None, tool_errors: list[str]) -> list[str]:
        caps: list[str] = []
        text = f"{user_prompt}\n{local_answer or ''}".lower()

        def add(cap: str) -> None:
            if cap and cap not in caps:
                caps.append(cap)

        for err in tool_errors:
            low = err.lower()
            if "brave_api_key" in low or "web_search" in low:
                add("web_search")
            if "web_fetch" in low or "http" in low or "fetch" in low:
                add("web_fetch")
            if "exec" in low or "shell" in low or "permission" in low:
                add("exec")

        if any(k in text for k in ("search", "lookup", "latest", "news", "查", "搜", "搜索", "最新", "新闻")):
            add("web_search")
        if any(k in text for k in ("http://", "https://", "url", "网页", "抓取", "fetch")):
            add("web_fetch")
        if any(k in text for k in ("运行", "执行", "命令", "shell", "bash", "python", "cmd")):
            add("exec")

        return caps

    async def _infer_required_caps_async(
        self, user_prompt: str, local_answer: str | None, tool_errors: list[str]
    ) -> list[str]:
        uc = self.universe_config
        vocab = [str(x).strip() for x in (uc.public_capability_vocab or []) if str(x).strip()]
        aliases = uc.public_capability_aliases or {}
        rule_caps = self._infer_required_caps(user_prompt, local_answer, tool_errors)
        caps = self._normalize_caps(rule_caps, vocab, aliases)

        if not caps and getattr(uc, "public_capability_llm_enabled", True):
            llm_caps = await self._llm_infer_caps(user_prompt, local_answer, tool_errors, vocab)
            caps = self._normalize_caps(llm_caps, vocab, aliases)

        default_kind = getattr(uc, "public_auto_delegate_kind", "evolvebot.agent")
        if default_kind and default_kind not in caps:
            caps.append(default_kind)
        return caps

    def _normalize_caps(self, caps: list[str], vocab: list[str], aliases: dict[str, list[str]]) -> list[str]:
        if not caps:
            return []
        vocab_set = {v.lower() for v in vocab} if vocab else set()
        alias_map: dict[str, str] = {}
        for canon, items in (aliases or {}).items():
            canon_key = str(canon).strip()
            if not canon_key:
                continue
            alias_map[canon_key.lower()] = canon_key
            for item in items or []:
                if isinstance(item, str) and item.strip():
                    alias_map[item.strip().lower()] = canon_key
        out: list[str] = []
        for cap in caps:
            if not cap:
                continue
            key = str(cap).strip()
            if not key:
                continue
            low = key.lower()
            canon = None
            if low in alias_map:
                canon = alias_map[low]
            elif not vocab_set or low in vocab_set:
                canon = key
            if not canon:
                continue
            if vocab_set and canon.lower() not in vocab_set:
                continue
            if canon not in out:
                out.append(canon)
        return out

    async def _llm_infer_caps(
        self,
        user_prompt: str,
        local_answer: str | None,
        tool_errors: list[str],
        vocab: list[str],
    ) -> list[str]:
        if not vocab:
            return []
        try:
            prompt = (
                "You are a capability classifier. Pick up to 3 items from the given list that best match\n"
                "the user's request. Only output a JSON array of strings from the list.\n\n"
                f"CAPABILITIES: {', '.join(vocab)}\n\n"
                f"USER_REQUEST:\n{user_prompt}\n\n"
                f"LOCAL_RESPONSE:\n{local_answer or ''}\n\n"
                f"TOOL_ERRORS:\n{'; '.join(tool_errors)}\n"
            )
            resp = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                model=self.model,
                temperature=0.0,
                max_tokens=128,
            )
            content = (resp.content or "").strip()
            data = json_repair.loads(content)
            if isinstance(data, list):
                return [str(x) for x in data if isinstance(x, str)]
        except Exception:
            return []
        return []

    async def _auto_pull_knowledge(self, provider_node: str, cap: str | None) -> None:
        uc = self.universe_config
        token = uc.public_registry_token or None
        try:
            from evolvebot.universe.public_client import knowledge_list, knowledge_get
            from evolvebot.universe.knowledge_store import save_pack

            limit = max(1, int(getattr(uc, "public_knowledge_auto_pull_limit", 1) or 1))
            tagged_only = bool(getattr(uc, "public_knowledge_auto_pull_tagged_only", True))
            inbox_dir = getattr(uc, "public_knowledge_inbox_dir", "") or None

            packs = await knowledge_list(
                registry_url=uc.public_registry_url,
                registry_token=token,
                owner_node=provider_node,
                tag=cap if tagged_only and cap else None,
                limit=limit,
            )
            if not packs and tagged_only:
                packs = await knowledge_list(
                    registry_url=uc.public_registry_url,
                    registry_token=token,
                    owner_node=provider_node,
                    tag=None,
                    limit=limit,
                )
            min_score = float(getattr(uc, "knowledge_learning_min_score", 0.0) or 0.0)
            if min_score > 0:
                packs = [p for p in packs if float(getattr(p, "score", 0) or 0) >= min_score]
            for meta in packs[:limit]:
                pack = await knowledge_get(
                    registry_url=uc.public_registry_url,
                    registry_token=token,
                    pack_id=meta.pack_id,
                )
                save_pack(pack, inbox_dir=inbox_dir)
        except Exception as e:
            logger.warning(f"auto knowledge pull failed: {e}")
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
