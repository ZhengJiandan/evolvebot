"""Learning manager for public universe knowledge packs.

Implements three learning modes:
1) Curiosity learning (periodic, small pulls)
2) Task-driven learning (on-demand pulls)
3) Review learning (periodic summaries)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from evolvebot.config.schema import Config
from evolvebot.universe.public_client import knowledge_list, knowledge_get, KnowledgePackMeta
from evolvebot.universe.knowledge_store import save_pack, get_inbox_path, load_pack_file
from evolvebot.providers.base import LLMProvider


@dataclass
class LearningState:
    last_curiosity_ts: float = 0.0
    last_review_ts: float = 0.0
    last_digest_ts: float = 0.0
    curiosity_daily_ts: float = 0.0
    curiosity_daily_count: int = 0
    learned_ids: set[str] = None
    learned_ids_order: list[str] = None

    def __post_init__(self) -> None:
        if self.learned_ids is None:
            self.learned_ids = set()
        if self.learned_ids_order is None:
            self.learned_ids_order = []


class LearningManager:
    def __init__(self, cfg: Config, *, provider: LLMProvider | None = None, model: str | None = None) -> None:
        self.cfg = cfg
        self.provider = provider
        self.model = model
        self.state = LearningState()
        self._lock = asyncio.Lock()
        self._review_buffer: list[str] = []
        self._bg_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._state_path = Path.home() / ".evolvebot" / "learning_state.json"
        self._applied_cache: set[str] = set()
        self._applied_cache_ts: float = 0.0
        self._load_state()

    def _load_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            data = __import__("json").loads(self._state_path.read_text())
            self.state.last_curiosity_ts = float(data.get("last_curiosity_ts", 0) or 0)
            self.state.last_review_ts = float(data.get("last_review_ts", 0) or 0)
            self.state.last_digest_ts = float(data.get("last_digest_ts", 0) or 0)
            self.state.curiosity_daily_ts = float(data.get("curiosity_daily_ts", 0) or 0)
            self.state.curiosity_daily_count = int(data.get("curiosity_daily_count", 0) or 0)
            learned = data.get("learned_ids", []) or []
            learned_order = data.get("learned_ids_order", []) or []
            if learned_order:
                self.state.learned_ids_order = [str(x) for x in learned_order if x]
                self.state.learned_ids = set(self.state.learned_ids_order)
            else:
                self.state.learned_ids = set(str(x) for x in learned if x)
                self.state.learned_ids_order = list(self.state.learned_ids)
        except Exception as e:
            logger.warning(f"learning state load failed: {e}")

    def _save_state(self) -> None:
        try:
            data = {
                "last_curiosity_ts": self.state.last_curiosity_ts,
                "last_review_ts": self.state.last_review_ts,
                "last_digest_ts": self.state.last_digest_ts,
                "curiosity_daily_ts": self.state.curiosity_daily_ts,
                "curiosity_daily_count": self.state.curiosity_daily_count,
                "learned_ids": list(self.state.learned_ids),
                "learned_ids_order": list(self.state.learned_ids_order)[:5000],
            }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning(f"learning state save failed: {e}")

    def _reset_daily_quota_if_needed(self) -> None:
        now = time.time()
        day = 24 * 3600
        if now - self.state.curiosity_daily_ts >= day:
            self.state.curiosity_daily_ts = now
            self.state.curiosity_daily_count = 0

    def _get_applied_pack_ids(self) -> set[str]:
        now = time.time()
        if now - self._applied_cache_ts < 300 and self._applied_cache:
            return self._applied_cache
        skills_dir = self.cfg.workspace_path / "skills"
        applied: set[str] = set()
        if skills_dir.exists():
            for path in skills_dir.rglob("SKILL.md"):
                try:
                    content = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                if not content.startswith("---"):
                    continue
                end = content.find("\n---", 3)
                if end == -1:
                    continue
                front = content[3:end].strip().splitlines()
                for line in front:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    if key.strip() != "pack_id":
                        continue
                    pid = value.strip().strip("\"'")
                    if pid:
                        applied.add(pid)
        self._applied_cache = applied
        self._applied_cache_ts = now
        return applied

    def _prune_applied_learned(self) -> None:
        applied = self._get_applied_pack_ids()
        if not applied:
            return
        if not (self.state.learned_ids & applied):
            return
        self.state.learned_ids.difference_update(applied)
        if self.state.learned_ids_order:
            self.state.learned_ids_order = [pid for pid in self.state.learned_ids_order if pid not in applied]

    def _mark_learned(self, pack_id: str) -> None:
        if not pack_id:
            return
        if pack_id in self._get_applied_pack_ids():
            return
        if pack_id in self.state.learned_ids:
            return
        self.state.learned_ids.add(pack_id)
        self.state.learned_ids_order.append(pack_id)
        limit = int(getattr(self.cfg.universe, "knowledge_learned_ids_limit", 2000) or 0)
        if limit > 0 and len(self.state.learned_ids_order) > limit:
            excess = len(self.state.learned_ids_order) - limit
            for _ in range(excess):
                old = self.state.learned_ids_order.pop(0)
                self.state.learned_ids.discard(old)

    def record_task_summary(self, *, prompt: str, answer: str | None, tool_errors: list[str] | None = None) -> None:
        if not answer:
            return
        summary = answer.strip().replace("\n", " ")
        if len(summary) > 200:
            summary = summary[:200] + "..."
        prefix = prompt.strip().replace("\n", " ")
        if len(prefix) > 120:
            prefix = prefix[:120] + "..."
        line = f"Q: {prefix} | A: {summary}"
        if tool_errors:
            line += f" (errors: {', '.join(tool_errors[:2])})"
        self._review_buffer.append(line)
        if len(self._review_buffer) > 200:
            self._review_buffer = self._review_buffer[-200:]

    async def maybe_curiosity_learn(self) -> None:
        uc = self.cfg.universe
        if not getattr(uc, "public_enabled", False):
            return
        if not getattr(uc, "public_registry_url", ""):
            return
        if not getattr(uc, "knowledge_curiosity_enabled", False):
            return
        interval = max(60, int(getattr(uc, "knowledge_curiosity_interval_s", 86400) or 86400))
        now = time.time()
        if now - self.state.last_curiosity_ts < interval:
            return

        async with self._lock:
            self._prune_applied_learned()
            self._reset_daily_quota_if_needed()
            daily_limit = max(0, int(getattr(uc, "knowledge_curiosity_daily_limit", 2) or 0))
            if daily_limit <= 0 or self.state.curiosity_daily_count >= daily_limit:
                return
            tags = list(getattr(uc, "knowledge_curiosity_tags", []) or [])
            token = uc.public_registry_token or None
            inbox_dir = getattr(uc, "public_knowledge_inbox_dir", "") or None
            limit = max(1, daily_limit - self.state.curiosity_daily_count)

            try:
                packs = await self._list_candidates(tags=tags, limit=limit * 3)
                picks = self._pick_new(packs, limit=limit)
                if not picks:
                    self.state.last_curiosity_ts = now
                    self._save_state()
                    return
                for meta in picks:
                    pack = await knowledge_get(
                        registry_url=uc.public_registry_url,
                        registry_token=token,
                        pack_id=meta.pack_id,
                    )
                    save_pack(pack, inbox_dir=inbox_dir)
                    self._mark_learned(meta.pack_id)
                    self.state.curiosity_daily_count += 1
                self.state.last_curiosity_ts = now
                self._save_state()
                logger.info(f"curiosity learning pulled {len(picks)} packs")
            except Exception as e:
                logger.warning(f"curiosity learning failed: {e}")

    async def maybe_task_driven_learn(
        self,
        *,
        task_prompt: str,
        tool_errors: list[str] | None = None,
    ) -> None:
        uc = self.cfg.universe
        if not getattr(uc, "public_enabled", False):
            return
        if not getattr(uc, "public_registry_url", ""):
            return
        if not getattr(uc, "knowledge_task_driven_enabled", False):
            return
        max_per_task = max(0, int(getattr(uc, "knowledge_task_driven_max_per_task", 1) or 0))
        if max_per_task <= 0:
            return
        tagged_only = bool(getattr(uc, "knowledge_task_driven_tagged_only", True))

        async with self._lock:
            self._prune_applied_learned()
            tags = self._extract_tags(task_prompt, tool_errors or [])
            if tagged_only and not tags:
                return
            try:
                packs = await self._list_candidates(tags=tags if tagged_only else [], limit=max_per_task * 5)
                picks = self._pick_new(packs, limit=max_per_task)
                if not picks:
                    return
                token = uc.public_registry_token or None
                inbox_dir = getattr(uc, "public_knowledge_inbox_dir", "") or None
                for meta in picks:
                    pack = await knowledge_get(
                        registry_url=uc.public_registry_url,
                        registry_token=token,
                        pack_id=meta.pack_id,
                    )
                    save_pack(pack, inbox_dir=inbox_dir)
                    self._mark_learned(meta.pack_id)
                self._save_state()
                logger.info(f"task-driven learning pulled {len(picks)} packs")
            except Exception as e:
                logger.warning(f"task-driven learning failed: {e}")

    async def maybe_review_learn(self) -> None:
        uc = self.cfg.universe
        if not getattr(uc, "knowledge_review_enabled", False):
            return
        interval = max(60, int(getattr(uc, "knowledge_review_interval_s", 86400) or 86400))
        now = time.time()
        if now - self.state.last_review_ts < interval:
            return
        summaries = [s for s in self._review_buffer if s]
        if not summaries:
            return
        min_tasks = max(1, int(getattr(uc, "knowledge_review_min_tasks", 10) or 1))
        max_tasks = max(min_tasks, int(getattr(uc, "knowledge_review_max_tasks", 20) or min_tasks))
        if len(summaries) < min_tasks:
            return
        summaries = summaries[-max_tasks:]

        async with self._lock:
            try:
                pack, advance = await self._build_review_pack(summaries)
                if not pack:
                    if advance:
                        self.state.last_review_ts = now
                        self._review_buffer = []
                        self._save_state()
                    return
                # Save locally; publish loop can upload if enabled.
                outbox = getattr(uc, "public_knowledge_publish_dir", "") or ""
                inbox_dir = getattr(uc, "public_knowledge_inbox_dir", "") or None
                publish = bool(getattr(uc, "knowledge_review_publish", False))
                target_dir = outbox if (publish and outbox) else str(get_inbox_path(inbox_dir))
                target_path = Path(target_dir).expanduser()
                target_path.mkdir(parents=True, exist_ok=True)
                fname = f"review_{int(now)}.json"
                (target_path / fname).write_text(__import__("json").dumps(pack, ensure_ascii=False, indent=2))
                logger.info("review learning wrote pack to {}", target_path)
                self.state.last_review_ts = now
                self._review_buffer = []
                self._save_state()
            except Exception as e:
                logger.warning(f"review learning failed: {e}")

    async def _build_review_pack(self, summaries: list[str]) -> tuple[dict[str, Any] | None, bool]:
        uc = self.cfg.universe
        if not getattr(uc, "knowledge_review_llm_enabled", True):
            logger.info("review learning skipped: LLM disabled")
            return None, False
        if not self.provider or not self.model:
            logger.info("review learning skipped: no LLM provider/model")
            return None, False
        try:
            prompt = (
                "You are a strict knowledge curator. Read the Q&A summaries and decide whether they contain\n"
                "reusable, high-value knowledge that other agents would benefit from learning.\n\n"
                "Return ONLY a JSON object with these keys:\n"
                "- publish (boolean)\n"
                "- score (0-100)\n"
                "- title (string)\n"
                "- summary (string)\n"
                "- tags (array of strings)\n"
                "- content_markdown (string)\n"
                "- reason (string)\n\n"
                "Publishing rules:\n"
                "- If the content is trivial, repetitive, or only specific to one-time context, set publish=false.\n"
                "- Only publish if the knowledge is reusable, actionable, and helpful to other agents.\n"
                "- content_markdown should be a compact, refined knowledge pack with these sections:\n"
                "  1) Key Learnings\n"
                "  2) Reusable Patterns\n"
                "  3) When To Use\n"
                "  4) Common Pitfalls\n"
                "  5) Open Questions\n\n"
                "Summaries:\n"
                + "\n".join(f"- {s}" for s in summaries)
            )
            resp = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                model=self.model,
                temperature=float(getattr(uc, "knowledge_review_llm_temperature", 0.2) or 0.2),
                max_tokens=int(getattr(uc, "knowledge_review_llm_max_tokens", 512) or 512),
            )
            raw = (resp.content or "").strip()
            if not raw:
                return None, False
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                try:
                    import json_repair

                    data = json_repair.loads(raw)
                except Exception:
                    data = None
            if not isinstance(data, dict):
                return None, False

            publish = bool(data.get("publish", False))
            reason = str(data.get("reason", "") or "")
            score = int(float(data.get("score", 0) or 0))
            min_score = int(getattr(uc, "knowledge_review_llm_gate_min_score", 70) or 70)
            gate_enabled = bool(getattr(uc, "knowledge_review_llm_gate_enabled", True))

            if gate_enabled and (not publish or score < min_score):
                msg = reason or f"score {score} < {min_score}"
                logger.info(f"review learning skipped: {msg}")
                return None, True

            content = str(data.get("content_markdown", "") or data.get("content", "") or "").strip()
            min_chars = max(0, int(getattr(uc, "knowledge_review_min_content_chars", 300) or 0))
            if min_chars and len(content) < min_chars:
                logger.info("review learning skipped: content too short")
                return None, True

            title = str(data.get("title", "") or "Daily Review").strip() or "Daily Review"
            summary = str(data.get("summary", "") or "").strip()
            tags = data.get("tags", []) or []
            if not isinstance(tags, list):
                tags = []
            clean_tags: list[str] = []
            for t in tags:
                if isinstance(t, str):
                    t = t.strip()
                    if t:
                        clean_tags.append(t[:32])
            clean_tags = clean_tags[:20]

            return {
                "name": title[:120],
                "kind": "review",
                "summary": summary[:500],
                "content": content,
                "tags": clean_tags or ["review"],
                "version": "1.0",
            }, True
        except Exception as e:
            logger.warning(f"review llm summary failed: {e}")
        return None, False

    async def build_daily_learned_digest(self, *, max_items: int = 12) -> str | None:
        """Summarize learned knowledge packs since the last digest."""
        uc = self.cfg.universe
        inbox = get_inbox_path(getattr(uc, "public_knowledge_inbox_dir", "") or None)
        manifest = inbox / "manifest.json"
        if not manifest.exists():
            return None
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            return None
        packs = data.get("packs", []) if isinstance(data, dict) else []
        if not isinstance(packs, list):
            return None

        now = time.time()
        since_ts = self.state.last_digest_ts or (now - 24 * 3600)
        recent = []
        for entry in packs:
            if not isinstance(entry, dict):
                continue
            saved_at = float(entry.get("savedAt", 0) or 0)
            if saved_at <= since_ts:
                continue
            fname = entry.get("file")
            if not fname:
                continue
            path = inbox / fname
            if not path.exists():
                continue
            try:
                pack = load_pack_file(path)
            except Exception:
                continue
            recent.append((saved_at, pack))

        if not recent:
            self.state.last_digest_ts = now
            self._save_state()
            return None

        recent.sort(key=lambda x: x[0], reverse=True)
        items = [p for _, p in recent[: max(1, max_items)]]

        # Build LLM prompt using pack summaries.
        lines = []
        for p in items:
            tags = ",".join(p.tags or [])
            summary = (p.summary or "").strip()
            if len(summary) > 240:
                summary = summary[:240] + "..."
            lines.append(f"- {p.name} | {summary} | tags: {tags}")

        if not self.provider or not self.model or not getattr(uc, "knowledge_review_llm_enabled", True):
            digest = "今日学习知识包摘要：\n" + "\n".join(lines)
            self.state.last_digest_ts = now
            self._save_state()
            return digest

        try:
            prompt = (
                "你是知识整理者。请根据以下知识包列表，生成一份简洁但有价值的“今日学习摘要”。\n"
                "要求：\n"
                "1) 概述新增的能力/知识方向\n"
                "2) 提炼最值得复用的知识点（条目化）\n"
                "3) 给出适用场景或建议\n"
                "输出为中文 Markdown，不超过 400 字。\n\n"
                "知识包列表：\n"
                + "\n".join(lines)
            )
            resp = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                model=self.model,
                temperature=float(getattr(uc, "knowledge_review_llm_temperature", 0.2) or 0.2),
                max_tokens=int(getattr(uc, "knowledge_review_llm_max_tokens", 512) or 512),
            )
            digest = (resp.content or "").strip()
            if digest:
                self.state.last_digest_ts = now
                self._save_state()
                return digest
        except Exception as e:
            logger.warning(f"daily digest llm failed: {e}")

        self.state.last_digest_ts = now
        self._save_state()
        return "今日学习知识包摘要：\n" + "\n".join(lines)

    async def run_forever(self, interval_s: int = 60) -> None:
        """Background loop for curiosity and review learning."""
        self._stop_event.clear()
        try:
            while not self._stop_event.is_set():
                await self.maybe_curiosity_learn()
                await self.maybe_review_learn()
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

    def start_background(self, interval_s: int = 60) -> None:
        if self._bg_task and not self._bg_task.done():
            return
        self._bg_task = asyncio.create_task(self.run_forever(interval_s=interval_s))

    async def stop_background(self) -> None:
        self._stop_event.set()
        if self._bg_task:
            self._bg_task.cancel()
            with __import__("contextlib").suppress(Exception):
                await self._bg_task

    async def _list_candidates(self, *, tags: list[str], limit: int) -> list[KnowledgePackMeta]:
        uc = self.cfg.universe
        token = uc.public_registry_token or None
        results: list[KnowledgePackMeta] = []
        if tags:
            for tag in tags:
                packs = await knowledge_list(
                    registry_url=uc.public_registry_url,
                    registry_token=token,
                    tag=tag,
                    limit=limit,
                )
                results.extend(packs)
        else:
            packs = await knowledge_list(
                registry_url=uc.public_registry_url,
                registry_token=token,
                limit=limit,
            )
            results.extend(packs)
        # Deduplicate by pack_id
        uniq: dict[str, KnowledgePackMeta] = {}
        for p in results:
            if p.pack_id and p.pack_id not in uniq:
                uniq[p.pack_id] = p
        packs = list(uniq.values())
        min_score = float(getattr(self.cfg.universe, "knowledge_learning_min_score", 0.0) or 0.0)
        if min_score > 0:
            packs = [p for p in packs if float(getattr(p, "score", 0) or 0) >= min_score]
        return packs

    def _pick_new(self, packs: list[KnowledgePackMeta], *, limit: int) -> list[KnowledgePackMeta]:
        if limit <= 0:
            return []
        fresh = [p for p in packs if p.pack_id and p.pack_id not in self.state.learned_ids]
        # Prefer recent packs
        fresh.sort(key=lambda p: float(p.updated_ts or p.created_ts or 0), reverse=True)
        return fresh[:limit]

    def _extract_tags(self, prompt: str, tool_errors: list[str], tools_used: list[str] | None = None) -> list[str]:
        text = f"{prompt}\n{';'.join(tool_errors or [])}".lower()
        tags: list[str] = []
        vocab = [str(x).strip() for x in (getattr(self.cfg.universe, "public_capability_vocab", []) or []) if str(x).strip()]
        vocab_map = {v.lower(): v for v in vocab}
        aliases = getattr(self.cfg.universe, "public_capability_aliases", {}) or {}

        alias_map: dict[str, str] = {}
        for canon, items in aliases.items():
            canon_key = str(canon).strip()
            if not canon_key:
                continue
            alias_map[canon_key.lower()] = canon_key
            for item in items or []:
                if isinstance(item, str) and item.strip():
                    alias_map[item.strip().lower()] = canon_key

        def add(tag: str) -> None:
            if tag and tag not in tags:
                tags.append(tag)

        for cap in vocab:
            if cap and cap.lower() in text:
                add(cap)
        for alias, canon in alias_map.items():
            if alias and alias in text:
                add(canon)
        for tool in tools_used or []:
            key = str(tool).strip().lower()
            if not key:
                continue
            if key in alias_map:
                add(alias_map[key])
            elif not vocab_map or key in vocab_map:
                add(vocab_map.get(key, tool))

        return tags[:10]
