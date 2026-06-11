"""Mempalace — automatic long-term research memory for Hermes.

A Hermes MemoryProvider backed by the local Letta archival store
(pgvector in embedded Postgres + nomic embeddings on :8091). Built for
autoresearch loops: data retention is STRUCTURAL, not behavioral —

  sync_turn        every completed exchange is archived automatically
  on_pre_compress  spans about to be summarized away are archived in full
  on_delegation    subagent task/result pairs are archived
  prefetch         relevant memories are auto-injected before every turn
  tools            memory_search / memory_save remain for explicit access

Retrieval is two-leg: a local verbatim TAG INDEX (built asynchronously by
mempalace/diffusion/tagger.py — every tag a validated substring of the
archived item, see results/tag-recall.md) plus Letta embedding search.
Injected content is always verbatim archive text; the tag layer changes
WHICH items surface, never what they say.

Config: ~/.hermes/mempalace.json  {"base_url": "...:8283", "agent_id": "agent-...",
        "tagger": {"db": "...", "queue_dir": "...", "enabled": true}}
Stdlib only — runs inside the hermes-agent process.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.expanduser("~/.hermes/mempalace.json")
TAG_DB_DEFAULT = "~/.hermes/mempalace-tags.db"
TAG_QUEUE_DEFAULT = "~/.hermes/mempalace-tag-queue"
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": ("Semantic + exact-tag search across long-term research "
                    "memory (all sessions, all time). Use for anything "
                    "possibly discussed or discovered before."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer",
                      "description": "Max results (default 5, max 20)."},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": ("Optional exact tag filter (file paths, "
                                     "flags, error strings, identifiers).")},
        },
        "required": ["query"],
    },
}
SAVE_SCHEMA = {
    "name": "memory_save",
    "description": ("Save an important finding, fact, decision or citation "
                    "to long-term memory. Turns are archived automatically; "
                    "use this for distilled conclusions worth surfacing on "
                    "their own."),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string",
                        "description": "Self-contained finding to remember."},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "Optional topic tags."},
        },
        "required": ["content"],
    },
}


def _strip_think(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


class MempalaceProvider(MemoryProvider):
    """Automatic archival memory over the local Letta + pgvector stack."""

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}
        self._session_id = ""
        self._context = "primary"
        self._lock = threading.Lock()
        self._prefetch_cache: Dict[str, str] = {}   # query-hash -> block
        self._recent_hashes: List[str] = []          # dedup for sync_turn
        self._tag_db = ""
        self._tag_queue = ""

    # ------------------------------------------------------------- identity

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        try:
            cfg = json.load(open(CONFIG_PATH))
            return bool(cfg.get("base_url")) and bool(cfg.get("agent_id"))
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._cfg = json.load(open(CONFIG_PATH))
        self._session_id = session_id or "unknown"
        self._context = kwargs.get("agent_context", "primary") or "primary"
        tagger = self._cfg.get("tagger") or {}
        if tagger.get("enabled", True):
            self._tag_db = os.path.expanduser(tagger.get("db", TAG_DB_DEFAULT))
            self._tag_queue = os.path.expanduser(
                tagger.get("queue_dir", TAG_QUEUE_DEFAULT))
        logger.info("mempalace: initialized (session=%s context=%s tag_db=%s)",
                    self._session_id, self._context, self._tag_db or "off")

    def shutdown(self) -> None:
        pass  # writes are synchronous within the manager's background executor

    # ------------------------------------------------------------- backend

    def _http(self, method: str, path: str, *, body=None, query=None,
              timeout: float = 15.0):
        url = self._cfg["base_url"].rstrip("/") + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))

    def _insert(self, text: str, tags: List[str]) -> Optional[str]:
        """Archive to Letta; returns the created passage id (or None)."""
        out = self._http(
            "POST", f"/v1/agents/{self._cfg['agent_id']}/archival-memory",
            body={"text": text[:12000], "tags": tags})
        try:
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return out[0].get("id")
            if isinstance(out, dict):
                return out.get("id")
        except Exception:
            pass
        return None

    def _enqueue_tag(self, item_id: Optional[str], text: str) -> None:
        """Hand the archived item to the async tagger (atomic queue write)."""
        if not self._tag_queue or not item_id:
            return
        try:
            os.makedirs(self._tag_queue, exist_ok=True)
            tmp = os.path.join(self._tag_queue, f".{item_id}.tmp")
            with open(tmp, "w") as f:
                json.dump({"item_id": item_id, "text": text}, f)
            os.replace(tmp, os.path.join(self._tag_queue, f"{item_id}.json"))
        except Exception as e:
            logger.debug("mempalace: tag enqueue failed: %s", e)

    def _archive(self, text: str, tags: List[str]) -> None:
        self._enqueue_tag(self._insert(text, tags), text)

    # ------------------------------------------------------------ retrieval

    def _tag_search(self, query: str, top_k: int) -> List[str]:
        """FTS over the verbatim tag index; returns verbatim item texts."""
        if not self._tag_db or not os.path.exists(self._tag_db):
            return []
        toks = re.findall(r"[A-Za-z0-9_./-]{3,}", query)[:12]
        if not toks:
            return []
        match = " OR ".join('"' + t.replace('"', '') + '"' for t in toks)
        try:
            db = sqlite3.connect(f"file:{self._tag_db}?mode=ro", uri=True,
                                 timeout=1.0)
            try:
                rows = db.execute(
                    "SELECT item_id, count(*) AS n FROM tags_fts "
                    "WHERE tags_fts MATCH ? GROUP BY item_id "
                    "ORDER BY n DESC LIMIT ?", (match, top_k)).fetchall()
                out = []
                for item_id, _n in rows:
                    r = db.execute("SELECT text FROM items WHERE item_id=?",
                                   (item_id,)).fetchone()
                    if r and r[0]:
                        out.append(r[0].strip()[:600])
                return out
            finally:
                db.close()
        except Exception as e:
            logger.debug("mempalace: tag search failed (non-fatal): %s", e)
            return []

    def _tag_filter(self, tags: List[str], top_k: int) -> List[str]:
        """Exact tag lookup (memory_search tool's tags param)."""
        if not self._tag_db or not os.path.exists(self._tag_db) or not tags:
            return []
        try:
            db = sqlite3.connect(f"file:{self._tag_db}?mode=ro", uri=True,
                                 timeout=1.0)
            try:
                ph = ",".join("?" * len(tags))
                rows = db.execute(
                    f"SELECT t.item_id, count(*) AS n FROM tags t "
                    f"WHERE t.tag IN ({ph}) GROUP BY t.item_id "
                    f"ORDER BY n DESC LIMIT ?", (*tags, top_k)).fetchall()
                out = []
                for item_id, _n in rows:
                    r = db.execute("SELECT text FROM items WHERE item_id=?",
                                   (item_id,)).fetchone()
                    if r and r[0]:
                        out.append(r[0].strip()[:600])
                return out
            finally:
                db.close()
        except Exception as e:
            logger.debug("mempalace: tag filter failed (non-fatal): %s", e)
            return []

    def _embed_search(self, query: str, top_k: int = 5,
                      timeout: float = 8.0) -> List[str]:
        out = self._http(
            "GET",
            f"/v1/agents/{self._cfg['agent_id']}/archival-memory/search",
            query={"query": query[:1500], "top_k": top_k}, timeout=timeout)
        hits = out.get("results", out) if isinstance(out, dict) else out
        texts = []
        for h in hits or []:
            t = h.get("content", h.get("text", "")) if isinstance(h, dict) else str(h)
            if t:
                texts.append(t.strip()[:600])
        return texts

    def _search(self, query: str, top_k: int = 5,
                timeout: float = 8.0) -> List[str]:
        """Two-leg retrieval: exact-tag hits first, embedding hits after.
        Everything returned is verbatim archive text."""
        tag_hits = self._tag_search(query, max(2, top_k // 2 + 1))
        try:
            emb_hits = self._embed_search(query, top_k, timeout)
        except Exception as e:
            logger.debug("mempalace: embed search failed: %s", e)
            emb_hits = []
        # embedding's best hit keeps rank 1 (it wins on semantic probes);
        # tag hits follow so exact-identifier items always surface
        ordered = emb_hits[:1] + tag_hits + emb_hits[1:]
        merged, seen = [], set()
        for t in ordered:
            key = re.sub(r"\s+", " ", t[:160]).lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(t)
            if len(merged) >= top_k:
                break
        return merged

    # ------------------------------------------------------- auto-capture

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages=None) -> None:
        user = (user_content or "").strip()[:6000]
        asst = _strip_think(assistant_content)[:6000]
        if not user and not asst:
            return
        digest = hashlib.sha256(f"{user}\x00{asst}".encode()).hexdigest()
        with self._lock:
            if digest in self._recent_hashes:
                return
            self._recent_hashes = (self._recent_hashes + [digest])[-64:]
        sid = session_id or self._session_id
        stamp = time.strftime("%Y-%m-%d %H:%M")
        try:
            self._archive(
                f"[{stamp}] [session {sid}] [{self._context}]\n"
                f"USER: {user}\nASSISTANT: {asst}",
                tags=["auto-turn", f"sess:{sid}", f"ctx:{self._context}"])
        except Exception as e:
            logger.warning("mempalace: sync_turn archive failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Archive the full conversation before compression discards turns."""
        try:
            chunk, chunks, size = [], [], 0
            for m in messages or []:
                role = m.get("role", "?")
                content = m.get("content") or ""
                if not isinstance(content, str):
                    content = json.dumps(content)[:1000]
                if role == "system" or not content.strip():
                    continue
                line = f"{role.upper()}: {_strip_think(content)[:2000]}"
                chunk.append(line)
                size += len(line)
                if size > 8000:
                    chunks.append("\n".join(chunk))
                    chunk, size = [], 0
            if chunk:
                chunks.append("\n".join(chunk))
            stamp = time.strftime("%Y-%m-%d %H:%M")
            for i, c in enumerate(chunks[:40]):
                self._archive(
                    f"[{stamp}] [session {self._session_id}] "
                    f"[pre-compression span {i + 1}/{len(chunks)}]\n{c}",
                    tags=["compress-span", f"sess:{self._session_id}"])
            logger.info("mempalace: archived %d pre-compression chunks",
                        len(chunks))
            return ("Note: the full uncompressed conversation has been "
                    "archived to long-term memory and is retrievable via "
                    "memory_search.")
        except Exception as e:
            logger.warning("mempalace: on_pre_compress failed: %s", e)
            return ""

    def on_delegation(self, task: str, result: str, **kwargs) -> None:
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M")
            self._archive(
                f"[{stamp}] [session {self._session_id}] [subagent]\n"
                f"TASK: {(task or '')[:3000]}\n"
                f"RESULT: {_strip_think(result)[:8000]}",
                tags=["delegation", f"sess:{self._session_id}"])
        except Exception as e:
            logger.warning("mempalace: on_delegation archive failed: %s", e)

    # -------------------------------------------------------- auto-recall

    def system_prompt_block(self) -> str:
        return ("Long-term research memory (mempalace) is active: every "
                "exchange is archived automatically and survives across "
                "sessions. Relevant memories are auto-injected each turn; "
                "call memory_search for deeper or targeted recall (it also "
                "matches exact identifiers: file paths, flags, error strings).")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        q = (query or "").strip()
        if len(q) < 8:
            return ""
        key = hashlib.sha256(q.encode()).hexdigest()
        with self._lock:
            cached = self._prefetch_cache.pop(key, None)
        if cached is not None:
            return cached
        return self._recall_block(q)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        q = (query or "").strip()
        if len(q) < 8:
            return
        key = hashlib.sha256(q.encode()).hexdigest()

        def work():
            block = self._recall_block(q)
            with self._lock:
                self._prefetch_cache[key] = block
                while len(self._prefetch_cache) > 8:
                    self._prefetch_cache.pop(next(iter(self._prefetch_cache)))

        threading.Thread(target=work, daemon=True).start()

    def _recall_block(self, query: str) -> str:
        try:
            hits = self._search(query, top_k=4, timeout=6.0)
        except Exception as e:
            logger.debug("mempalace: prefetch search failed: %s", e)
            return ""
        if not hits:
            return ""
        body = "\n".join(f"- {h}" for h in hits)
        return ("[MEMPALACE RECALL — auto-retrieved from long-term memory; "
                f"may be relevant to the current turn]\n{body}")

    # -------------------------------------------------------------- tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, SAVE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        try:
            if tool_name == "memory_search":
                top_k = min(int(args.get("top_k", 5)), 20)
                tag_filter = [str(t) for t in (args.get("tags") or []) if t]
                hits = self._tag_filter(tag_filter, top_k) if tag_filter else []
                for h in self._search(str(args.get("query", "")), top_k=top_k):
                    if len(hits) >= top_k:
                        break
                    if h not in hits:
                        hits.append(h)
                return "\n".join(f"- {h}" for h in hits) or "no matching memories"
            if tool_name == "memory_save":
                tags = [str(t) for t in (args.get("tags") or [])]
                self._archive(str(args.get("content", "")),
                              tags=["explicit-save"] + tags)
                return "saved to long-term memory"
            return f"unknown memory tool: {tool_name}"
        except Exception as e:
            return f"memory error: {e}"
