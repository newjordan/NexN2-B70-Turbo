#!/usr/bin/env python3
"""MemPalace rolling controller — MemGPT-style tiered memory over the palace.

Hot window (token-budgeted)            | Palace (unbounded, on disk)
---------------------------------------+--------------------------------
system prompt + CORE memory (self-     | linked markdown rooms +
edited small state)                    | FTS5 BM25 + embeddings + links
retrieved rooms for the current step   |
recent turns (rolling)                 |

Loop per step: retrieve -> think (LLM) -> parse CORE-UPDATE / SAVE-ROOM
directives -> append turn -> compact oldest turns into rooms when the
window nears the budget -> periodic reflection into insight rooms.

The model self-edits memory through fenced directives in its answer:

  ```core-update
  {"hypotheses": ["..."], "outline": ["..."]}
  ```
  ```save-room
  {"title": "...", "importance": 0.8, "body": "... [[other room]] ..."}
  ```
Directives are stripped from the visible answer.
"""
from __future__ import annotations

import json
import os
import re
import time

from tokenizers import Tokenizer

from store import PalaceStore
from retriever import Retriever
from embed import EmbedClient
from llm import LLM

TOKENIZER = "/home/frosty40/models/nex-n2-mini-bf16/tokenizer.json"
DIRECTIVE_RE = re.compile(
    r"```(core-update|save-room)\s*\n(.*?)```", re.DOTALL)

SYSTEM = """\
You are a long-form autonomous research assistant with a memory palace.
Your context window is limited; durable knowledge lives in linked markdown
"rooms" retrieved for you each step under RETRIEVED ROOMS.

CORE MEMORY (below) is your small self-edited state: research question, live
hypotheses, key entities, output outline. Keep it current — emit a fenced
```core-update``` block with a JSON object to merge whenever it changes.

Save any finding worth keeping as a room — emit ```save-room``` with JSON
{"title", "importance" (0..1), "body"}; reference related rooms inline as
[[room title]] so the palace graph stays connected. Save eagerly: anything
not saved or in core memory may be forgotten when old turns are compacted.
"""


class Controller:
    def __init__(self, palace_root: str, *, hot_budget: int = 24000,
                 keep_recent: int = 6, reflect_every: int = 8,
                 embedder: EmbedClient | None = None, llm: LLM | None = None):
        self.store = PalaceStore(palace_root)
        self.embedder = embedder if embedder is not None else EmbedClient()
        self.retriever = Retriever(self.store, self.embedder)
        self.llm = llm or LLM()
        self.tok = Tokenizer.from_file(TOKENIZER)
        self.hot_budget = hot_budget
        self.keep_recent = keep_recent
        self.reflect_every = reflect_every

        self.root = os.path.abspath(palace_root)
        self.core_path = os.path.join(self.root, "core.json")
        self.turns_path = os.path.join(self.root, "session.jsonl")
        self.core = self._load_core()
        self.turns = self._load_turns()
        self.step_count = sum(1 for t in self.turns if t["role"] == "user")
        self.metrics = {"prompt_tokens": 0, "compactions": 0, "rooms_saved": 0}

    # ------------------------------------------------------------ main loop

    def step(self, user_input: str, *, max_tokens: int = 1200) -> str:
        retrieved = self.retriever.retrieve(user_input, k=6)
        messages = self._assemble(user_input, retrieved)
        out = self.llm.chat_raw(messages, max_tokens=max_tokens)
        self.metrics["prompt_tokens"] += out["timings"].get("prompt_n", 0)
        from llm import strip_think
        answer = strip_think(out["text"])
        answer = self._apply_directives(answer)

        self._append_turn("user", user_input)
        self._append_turn("assistant", answer)
        self.step_count += 1

        if self.step_count % self.reflect_every == 0:
            self.reflect()
        self._maybe_compact()
        return answer

    # ----------------------------------------------------------- assembly

    def _assemble(self, user_input: str, retrieved: list[dict]):
        rooms_txt = "\n\n".join(
            f"### [[{r['title']}]] (importance {r['importance']:.2f})\n"
            f"{r['body'][:4000]}" for r in retrieved) or "(palace is empty)"
        sys_msg = (f"{SYSTEM}\n## CORE MEMORY\n"
                   f"{json.dumps(self.core, indent=1)}\n\n"
                   f"## RETRIEVED ROOMS\n{rooms_txt}")
        msgs = [{"role": "system", "content": sys_msg}]
        msgs += [{"role": t["role"], "content": t["content"]}
                 for t in self.turns]
        msgs.append({"role": "user", "content": user_input})
        return msgs

    def _ntokens(self, text: str) -> int:
        return len(self.tok.encode(text).ids)

    def _hot_size(self) -> int:
        return (self._ntokens(SYSTEM) + self._ntokens(json.dumps(self.core))
                + sum(self._ntokens(t["content"]) for t in self.turns))

    # ---------------------------------------------------------- directives

    def _apply_directives(self, answer: str) -> str:
        def handle(m):
            kind, payload = m.group(1), m.group(2)
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                return ""  # malformed directive: drop silently, keep answer
            if kind == "core-update" and isinstance(obj, dict):
                self.core.update(obj)
                self._save_core()
            elif kind == "save-room" and isinstance(obj, dict) and \
                    obj.get("title") and obj.get("body"):
                self.save_room(obj["title"], str(obj["body"]),
                               importance=float(obj.get("importance", 0.5)))
            return ""
        return DIRECTIVE_RE.sub(handle, answer).strip()

    def save_room(self, title: str, body: str, *, importance: float = 0.5,
                  kind: str = "note") -> str:
        slug = self.store.write_room(title, body, importance=importance,
                                     kind=kind)
        try:
            self.store.set_embedding(
                slug, self.embedder.embed_docs([f"{title}\n{body}"])[0])
        except Exception as e:
            print(f"[controller] embedding skipped for {slug}: {e}")
        self.metrics["rooms_saved"] += 1
        return slug

    # ---------------------------------------------------------- compaction

    def _maybe_compact(self):
        while (self._hot_size() > self.hot_budget
               and len(self.turns) > self.keep_recent):
            evict = self.turns[:-self.keep_recent][:8]
            if not evict:
                break
            self._compact(evict)
            self.turns = self.turns[len(evict):]
            self._save_turns()
            self.metrics["compactions"] += 1

    def _compact(self, turns: list[dict]):
        convo = "\n\n".join(f"[{t['role']}] {t['content'][:2000]}"
                            for t in turns)
        prompt = (
            "Compact the following conversation span into 1-3 standalone "
            "memory notes that keep every durable fact, decision, citation "
            "and number. Reply ONLY with ```save-room``` blocks (JSON: "
            "title, importance 0..1, body; cross-reference related notes "
            f"with [[title]]).\n\n{convo}")
        out = self.llm.chat([{"role": "user", "content": prompt}],
                            max_tokens=1600)
        kept = DIRECTIVE_RE.findall(out)
        for kind, payload in kept:
            if kind != "save-room":
                continue
            try:
                obj = json.loads(payload)
                self.save_room(obj["title"], str(obj["body"]),
                               importance=float(obj.get("importance", 0.5)),
                               kind="compaction")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        if not kept:  # fall back: never lose the span silently
            self.save_room(f"Conversation span @step {self.step_count}",
                           convo[:6000], importance=0.3, kind="compaction")

    def reflect(self):
        recent = [self.store.read_room(s, touch=False)
                  for s in self.store.all_slugs()[-12:]]
        notes = "\n\n".join(f"[[{r['title']}]]\n{r['body'][:1500]}"
                            for r in recent if r)
        if not notes:
            return
        prompt = (
            "From these recent research notes, write 1-2 higher-level "
            "insight notes (patterns, contradictions, open questions). "
            "Reply ONLY with ```save-room``` blocks (JSON: title, "
            f"importance, body, [[links]]).\n\n{notes}")
        out = self.llm.chat([{"role": "user", "content": prompt}],
                            max_tokens=1200)
        for kind, payload in DIRECTIVE_RE.findall(out):
            if kind != "save-room":
                continue
            try:
                obj = json.loads(payload)
                self.save_room(obj["title"], str(obj["body"]),
                               importance=float(obj.get("importance", 0.6)),
                               kind="insight")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    # -------------------------------------------------------- persistence

    def _load_core(self) -> dict:
        if os.path.exists(self.core_path):
            return json.load(open(self.core_path))
        return {"question": "", "hypotheses": [], "entities": [],
                "outline": []}

    def _save_core(self):
        json.dump(self.core, open(self.core_path, "w"), indent=1)

    def _load_turns(self) -> list[dict]:
        if not os.path.exists(self.turns_path):
            return []
        return [json.loads(l) for l in open(self.turns_path) if l.strip()]

    def _save_turns(self):
        with open(self.turns_path, "w") as f:
            for t in self.turns:
                f.write(json.dumps(t) + "\n")

    def _append_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content,
                           "ts": time.time()})
        with open(self.turns_path, "a") as f:
            f.write(json.dumps(self.turns[-1]) + "\n")
