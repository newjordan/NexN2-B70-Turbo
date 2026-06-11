#!/usr/bin/env python3
"""Mempalace tagger service — builds the verbatim tag index.

Watches a queue directory for archived-item files ({"item_id", "text"} JSON),
extracts structured slot tags with a pluggable backend, enforces the
verbatim-substring invariant, and writes a local SQLite index:

  items(item_id PRIMARY KEY, text, created)   -- verbatim copy of the archive item
  tags(item_id, slot, tag)                    -- every tag a validated substring
  tags_fts                                    -- FTS5 over tags for the query leg

The index NEVER stores model-generated prose: items.text is a byte copy of
what was archived; tags that are not literal substrings of it are dropped.

Backends:
  dgemma  llama-diffusion-cli (PR #24423 + local stdio patch) kept resident,
          DiffusionGemma Q4_K_M on CPU. ~54 s/item pinned to 12 cores.
  nexn2   the production llama-server on :8090 (enable_thinking off).
          ~5 s/item but serializes with live agent turns on the single slot.

Run via serve_diffusion.sh (sets pinning/nice/env). Stdlib only.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request

logger = logging.getLogger("tagger")

QUEUE_DIR = os.path.expanduser(os.environ.get(
    "TAG_QUEUE", "~/.hermes/mempalace-tag-queue"))
DB_PATH = os.path.expanduser(os.environ.get(
    "TAG_DB", "~/.hermes/mempalace-tags.db"))
BACKEND = os.environ.get("TAG_BACKEND", "dgemma")
STEPS = int(os.environ.get("TAG_STEPS", "8"))
THREADS = int(os.environ.get("TAG_THREADS", "12"))
BIN = os.environ.get(
    "TAG_BIN", "/home/frosty40/llama.cpp/build-diffusion-cpu/bin/llama-diffusion-cli")
MODEL = os.environ.get(
    "TAG_MODEL",
    "/home/frosty40/models/diffusion/diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
NEXN2_URL = os.environ.get("TAG_NEXN2_URL", "http://127.0.0.1:8090/v1/chat/completions")

SLOTS = ["ENTITIES", "METRICS", "FILES", "NUMBERS", "ERRORS", "DECISIONS", "TOPICS"]
MAX_TAGS_PER_SLOT = 8
MAX_TAG_LEN = 120
MAX_ITEM_CHARS = 6000

PROMPT_TMPL = """Extract index tags from the TEXT below.

Rules:
- Each tag MUST be an exact substring copied character-for-character from the TEXT.
- Do not paraphrase, reword, or invent anything.
- Output EXACTLY seven lines, one per slot, in this order, nothing else:

ENTITIES: <tools, models, systems, hardware named in the text>
METRICS: <measured values with their units, e.g. "28 t/s">
FILES: <file paths, scripts, URLs>
NUMBERS: <important bare numbers, versions, sizes>
ERRORS: <error messages or failure descriptions>
DECISIONS: <decisions or conclusions stated>
TOPICS: <single topic words that appear in the text>

Separate multiple tags on a line with " ; ". Write NONE if a slot has no match.
Give AT MOST 5 tags per line — pick only the most important ones. Be concise.

TEXT:
<<<
{text}
>>>
"""


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_and_validate(output: str, source: str):
    """Parse slot lines; keep only tags that are verbatim substrings of source."""
    norm_src = normalize(source)
    kept, dropped = [], 0
    per_slot: dict = {}
    for line in output.splitlines():
        m = re.match(r"^\s*([A-Z]+)\s*:\s*(.*)$", line)
        if not m or m.group(1) not in SLOTS:
            continue
        slot, body = m.group(1), m.group(2).strip()
        if not body or body.upper() == "NONE":
            continue
        for tag in body.split(";"):
            tag = tag.strip().strip('"').strip()
            if not tag or tag.upper() == "NONE" or len(tag) > MAX_TAG_LEN:
                continue
            if per_slot.get(slot, 0) >= MAX_TAGS_PER_SLOT:
                continue
            if normalize(tag) and normalize(tag) in norm_src:
                kept.append((slot, tag))
                per_slot[slot] = per_slot.get(slot, 0) + 1
            else:
                dropped += 1
    return kept, dropped


# ---------------------------------------------------------------- backends

class DiffusionBackend:
    """Drives the patched llama-diffusion-cli in LLAMA_DIFFUSION_STDIO mode."""

    def __init__(self):
        self.proc = None

    def _spawn(self):
        env = dict(os.environ,
                   LLAMA_DIFFUSION_STDIO="1", LLAMA_DIFFUSION_NO_THINK="1")
        cmd = [BIN, "-m", MODEL, "-cnv", "--temp", "0", "-t", str(THREADS),
               "-n", "256", "--diffusion-eb-max-steps", str(STEPS),
               "-c", "8192", "-ub", "8192", "-b", "8192"]
        logger.info("spawning: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        self._send("/reset")
        self._read_until("<<RESET>>", timeout=600)  # model load happens here
        logger.info("diffusion backend ready")

    def _send(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, sentinel: str, timeout: float):
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("diffusion backend died (EOF)")
            stripped = line.strip().lstrip("> ").strip()
            if stripped == sentinel:
                return lines
            lines.append(line.rstrip("\n"))
        raise TimeoutError(f"no {sentinel} within {timeout}s")

    def extract(self, prompt: str) -> str:
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()
        try:
            self._send("/reset")
            self._read_until("<<RESET>>", timeout=60)
            self._send(prompt.replace("\\", "\\\\").replace("\n", "\\n"))
            lines = self._read_until("<<DONE>>", timeout=900)
        except (RuntimeError, TimeoutError, BrokenPipeError) as e:
            logger.warning("backend failed (%s); respawning once", e)
            self.kill()
            self._spawn()
            self._send(prompt.replace("\\", "\\\\").replace("\n", "\\n"))
            lines = self._read_until("<<DONE>>", timeout=900)
        return "\n".join(l for l in lines if not l.startswith("total time:"))

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None


class NexN2Backend:
    def extract(self, prompt: str) -> str:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 900,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            NEXN2_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        return out["choices"][0]["message"]["content"]

    def kill(self):
        pass


# ------------------------------------------------------------------- index

def open_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS items(
            item_id TEXT PRIMARY KEY, text TEXT NOT NULL, created REAL);
        CREATE TABLE IF NOT EXISTS tags(
            item_id TEXT, slot TEXT, tag TEXT,
            UNIQUE(item_id, slot, tag));
        CREATE VIRTUAL TABLE IF NOT EXISTS tags_fts
            USING fts5(tag, item_id UNINDEXED, slot UNINDEXED);
    """)
    return db


def index_item(db, item_id: str, text: str, tags):
    db.execute("INSERT OR REPLACE INTO items VALUES (?,?,?)",
               (item_id, text, time.time()))
    for slot, tag in tags:
        cur = db.execute(
            "INSERT OR IGNORE INTO tags VALUES (?,?,?)", (item_id, slot, tag))
        if cur.rowcount:
            db.execute("INSERT INTO tags_fts VALUES (?,?,?)",
                       (tag, item_id, slot))
    db.commit()


# ----------------------------------------------------------------- service

def make_backend():
    return NexN2Backend() if BACKEND == "nexn2" else DiffusionBackend()


def process_file(db, backend, path: str) -> bool:
    with open(path) as f:
        req = json.load(f)
    item_id, text = str(req["item_id"]), str(req["text"])[:MAX_ITEM_CHARS]
    t0 = time.time()
    output = backend.extract(PROMPT_TMPL.format(text=text))
    kept, dropped = parse_and_validate(output, text)
    index_item(db, item_id, text, kept)
    logger.info("tagged %s: %d kept, %d dropped (invariant), %.1fs",
                item_id, len(kept), dropped, time.time() - t0)
    return True


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[tagger %(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(QUEUE_DIR, exist_ok=True)
    failed_dir = os.path.join(QUEUE_DIR, "failed")
    os.makedirs(failed_dir, exist_ok=True)
    db = open_db()
    backend = make_backend()
    logger.info("ready: backend=%s queue=%s db=%s", BACKEND, QUEUE_DIR, DB_PATH)
    while True:
        files = sorted(
            (e for e in os.scandir(QUEUE_DIR) if e.name.endswith(".json")),
            key=lambda e: e.stat().st_mtime)
        if not files:
            time.sleep(2.0)
            continue
        for entry in files:
            try:
                process_file(db, backend, entry.path)
                os.unlink(entry.path)
            except Exception as e:
                logger.warning("FAILED %s: %s", entry.name, e)
                os.replace(entry.path,
                           os.path.join(failed_dir, entry.name))


if __name__ == "__main__":
    main()
