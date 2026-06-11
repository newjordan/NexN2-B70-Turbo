#!/usr/bin/env python3
"""Strict end-to-end test of automatic memory + tag layer (production stack).

Session A casually mentions a fake precise flag via sync_turn (zero memory
instructions). After the async tagger indexes it, a FRESH provider in a new
session must surface the exact string via plain prefetch.
"""
import os
import sqlite3
import sys
import time
import types

agent_pkg = types.ModuleType("agent")
mp = types.ModuleType("agent.memory_provider")


class MemoryProvider:
    pass


mp.MemoryProvider = MemoryProvider
agent_pkg.memory_provider = mp
sys.modules["agent"] = agent_pkg
sys.modules["agent.memory_provider"] = mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "mempalace", "hermes-plugin"))
import mempalace as plug  # noqa: E402

FLAG = "--reshard-fanout=23"

a = plug.MempalaceProvider()
a.initialize("strict-test-A")
a.sync_turn(
    f"by the way the lattice resharder kept crashing until i set {FLAG}, "
    "just mentioning it in passing",
    f"Noted — the resharder is stable with {FLAG}. Anything else?")
print("archived via sync_turn; waiting for tagger...", flush=True)

db_path = os.path.expanduser("~/.hermes/mempalace-tags.db")
deadline = time.time() + 240
tagged = False
while time.time() < deadline:
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        n = db.execute("SELECT count(*) FROM tags WHERE tag LIKE "
                       "'%reshard-fanout%'").fetchone()[0]
        db.close()
        if n:
            tagged = True
            break
    except Exception:
        pass
    time.sleep(10)
print(f"tagged: {tagged}", flush=True)

b = plug.MempalaceProvider()
b.initialize("strict-test-B")
block = b.prefetch("what fanout did we use to stop the lattice resharder crashing?")
ok = FLAG in block
print(f"fresh-session recall contains exact flag: {ok}", flush=True)
print(block[:400] if block else "(empty block)", flush=True)
sys.exit(0 if (ok and tagged) else 1)
