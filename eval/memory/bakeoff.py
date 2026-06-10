#!/usr/bin/env python3
"""Memory-substrate bakeoff: lean MemPalace vs Letta, same benchmark.

LongMemEval/LoCoMo-style: inject facts early in a long rolling session,
keep rolling with distractor research steps, query the facts late. Measures:

  recall          — fraction of injected facts retrievable verbatim at the end
  correctness     — substring match of the canonical value in the answer
  tokens_prefilled— total prompt tokens consumed across the session
  latency_per_step— mean wall seconds per roll step
  storage         — bytes on disk used by the substrate

Usage:
  bakeoff.py --system lean  --steps 40 --facts 8 --csv results/memory-bakeoff.csv
  bakeoff.py --system letta --steps 40 --facts 8 --csv results/memory-bakeoff.csv

Facts and distractors are deterministic (seeded) so both systems see the
identical session. Grading is substring-based here; LLM-judge coherence is a
separate pass under the ground-truth-results gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "mempalace", "lean"))

TOPICS = ["solar microgrids", "deep-sea mining", "mRNA storage", "rail freight",
          "lithium recycling", "fungal computing", "tidal energy", "vertical farms",
          "ion engines", "desalination", "carbon capture", "asteroid spectroscopy"]


def make_facts(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    facts = []
    for i in range(n):
        code = hashlib.sha256(f"fact/{seed}/{i}".encode()).hexdigest()[:8].upper()
        topic = TOPICS[i % len(TOPICS)]
        facts.append({
            "id": i,
            "topic": topic,
            "inject": (f"Field note #{i}: the verified reference id for the "
                       f"{topic} dataset is {code}. This id will matter later."),
            "query": (f"What is the verified reference id for the {topic} "
                      f"dataset mentioned earlier in our research?"),
            "answer": code,
        })
    rng.shuffle(facts)
    return facts


def make_distractor(step: int, seed: int = 7) -> str:
    rng = random.Random(f"{seed}/{step}")
    t1, t2 = rng.sample(TOPICS, 2)
    return (f"Research step {step}: compare {t1} with {t2} — summarize one "
            f"plausible synergy and one risk in under 120 words, and keep our "
            f"running outline current.")


# ---------------------------------------------------------------- adapters

class LeanSystem:
    name = "lean"

    def __init__(self, workdir: str):
        from controller import Controller
        self.c = Controller(os.path.join(workdir, "palace"))
        self.workdir = workdir

    def step(self, text: str) -> str:
        return self.c.step(text)

    def query(self, text: str) -> str:
        return self.c.step(text)

    def stats(self) -> dict:
        return {"tokens_prefilled": self.c.metrics["prompt_tokens"],
                "rooms": self.c.store.stats()["rooms"],
                "compactions": self.c.metrics["compactions"]}


class LettaSystem:
    name = "letta"

    def __init__(self, workdir: str, base_url: str = "http://127.0.0.1:8283"):
        from letta_client import Letta
        # letta steps fan out into several local-model calls; the SDK's
        # default ~60s request timeout is far too short
        self.client = Letta(base_url=base_url, timeout=900.0)
        agents = [a for a in self.client.agents.list()
                  if a.name == "mempalace-letta"]
        if not agents:
            raise RuntimeError("run mempalace/letta/setup_agent.py first")
        self.agent_id = agents[0].id
        self.tokens = 0

    def step(self, text: str) -> str:
        resp = self.client.agents.messages.create(
            agent_id=self.agent_id,
            messages=[{"role": "user", "content": text}])
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.tokens += getattr(usage, "prompt_tokens", 0) or 0
        out = []
        for m in resp.messages:
            if getattr(m, "message_type", "") == "assistant_message":
                out.append(str(getattr(m, "content", "")))
        return "\n".join(out)

    query = step

    def stats(self) -> dict:
        return {"tokens_prefilled": self.tokens}


# ------------------------------------------------------------------ runner

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["lean", "letta"], required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--facts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--csv", default="results/memory-bakeoff.csv")
    args = ap.parse_args()

    workdir = args.workdir or f"/home/frosty40/nx2-palace-run/bakeoff-{args.system}-s{args.seed}"
    os.makedirs(workdir, exist_ok=True)
    system = LeanSystem(workdir) if args.system == "lean" else LettaSystem(workdir)

    facts = make_facts(args.facts, args.seed)
    # inject facts at evenly spaced early steps (first half of the session)
    inject_at = {int((k + 1) * args.steps / (2 * len(facts))): f
                 for k, f in enumerate(facts)}

    t_steps = []
    for step in range(args.steps):
        text = (inject_at[step]["inject"] + "\n\n" + make_distractor(step, args.seed)
                if step in inject_at else make_distractor(step, args.seed))
        t0 = time.time()
        system.step(text)
        t_steps.append(time.time() - t0)
        print(f"[bakeoff/{args.system}] step {step + 1}/{args.steps} "
              f"({t_steps[-1]:.1f}s){' [FACT]' if step in inject_at else ''}",
              file=sys.stderr)

    hits = 0
    for f in facts:
        ans = system.query(f["query"])
        ok = f["answer"] in ans
        hits += ok
        print(f"[bakeoff/{args.system}] fact {f['id']} ({f['topic']}): "
              f"{'PASS' if ok else 'FAIL'} tail={ans[-90:]!r}", file=sys.stderr)

    du = sum(os.path.getsize(os.path.join(dp, fn))
             for dp, _, fns in os.walk(workdir) for fn in fns)
    row = {
        "system": args.system, "steps": args.steps, "facts": len(facts),
        "recall": round(hits / len(facts), 3),
        "latency_s_per_step": round(sum(t_steps) / len(t_steps), 1),
        "storage_bytes": du, "seed": args.seed,
        **system.stats(),
    }
    fields = ["system", "steps", "facts", "recall", "latency_s_per_step",
              "tokens_prefilled", "storage_bytes", "rooms", "compactions",
              "seed"]
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"[bakeoff/{args.system}] {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
