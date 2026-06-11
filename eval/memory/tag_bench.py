#!/usr/bin/env python3
"""Phase-0 feasibility benchmark for the compression-tagging layer.

Measures, per backend ("arm"), the cost and fidelity of slot-infill tag
extraction over real campaign documents formatted exactly like mempalace
auto-turn archive items.

Arms:
  dgemma  llama-diffusion-cli (PR #24423 CPU build) + DiffusionGemma Q4_K_M
  dream   llama-diffusion-cli + Dream-7B Q4_K_M (timestep schedule)
  nexn2   the live llama-server on :8090 (autoregressive control)

Invariant under test: every emitted tag must be a verbatim substring of the
source item (whitespace-collapsed, case-normalized). Tags that fail are
dropped; the pass-rate is the headline fidelity metric.

Usage:
  python3 tag_bench.py --arm nexn2
  python3 tag_bench.py --arm dream  --steps 16 --threads 16
  python3 tag_bench.py --arm dgemma --steps 16 --threads 16 --taskset 0-15

CSV rows are appended to results/tag-recall.csv (header auto-written).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_PATH = os.path.join(REPO, "eval", "memory", "tag_corpus.json")
CSV_PATH = os.path.join(REPO, "results", "tag-recall.csv")

BIN_PR = "/home/frosty40/llama.cpp/build-diffusion-cpu/bin/llama-diffusion-cli"
DGEMMA = "/home/frosty40/models/diffusion/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
DREAM = "/home/frosty40/models/diffusion/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf"
NEXN2_URL = "http://127.0.0.1:8090/v1/chat/completions"

SLOTS = ["ENTITIES", "METRICS", "FILES", "NUMBERS", "ERRORS", "DECISIONS", "TOPICS"]

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

# Corpus: verbatim excerpts from real campaign artifacts, wrapped in the exact
# shape sync_turn archives ("[stamp] [session ...] [ctx]\nUSER: ...\nASSISTANT: ...").
CORPUS_SOURCES = [
    ("summary-upstream", "results/upstream-pr/SUMMARY.md",
     "what did the SYCL patch validation conclude?"),
    ("niah-pareto", "results/niah-pareto.md",
     "what were the long-context sweep results?"),
    ("memory-bakeoff", "results/memory-bakeoff.md",
     "how did the memory bakeoff go?"),
    ("letta-readme", "mempalace/letta/README.md",
     "how do I bring the letta stack up?"),
    ("serve-sh", "eval/niah/serve.sh",
     "show me the server relauncher script"),
    ("methodology", "docs/methodology.md",
     "what is the measurement methodology?"),
]


def build_corpus() -> list:
    if os.path.exists(CORPUS_PATH):
        return json.load(open(CORPUS_PATH))
    items = []
    for item_id, rel, question in CORPUS_SOURCES:
        text = open(os.path.join(REPO, rel)).read()[:2400]
        items.append({
            "id": item_id,
            "text": (f"[2026-06-10 12:00] [session bench] [primary]\n"
                     f"USER: {question}\nASSISTANT: {text}"),
        })
    json.dump(items, open(CORPUS_PATH, "w"), indent=1)
    return items


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_and_validate(output: str, source: str):
    """-> (tags_total, tags_valid, slots_found, valid_tags)"""
    norm_src = normalize(source)
    total = valid = slots_found = 0
    valid_tags = []
    for line in output.splitlines():
        m = re.match(r"^\s*([A-Z]+)\s*:\s*(.*)$", line)
        if not m or m.group(1) not in SLOTS:
            continue
        body = m.group(2).strip()
        if not body or body.upper() == "NONE":
            continue
        slots_found += 1
        for tag in body.split(";"):
            tag = tag.strip().strip('"').strip()
            if not tag or tag.upper() == "NONE" or len(tag) > 120:
                continue
            total += 1
            if normalize(tag) and normalize(tag) in norm_src:
                valid += 1
                valid_tags.append((m.group(1), tag))
    return total, valid, slots_found, valid_tags


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def run_diffusion(binpath, model, prompt, steps, threads, taskset, dream_mode):
    cmd = []
    if taskset:
        cmd += ["taskset", "-c", taskset]
    cmd += [binpath, "-m", model, "-p", prompt, "--temp", "0",
            "-t", str(threads)]
    if dream_mode:
        # fixed-length pass: ubatch must hold prompt + output
        est = int(len(prompt) / 2.8) + 384
        cmd += ["--diffusion-steps", str(steps), "--diffusion-eps", "0.001",
                "-ub", str(est), "-b", str(est), "-c", str(est)]
    else:
        cmd += ["-n", "256", "--diffusion-eb-max-steps", str(steps),
                "-c", "8192", "-ub", "8192", "-b", "8192"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    out = r.stdout
    decode_ms = ""
    m = re.search(r"total time:\s*([\d.]+)ms", out + r.stderr)
    if m:
        decode_ms = m.group(1)
    # response = stdout minus timing/log lines
    resp = "\n".join(l for l in out.splitlines()
                     if not l.startswith("total time:"))
    return resp, wall, decode_ms, r.returncode


def run_nexn2(prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 900,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(NEXN2_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.loads(resp.read())
    wall = time.time() - t0
    return strip_think(out["choices"][0]["message"]["content"]), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["dgemma", "dream", "nexn2"])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--taskset", default="")
    ap.add_argument("--items", default="", help="comma-sep item ids (default all)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    corpus = build_corpus()
    if args.items:
        keep = set(args.items.split(","))
        corpus = [c for c in corpus if c["id"] in keep]

    new_file = not os.path.exists(CSV_PATH)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    w = csv.writer(open(CSV_PATH, "a", newline=""))
    if new_file:
        w.writerow(["arm", "steps", "threads", "taskset", "item", "item_chars",
                    "wall_s", "decode_ms", "tags_total", "tags_valid",
                    "pass_rate", "slots_found", "note"])

    for item in corpus:
        prompt = PROMPT_TMPL.format(text=item["text"])
        print(f"[{args.arm} steps={args.steps}] {item['id']} "
              f"({len(item['text'])} chars)...", flush=True)
        if args.arm == "nexn2":
            resp, wall = run_nexn2(prompt)
            decode_ms, rc = "", 0
        else:
            resp, wall, decode_ms, rc = run_diffusion(
                BIN_PR, DGEMMA if args.arm == "dgemma" else DREAM,
                prompt, args.steps, args.threads, args.taskset,
                dream_mode=(args.arm == "dream"))
        total, valid, slots, vtags = parse_and_validate(resp, item["text"])
        rate = round(valid / total, 3) if total else 0.0
        print(f"  rc={rc} wall={wall:.1f}s decode={decode_ms}ms "
              f"tags={valid}/{total} ({rate:.0%}) slots={slots}", flush=True)
        for s, t in vtags[:6]:
            print(f"    {s}: {t}")
        w.writerow([args.arm, args.steps, args.threads, args.taskset,
                    item["id"], len(item["text"]), round(wall, 2), decode_ms,
                    total, valid, rate, slots, args.note])
        # raw output for post-mortem
        raw_dir = os.path.join(REPO, "results", "tag-bench-raw")
        os.makedirs(raw_dir, exist_ok=True)
        with open(os.path.join(raw_dir,
                  f"{args.arm}-s{args.steps}-{item['id']}.txt"), "w") as f:
            f.write(resp)


if __name__ == "__main__":
    main()
