#!/usr/bin/env python3
"""Measure NexN2 decode impact while a CPU tagger extraction runs.

NexN2's 30 Delta-Net layers decode on the CPU, so a CPU-resident diffusion
tagger contends directly with live decode. Gate: <= ~10% t/s impact pinned.

Usage:
  python3 contention_probe.py --label baseline
  python3 contention_probe.py --label dream-pinned \
      --load "taskset -c 0-15 python3 eval/memory/tag_bench.py --arm dream --steps 16 --threads 16 --note contention"

Appends to results/tag-contention.csv. The probe uses /completion with
n_predict fixed and ignore_eos so decoded-token count is constant.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(REPO, "results", "tag-contention.csv")
URL = "http://127.0.0.1:8090/completion"

PROBE_PROMPT = ("Explain, in detail, how speculative decoding interacts with "
                "mixture-of-experts routing in modern transformer inference stacks.")


def probe_once(n_predict: int = 192) -> float:
    body = json.dumps({"prompt": PROBE_PROMPT, "n_predict": n_predict,
                       "ignore_eos": True, "temperature": 0.7,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read())
    t = out.get("timings", {})
    if t.get("predicted_per_second"):
        return float(t["predicted_per_second"])
    raise RuntimeError(f"no timings in response: {list(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--load", default="", help="command to run during probes")
    ap.add_argument("--probes", type=int, default=3)
    ap.add_argument("--warmup-s", type=float, default=20.0,
                    help="let the load command get past model-load first")
    args = ap.parse_args()

    proc = None
    if args.load:
        proc = subprocess.Popen(shlex.split(args.load), cwd=REPO,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        time.sleep(args.warmup_s)

    rates = []
    try:
        for i in range(args.probes):
            if proc and proc.poll() is not None:
                print(f"WARNING: load command exited (rc={proc.returncode}) "
                      f"before probe {i + 1} — remaining probes are unloaded")
            r = probe_once()
            rates.append(r)
            print(f"probe {i + 1}/{args.probes}: {r:.2f} t/s", flush=True)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    avg = sum(rates) / len(rates)
    new_file = not os.path.exists(CSV_PATH)
    w = csv.writer(open(CSV_PATH, "a", newline=""))
    if new_file:
        w.writerow(["label", "probes", "avg_ts", "min_ts", "max_ts", "load_cmd"])
    w.writerow([args.label, len(rates), round(avg, 2), round(min(rates), 2),
                round(max(rates), 2), args.load])
    print(f"{args.label}: avg {avg:.2f} t/s over {len(rates)} probes")


if __name__ == "__main__":
    main()
