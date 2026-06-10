#!/usr/bin/env python3
"""Cache-smart multi-needle NIAH sweep against the local llama.cpp server.

RULER-style: ONE haystack of exact target token length carrying K unique
magic codes at K depths, each keyed by a distinct callsign in distinct
numbered filler. One expensive prefill (paid by the first probe), then K
cheap depth probes — the prompt cache rolls back to the haystack boundary
(verified: llama.cpp context checkpoints cover the recurrent Delta-Net
state; probe>=2 prompt_n ~= question length).

Per-probe CSV checkpointing; re-running with the same csv resumes (rows
already present are skipped).

NexN2 is a reasoning model: max_tokens must stay >= ~300 or the <think>
trace eats the budget and probes false-FAIL. temp 0, fixed seed,
substring-match grading.

Usage:
  niah_sweep.py --tokens 262144 --depths 10,50,90 --sample 0 \
      --rope-config native --kv-type f16 --csv results/niah-pareto.csv
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request

from tokenizers import Tokenizer

TOKENIZER = "/home/frosty40/models/nex-n2-mini-bf16/tokenizer.json"
URL = "http://127.0.0.1:8090/v1/chat/completions"
FIELDS = ["verdict", "ctx_tokens", "needle_depth_pct", "prefill_s",
          "decode_ts", "rope_config", "kv_type", "sample",
          "prompt_n", "cache_n"]
CALLSIGNS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliett", "kilo", "lima"]
# A probe whose prompt_n exceeds this fraction of the haystack re-prefilled —
# the cache rollback failed and the amortized cost model is void.
CACHE_MISS_FRAC = 0.5


def filler(i: int) -> str:
    return (f"Note {i}: inventory record {i} lists {i % 97 + 3} crates in "
            f"warehouse {i % 12}, checked on day {i % 365}.\n")


def magic_code(ctx: int, depth: int, sample: int) -> str:
    h = hashlib.sha256(f"{ctx}/{depth}/{sample}".encode()).hexdigest().upper()
    return f"{h[:4]}-{h[4:8]}-{h[8:12]}"


def needle(callsign: str, code: str) -> str:
    return (f"\nIMPORTANT: the secret magic code for operation "
            f"{callsign.upper()} is {code}. Remember it.\n")


def question(callsign: str) -> str:
    return (f"\n\nWhat is the secret magic code for operation "
            f"{callsign.upper()}? Reply with the code only.")


def build_haystack(tok, target_tokens, depths, sample):
    """Return (haystack_text, {depth: (callsign, code)})."""
    keyed = {d: (CALLSIGNS[k % len(CALLSIGNS)],
                 magic_code(target_tokens, d, sample))
             for k, d in enumerate(sorted(depths))}
    needles_len = sum(len(tok.encode(needle(c, m)).ids)
                      for c, m in keyed.values())
    overhead = needles_len + len(tok.encode(question("alpha")).ids) + 64

    base = sample * 10_000_000 + 1  # distinct filler content per sample
    per100 = len(tok.encode("".join(
        filler(i) for i in range(base, base + 100))).ids)
    n_lines = max(len(depths) + 1, (target_tokens - overhead) * 100 // per100)
    for _ in range(6):  # converge on the exact token target
        body = "".join(filler(i) for i in range(base, base + n_lines))
        got = len(tok.encode(body).ids) + overhead
        if abs(got - target_tokens) <= max(64, target_tokens // 200):
            break
        n_lines = max(len(depths) + 1,
                      int(n_lines * (target_tokens - overhead) / (got - overhead)))

    lines = [filler(i) for i in range(base, base + n_lines)]
    # insert deepest first so earlier indices stay valid
    for d in sorted(depths, reverse=True):
        c, m = keyed[d]
        lines.insert(min(len(lines), int(len(lines) * d / 100)), needle(c, m))
    return "".join(lines), keyed


def ask(prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": "nex-n2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(
                    r for r in f if not r.startswith("#")):
                done.add((row["ctx_tokens"], row["needle_depth_pct"],
                          row["rope_config"], row["kv_type"],
                          row.get("sample", "0")))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--depths", default="0,10,25,50,75,90,100")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--rope-config", default="native")
    ap.add_argument("--kv-type", default="f16")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    tok = Tokenizer.from_file(TOKENIZER)
    hay, keyed = build_haystack(tok, args.tokens, depths, args.sample)
    hay_tokens = len(tok.encode(hay).ids)

    # resume key uses the ACTUAL haystack length (what the csv stores), not
    # the target — the build is deterministic, so re-runs land on the same value
    done = load_done(args.csv)
    todo = [d for d in depths
            if (str(hay_tokens), str(d), args.rope_config, args.kv_type,
                str(args.sample)) not in done]
    if not todo:
        print(f"[sweep] {hay_tokens} tokens: all {len(depths)} depths "
              f"already in {args.csv}; skipping", file=sys.stderr)
        return 0
    print(f"[sweep] haystack ~{hay_tokens} tokens "
          f"(target {args.tokens}), probing depths {todo} "
          f"(sample {args.sample})", file=sys.stderr)

    new_file = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            f.flush()
        for k, d in enumerate(todo):
            callsign, code = keyed[d]
            obj = ask(hay + question(callsign), args.max_tokens, args.timeout)
            reply = obj["choices"][0]["message"]["content"]
            t = obj.get("timings", {})
            prompt_n = t.get("prompt_n", -1)
            row = {
                "verdict": "PASS" if code in reply else "FAIL",
                "ctx_tokens": hay_tokens,
                "needle_depth_pct": d,
                "prefill_s": round(t.get("prompt_ms", 0) / 1000, 1),
                "decode_ts": round(t.get("predicted_per_second", 0), 1),
                "rope_config": args.rope_config,
                "kv_type": args.kv_type,
                "sample": args.sample,
                "prompt_n": prompt_n,
                "cache_n": t.get("cache_n", -1),
            }
            w.writerow(row)
            f.flush()
            print(f"[sweep] depth {d}% ({callsign}): {row['verdict']} "
                  f"prompt_n={prompt_n} prefill_s={row['prefill_s']} "
                  f"tail={reply[-60:]!r}", file=sys.stderr)
            if k >= 1 and prompt_n > hay_tokens * CACHE_MISS_FRAC:
                print(f"[sweep] WARNING: probe {k + 1} re-prefilled "
                      f"{prompt_n}/{hay_tokens} tokens — cache rollback "
                      f"FAILED; amortized cost model void at this config",
                      file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
