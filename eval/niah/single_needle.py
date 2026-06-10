#!/usr/bin/env python3
"""Single-needle NIAH probe against the local llama.cpp server.

Reproduces the original results/niah.csv methodology: one unique magic code
buried at a given depth inside distinct numbered filler, graded by substring
match on the reply. Prints a CSV row compatible with the niah.csv schema
(verdict,ctx_tokens,needle_depth_pct,prefill_s) plus decode t/s.

NexN2 is a reasoning model: max_tokens must stay >= ~300 or the <think>
trace eats the budget and the probe false-FAILs.
"""
import argparse
import json
import sys
import urllib.request

from tokenizers import Tokenizer

TOKENIZER = "/home/frosty40/models/nex-n2-mini-bf16/tokenizer.json"
URL = "http://127.0.0.1:8090/v1/chat/completions"
MAGIC = "7C4F-9B2E-31AD"

QUESTION = (
    "\n\nWhat is the secret magic code mentioned in the notes above? "
    "Reply with the code only."
)
NEEDLE = f"\nIMPORTANT: the secret magic code is {MAGIC}. Remember it.\n"


def filler(i: int) -> str:
    return (f"Note {i}: inventory record {i} lists {i % 97 + 3} crates in "
            f"warehouse {i % 12}, checked on day {i % 365}.\n")


def build_prompt(tok: Tokenizer, target_tokens: int, depth_pct: int) -> str:
    probe = filler(123456)
    per_line = len(tok.encode(probe).ids)
    overhead = len(tok.encode(NEEDLE + QUESTION).ids) + 64  # chat template slack
    n_lines = max(1, (target_tokens - overhead) // per_line)
    for _ in range(6):  # early line numbers are shorter than the estimate
        body = "".join(filler(i) for i in range(1, n_lines + 1))
        got = len(tok.encode(body).ids) + overhead
        if abs(got - target_tokens) <= max(64, target_tokens // 200):
            break
        n_lines = max(1, int(n_lines * target_tokens / got))
    lines = [filler(i) for i in range(1, n_lines + 1)]
    lines.insert(int(len(lines) * depth_pct / 100), NEEDLE)
    return "".join(lines) + QUESTION


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=120000)
    ap.add_argument("--depth", type=int, default=90)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOKENIZER)
    prompt = build_prompt(tok, args.tokens, args.depth)
    print(f"[probe] built prompt: ~{len(tok.encode(prompt).ids)} tokens, "
          f"needle at {args.depth}%", file=sys.stderr)

    payload = {
        "model": "nex-n2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8", "ignore"))

    reply = obj["choices"][0]["message"]["content"]
    t = obj.get("timings", {})
    verdict = "PASS" if MAGIC in reply else "FAIL"
    prompt_n = t.get("prompt_n", -1)
    prefill_s = round(t.get("prompt_ms", 0) / 1000)
    decode_ts = round(t.get("predicted_per_second", 0), 1)

    print(f"[probe] reply tail: ...{reply[-160:]!r}", file=sys.stderr)
    print(f"[probe] prompt_n={prompt_n} prefill_s={prefill_s} "
          f"decode_ts={decode_ts}", file=sys.stderr)
    print(f"{verdict},{prompt_n},{args.depth},{prefill_s}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
