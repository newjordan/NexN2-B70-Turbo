#!/usr/bin/env python3
"""Verify the multi-needle cost model: does a second probe sharing the same
haystack prefix reuse the prompt cache (prompt_n ~= question length), or does
the recurrent Delta-Net state force a full re-prefill?

Sends the same haystack with two different trailing questions and reports
prompt_n / cache_n / prompt_ms for each request.
"""
import json
import sys
import urllib.request

from tokenizers import Tokenizer

TOKENIZER = "/home/frosty40/models/nex-n2-mini-bf16/tokenizer.json"
URL = "http://127.0.0.1:8090/v1/chat/completions"


def filler(i: int) -> str:
    return (f"Note {i}: inventory record {i} lists {i % 97 + 3} crates in "
            f"warehouse {i % 12}, checked on day {i % 365}.\n")


def ask(prompt: str, timeout: float = 3600.0) -> dict:
    payload = {
        "model": "nex-n2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 400,
        "stream": False,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
    tok = Tokenizer.from_file(TOKENIZER)
    n_lines = target * 100 // len(tok.encode("".join(
        filler(i) for i in range(1, 101))).ids)
    hay = "".join(filler(i) for i in range(1, n_lines + 1))
    hay = (f"{hay}\nIMPORTANT: the secret magic code is 7C4F-9B2E-31AD. "
           f"Remember it.\n")
    print(f"[cache] haystack ~{len(tok.encode(hay).ids)} tokens", file=sys.stderr)

    questions = [
        "\n\nWhat is the secret magic code? Reply with the code only.",
        "\n\nHow many crates does Note 7 list? Reply with the number only.",
        "\n\nWhat day was Note 42 checked on? Reply with the number only.",
    ]
    for k, q in enumerate(questions, 1):
        obj = ask(hay + q)
        t = obj.get("timings", {})
        reply = obj["choices"][0]["message"]["content"]
        print(f"probe{k}: prompt_n={t.get('prompt_n')} "
              f"cache_n={t.get('cache_n')} "
              f"prompt_s={round(t.get('prompt_ms', 0) / 1000, 1)} "
              f"reply_tail={reply[-60:]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
