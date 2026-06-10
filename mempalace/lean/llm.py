#!/usr/bin/env python3
"""Minimal chat client for the local llama.cpp server (:8090).

NexN2 is a reasoning model: it emits a <think> trace before the answer.
chat() returns the visible answer (trace stripped); the raw text and server
timings are available via chat_raw().
"""
from __future__ import annotations

import json
import re
import urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text)
    # unclosed trace (max_tokens hit mid-think): drop everything
    if "<think>" in text:
        text = text.split("<think>")[0]
    # server may omit the opening tag and emit a bare closing tag
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


class LLM:
    def __init__(self, url: str = URL, timeout: float = 3600.0):
        self.url = url
        self.timeout = timeout

    def chat_raw(self, messages, *, max_tokens: int = 1024,
                 temp: float = 0.0, seed: int = 0) -> dict:
        payload = {
            "model": "nex-n2-mini",
            "messages": messages,
            "temperature": temp,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8", "ignore"))
        return {
            "text": obj["choices"][0]["message"]["content"],
            "timings": obj.get("timings", {}),
        }

    def chat(self, messages, *, max_tokens: int = 1024,
             temp: float = 0.0, seed: int = 0) -> str:
        return strip_think(self.chat_raw(
            messages, max_tokens=max_tokens, temp=temp, seed=seed)["text"])


if __name__ == "__main__":
    llm = LLM()
    out = llm.chat([{"role": "user", "content":
                     "Reply with exactly: LLM-CLIENT-OK"}], max_tokens=400)
    print(repr(out))
    assert "LLM-CLIENT-OK" in out
    print("llm self-test OK")
