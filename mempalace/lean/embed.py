#!/usr/bin/env python3
"""Client for the local embedding endpoint (llama.cpp --embedding on :8091).

nomic-embed-text-v1.5 is asymmetric: documents must be prefixed with
"search_document: " and queries with "search_query: " or retrieval quality
drops sharply.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np

URL = "http://127.0.0.1:8091/v1/embeddings"


class EmbedClient:
    def __init__(self, url: str = URL, timeout: float = 120.0):
        self.url = url
        self.timeout = timeout

    def _embed(self, texts: list[str]) -> np.ndarray:
        req = urllib.request.Request(
            self.url,
            data=json.dumps({"input": texts, "model": "nomic-embed"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8", "ignore"))
        data = sorted(obj["data"], key=lambda d: d["index"])
        return np.asarray([d["embedding"] for d in data], dtype=np.float32)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"search_document: {t}"[:16000] for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([f"search_query: {text}"[:16000]])[0]


if __name__ == "__main__":
    c = EmbedClient()
    docs = c.embed_docs(["the cat sat on the mat", "stock markets fell today"])
    q = c.embed_query("where did the cat sit?")
    sims = (docs @ q) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))
    print(f"dim={docs.shape[1]} sims={sims.round(3).tolist()}")
    assert sims[0] > sims[1], "cat query should match cat doc"
    print("embed self-test OK")
