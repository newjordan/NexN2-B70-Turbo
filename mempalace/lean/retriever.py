#!/usr/bin/env python3
"""Hybrid retriever over the PalaceStore.

Pipeline: BM25 top-N + vector top-N fused with Reciprocal Rank Fusion,
1-hop [[link]] expansion at a decayed score (HippoRAG-style associative
recall), then Generative-Agents-style scoring:

    score = w_rel * relevance + w_rec * recency + w_imp * importance

recency decays exponentially on hours since last access.
"""
from __future__ import annotations

import math
import time

from store import PalaceStore
from embed import EmbedClient

RRF_K = 60
LINK_DECAY = 0.6          # neighbor inherits this fraction of the seed's score
RECENCY_HALFLIFE_H = 24.0


class Retriever:
    def __init__(self, store: PalaceStore, embedder: EmbedClient | None = None,
                 *, w_rel: float = 1.0, w_rec: float = 0.25,
                 w_imp: float = 0.25):
        self.store = store
        self.embedder = embedder
        self.w = (w_rel, w_rec, w_imp)

    def retrieve(self, query: str, k: int = 6) -> list[dict]:
        """Return up to k rooms (dicts with slug/title/body/score)."""
        pool = max(k * 3, 12)
        ranks: dict[str, float] = {}

        for rank, (slug, _s) in enumerate(self.store.search_bm25(query, pool)):
            ranks[slug] = ranks.get(slug, 0.0) + 1.0 / (RRF_K + rank + 1)
        if self.embedder is not None:
            try:
                qvec = self.embedder.embed_query(query)
                for rank, (slug, _s) in enumerate(
                        self.store.search_vec(qvec, pool)):
                    ranks[slug] = ranks.get(slug, 0.0) + 1.0 / (RRF_K + rank + 1)
            except Exception as e:  # embedding endpoint down -> BM25 only
                print(f"[retriever] vector leg skipped: {e}")

        # 1-hop associative expansion from the current top seeds
        for slug, base in sorted(ranks.items(), key=lambda t: -t[1])[:k]:
            for nb in self.store.neighbors(slug):
                if self.store.meta(nb) is not None:
                    ranks[nb] = max(ranks.get(nb, 0.0), base * LINK_DECAY)

        if not ranks:
            return []
        top_rel = max(ranks.values())
        now = time.time()
        w_rel, w_rec, w_imp = self.w
        scored = []
        for slug, rel in ranks.items():
            m = self.store.meta(slug)
            if m is None:
                continue
            age_h = max(0.0, (now - m["accessed_ts"]) / 3600.0)
            recency = math.exp(-math.log(2) * age_h / RECENCY_HALFLIFE_H)
            score = (w_rel * rel / top_rel + w_rec * recency
                     + w_imp * m["importance"])
            scored.append((score, slug))
        scored.sort(key=lambda t: -t[0])

        out = []
        for score, slug in scored[:k]:
            room = self.store.read_room(slug)   # touches accessed_ts
            room["score"] = round(score, 4)
            out.append(room)
        return out


if __name__ == "__main__":  # self-test, BM25-only (no endpoint needed)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        st = PalaceStore(d)
        st.write_room("Delta-Net layers", "NX2 runs 30 recurrent Delta-Net "
                      "layers on CPU; see [[KV cache math]].")
        st.write_room("KV cache math", "About 20 KiB per token at f16.")
        st.write_room("Unrelated", "Pasta recipes and garden tips.")
        r = Retriever(st, embedder=None)
        got = [x["slug"] for x in r.retrieve("recurrent delta-net CPU", k=2)]
        assert got[0] == "delta-net-layers", got
        assert "kv-cache-math" in got, got  # pulled in via [[link]]
        print("retriever self-test OK:", got)
