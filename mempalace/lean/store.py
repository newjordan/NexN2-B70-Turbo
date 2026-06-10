#!/usr/bin/env python3
"""MemPalace store — findings live as linked Markdown "rooms" on disk,
indexed for hybrid retrieval (sqlite FTS5 BM25 + embedding vectors) with a
[[link]] graph for associative recall.

Layout under the palace root:
  rooms/<slug>.md     one room per finding/note (markdown, [[link]]s inline)
  palace.db           sqlite: rooms table, FTS5 mirror, embeddings, links

The markdown files are the source of truth; the db is a derived index and
can be rebuilt from them (reindex()).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time

import numpy as np

LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
SLUG_RE = re.compile(r"[^a-z0-9]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms (
    id INTEGER PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL,
    accessed_ts REAL NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    kind TEXT NOT NULL DEFAULT 'note'
);
CREATE VIRTUAL TABLE IF NOT EXISTS rooms_fts USING fts5(
    slug, title, body, tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS embeddings (
    room_id INTEGER PRIMARY KEY REFERENCES rooms(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS links (
    src INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    dst_slug TEXT NOT NULL,
    PRIMARY KEY (src, dst_slug)
);
"""


def slugify(title: str) -> str:
    return SLUG_RE.sub("-", title.lower()).strip("-")[:80] or "untitled"


class PalaceStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.rooms_dir = os.path.join(self.root, "rooms")
        os.makedirs(self.rooms_dir, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(self.root, "palace.db"))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---------------------------------------------------------------- write

    def write_room(self, title: str, body: str, *, importance: float = 0.5,
                   kind: str = "note", embedding=None) -> str:
        """Create or overwrite a room. Returns the slug."""
        slug = slugify(title)
        now = time.time()
        path = os.path.join(self.rooms_dir, f"{slug}.md")
        with open(path, "w") as f:
            f.write(f"# {title}\n\n{body.strip()}\n")

        cur = self.db.execute("SELECT id FROM rooms WHERE slug=?", (slug,))
        row = cur.fetchone()
        if row:
            rid = row["id"]
            self.db.execute(
                "UPDATE rooms SET title=?, updated_ts=?, accessed_ts=?, "
                "importance=?, kind=? WHERE id=?",
                (title, now, now, importance, kind, rid))
            self.db.execute("DELETE FROM rooms_fts WHERE slug=?", (slug,))
            self.db.execute("DELETE FROM links WHERE src=?", (rid,))
        else:
            cur = self.db.execute(
                "INSERT INTO rooms (slug,title,created_ts,updated_ts,"
                "accessed_ts,importance,kind) VALUES (?,?,?,?,?,?,?)",
                (slug, title, now, now, now, importance, kind))
            rid = cur.lastrowid
        self.db.execute(
            "INSERT INTO rooms_fts (slug,title,body) VALUES (?,?,?)",
            (slug, title, body))
        for target in set(LINK_RE.findall(body)):
            self.db.execute(
                "INSERT OR IGNORE INTO links (src,dst_slug) VALUES (?,?)",
                (rid, slugify(target)))
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            self.db.execute(
                "INSERT OR REPLACE INTO embeddings (room_id,dim,vec) "
                "VALUES (?,?,?)", (rid, vec.size, vec.tobytes()))
        self.db.commit()
        return slug

    def set_embedding(self, slug: str, embedding) -> None:
        rid = self._rid(slug)
        vec = np.asarray(embedding, dtype=np.float32)
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings (room_id,dim,vec) "
            "VALUES (?,?,?)", (rid, vec.size, vec.tobytes()))
        self.db.commit()

    # ----------------------------------------------------------------- read

    def read_room(self, slug: str, *, touch: bool = True) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM rooms WHERE slug=?", (slug,)).fetchone()
        if not row:
            return None
        path = os.path.join(self.rooms_dir, f"{slug}.md")
        body = open(path).read() if os.path.exists(path) else ""
        if touch:
            self.db.execute("UPDATE rooms SET accessed_ts=? WHERE id=?",
                            (time.time(), row["id"]))
            self.db.commit()
        return {**dict(row), "body": body}

    def search_bm25(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return [(slug, bm25_score)] — higher is better."""
        # sanitize: FTS5 query syntax chokes on raw punctuation
        q = " ".join(re.findall(r"[A-Za-z0-9]+", query)) or '""'
        rows = self.db.execute(
            "SELECT slug, bm25(rooms_fts) AS s FROM rooms_fts "
            "WHERE rooms_fts MATCH ? ORDER BY s LIMIT ?", (q, k)).fetchall()
        return [(r["slug"], -r["s"]) for r in rows]  # bm25() is lower-better

    def search_vec(self, qvec, k: int = 10) -> list[tuple[str, float]]:
        """Brute-force cosine over all stored embeddings (fine at palace scale)."""
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        rows = self.db.execute(
            "SELECT r.slug, e.vec FROM embeddings e "
            "JOIN rooms r ON r.id=e.room_id").fetchall()
        scored = []
        for r in rows:
            v = np.frombuffer(r["vec"], dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            scored.append((r["slug"], float(q @ v)))
        scored.sort(key=lambda t: -t[1])
        return scored[:k]

    def neighbors(self, slug: str) -> list[str]:
        """Out-links plus back-links (1 hop)."""
        rid = self._rid(slug)
        out = [r["dst_slug"] for r in self.db.execute(
            "SELECT dst_slug FROM links WHERE src=?", (rid,))]
        back = [r["slug"] for r in self.db.execute(
            "SELECT r.slug FROM links l JOIN rooms r ON r.id=l.src "
            "WHERE l.dst_slug=?", (slug,))]
        return sorted(set(out + back) - {slug})

    def meta(self, slug: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM rooms WHERE slug=?", (slug,)).fetchone()
        return dict(row) if row else None

    def all_slugs(self) -> list[str]:
        return [r["slug"] for r in
                self.db.execute("SELECT slug FROM rooms ORDER BY id")]

    def stats(self) -> dict:
        n = self.db.execute("SELECT COUNT(*) c FROM rooms").fetchone()["c"]
        e = self.db.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]
        l = self.db.execute("SELECT COUNT(*) c FROM links").fetchone()["c"]
        return {"rooms": n, "embedded": e, "links": l}

    # ------------------------------------------------------------- internal

    def _rid(self, slug: str) -> int:
        row = self.db.execute(
            "SELECT id FROM rooms WHERE slug=?", (slug,)).fetchone()
        if not row:
            raise KeyError(f"no room {slug!r}")
        return row["id"]


if __name__ == "__main__":  # tiny self-test on a temp palace
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        st = PalaceStore(d)
        st.write_room("Alpha finding", "Quantum cats relate to [[Beta note]].",
                      embedding=np.ones(8))
        st.write_room("Beta note", "Beta links back to [[Alpha finding]].",
                      embedding=-np.ones(8))
        assert st.search_bm25("quantum cats")[0][0] == "alpha-finding"
        assert st.search_vec(np.ones(8))[0][0] == "alpha-finding"
        assert st.neighbors("alpha-finding") == ["beta-note"]
        assert st.read_room("beta-note")["body"].startswith("# Beta note")
        print("store self-test OK", json.dumps(st.stats()))
