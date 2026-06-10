# Memory substrate bakeoff — lean MemPalace vs Letta (round 1)

Identical seeded 40-step rolling session (8 facts injected early, distractor
research steps, recall queried at the end), same local model (NexN2 Q5_K_M,
native 262144, f16 KV on :8090), same embedding endpoint (:8091).
Raw rows: `memory-bakeoff.csv`. Harness: `eval/memory/bakeoff.py`.

| metric | lean | letta |
|---|---:|---:|
| recall@end (8 facts) | **0.25** (2/8) | **1.00** (8/8) |
| latency s/step (mean) | 30.0 | 17.2 |
| tokens prefilled (session) | **390,128** | 2,352,724 (6.0×) |
| substrate storage | 277 KB (files+sqlite) | postgres (not measured — adapter gap) |
| hot-window policy | 24k budget, 7 compactions | 262k window, never evicted |
| install friction | stdlib+numpy, zero setup | 988 MB venv + embedded postgres + 5 bring-up fixes (see mempalace/letta/README.md) |
| think-trace handling | stripped | leaks `</think>` into replies |

## What actually happened (read before citing the recall number)

- **Letta never exercised tiered memory.** With `context_window=262144` and a
  ~2.35M-token session spread over 48 calls, everything stayed in its rolling
  context — round 1 measures *giant raw window* vs *tiered memory under
  pressure*, and raw window won, at 6× the token cost. That is itself a real
  finding: **below ~200k cumulative hot tokens, just using the 256k native
  window beats any palace.**
- **Lean's losses are diagnosable:** facts the model never saved as dedicated
  rooms got buried in generic compaction rooms; query-time retrieval (k=6)
  surfaced distractor rooms; NX2's meta-monologue pollutes short answers.
  Fix levers: eager per-fact room extraction at write time, larger k +
  reranking, answer-format prompting.
- **What's good at what:** lean = 6× cheaper tokens, transparent storage,
  trivially hackable, but its controller loses facts under compaction
  pressure. Letta = turnkey recall while the session fits its window, at high
  token cost, heavy install, and opaque storage.

## Verdict

Round-1 winner on measurements: **Letta** — it is the integration candidate
(T9). The decisive regime (sessions that overflow 262k — the actual
"millions of cumulative tokens" use case) is untested in round 1; a round 2
with letta's `context_window` capped (e.g. 32k) and/or 200+ steps would
measure real eviction behavior for both. Lean remains worth iterating for
its 6× cost advantage if its save/retrieve loop is hardened.
