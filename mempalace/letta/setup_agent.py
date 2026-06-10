#!/usr/bin/env python3
"""Create (or recreate) the MemPalace Letta agent against local endpoints.

Pre-req: `~/letta-venv/bin/letta server` running on :8283 (see serve_letta.sh),
NexN2 on :8090, nomic-embed on :8091.

Run with ~/letta-venv/bin/python.
"""
import argparse
import sys

from letta_client import Letta

AGENT_NAME = "mempalace-letta"

LLM_CONFIG = {
    "model": "nex-n2-mini",
    "model_endpoint_type": "openai",
    "model_endpoint": "http://127.0.0.1:8090/v1",
    "context_window": 262144,
    "put_inner_thoughts_in_kwargs": True,
    "temperature": 0.0,
}
EMBEDDING_CONFIG = {
    "embedding_model": "nomic-embed",
    "embedding_endpoint_type": "openai",
    "embedding_endpoint": "http://127.0.0.1:8091/v1",
    "embedding_dim": 768,
    "embedding_chunk_size": 512,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8283")
    ap.add_argument("--fresh", action="store_true",
                    help="delete an existing agent of the same name first")
    args = ap.parse_args()

    client = Letta(base_url=args.base_url)
    existing = [a for a in client.agents.list() if a.name == AGENT_NAME]
    if existing and args.fresh:
        for a in existing:
            client.agents.delete(a.id)
        existing = []
    if existing:
        print(f"agent exists: {existing[0].id}")
        return 0

    agent = client.agents.create(
        name=AGENT_NAME,
        llm_config=LLM_CONFIG,
        embedding_config=EMBEDDING_CONFIG,
        memory_blocks=[
            {"label": "persona",
             "value": "I am a long-form autonomous research assistant. I keep "
                      "my core memory current and archive findings eagerly."},
            {"label": "human",
             "value": "Researcher running long rolling research sessions on "
                      "a local workstation."},
        ],
    )
    print(f"created agent: {agent.id}")

    reply = client.agents.messages.create(
        agent_id=agent.id,
        messages=[{"role": "user",
                   "content": "Reply with exactly: LETTA-LOCAL-OK"}])
    for m in reply.messages:
        print(f"  {getattr(m, 'message_type', '?')}: "
              f"{str(getattr(m, 'content', ''))[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
