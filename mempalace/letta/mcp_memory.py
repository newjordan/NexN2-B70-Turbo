#!/usr/bin/env python3
"""MCP stdio server exposing the Letta agent's archival memory to Hermes.

Tools:
  memory_save(content, tags?)   -> POST /v1/agents/{id}/archival-memory
  memory_search(query, top_k?)  -> GET  /v1/agents/{id}/archival-memory/search

The Letta agent's archival store (pgvector + local nomic embeddings) does the
heavy lifting; this bridge skips the Letta agent loop entirely, so Hermes
keeps its own brain (:8090) and just gains durable semantic memory.

Stdlib only; newline-delimited JSON-RPC per the MCP stdio transport.
Env: LETTA_BASE (default http://127.0.0.1:8283), LETTA_AGENT_ID (required,
or auto-resolved from the agent name 'mempalace-letta').
"""
import json
import os
import sys
import urllib.parse
import urllib.request

BASE = os.environ.get("LETTA_BASE", "http://127.0.0.1:8283").rstrip("/")
AGENT = os.environ.get("LETTA_AGENT_ID", "")

TOOLS = [
    {
        "name": "memory_save",
        "description": (
            "Save a durable research finding, fact, decision or citation to "
            "long-term memory. Saved items survive across sessions and are "
            "retrievable by semantic search. Save eagerly: anything not "
            "saved may be lost when the conversation window rolls."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string",
                            "description": "the finding to remember, self-contained"},
                "tags": {"type": "array", "items": {"type": "string"},
                         "description": "optional topic tags"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Semantically search long-term memory for previously saved "
            "findings. Use before answering questions about earlier "
            "research, prior sessions, or anything possibly discussed before."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]


def http(method: str, path: str, body=None, query=None):
    url = f"{BASE}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def resolve_agent() -> str:
    global AGENT
    if AGENT:
        return AGENT
    agents = http("GET", "/v1/agents/")
    for a in agents:
        if a.get("name") == "mempalace-letta":
            AGENT = a["id"]
            return AGENT
    raise RuntimeError("no LETTA_AGENT_ID and no agent named mempalace-letta")


def memory_save(args: dict) -> str:
    body = {"text": args["content"]}
    if args.get("tags"):
        body["tags"] = list(args["tags"])
    http("POST", f"/v1/agents/{resolve_agent()}/archival-memory", body=body)
    return "saved to long-term memory"


def memory_search(args: dict) -> str:
    out = http("GET", f"/v1/agents/{resolve_agent()}/archival-memory/search",
               query={"query": args["query"],
                      "top_k": int(args.get("top_k", 5))})
    hits = out.get("results", out) if isinstance(out, dict) else out
    if not hits:
        return "no matching memories"
    lines = []
    for h in hits[:int(args.get("top_k", 5))]:
        text = h.get("content", h.get("text", str(h)))
        lines.append(f"- {text}")
    return "\n".join(lines)


def handle(req: dict):
    method = req.get("method", "")
    if method == "initialize":
        return {"protocolVersion": req["params"].get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace-memory", "version": "1.0.0"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = req["params"]["name"]
        args = req["params"].get("arguments", {})
        try:
            text = {"memory_save": memory_save,
                    "memory_search": memory_search}[name](args)
            return {"content": [{"type": "text", "text": text}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"memory error: {e}"}],
                    "isError": True}
    if method == "ping":
        return {}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in req:       # notification — no response
            continue
        result = handle(req)
        resp = {"jsonrpc": "2.0", "id": req["id"]}
        if result is None:
            resp["error"] = {"code": -32601,
                             "message": f"method not found: {req.get('method')}"}
        else:
            resp["result"] = result
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
