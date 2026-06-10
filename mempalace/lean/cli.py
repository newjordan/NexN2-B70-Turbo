#!/usr/bin/env python3
"""MemPalace CLI — drive the lean substrate from the shell.

  cli.py --palace ~/nx2-palace stats
  cli.py --palace ~/nx2-palace add "Title" "body text [[link]]" --importance 0.7
  cli.py --palace ~/nx2-palace search "delta net cpu"
  cli.py --palace ~/nx2-palace step "What do we know about KV cache cost?"
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--palace", default="/home/frosty40/nx2-palace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("add")
    p.add_argument("title")
    p.add_argument("body")
    p.add_argument("--importance", type=float, default=0.5)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=6)
    p = sub.add_parser("step")
    p.add_argument("prompt")
    args = ap.parse_args()

    from controller import Controller
    c = Controller(args.palace)

    if args.cmd == "stats":
        print(json.dumps({**c.store.stats(), "turns": len(c.turns),
                          "hot_tokens": c._hot_size(),
                          "core": c.core}, indent=1))
    elif args.cmd == "add":
        slug = c.save_room(args.title, args.body, importance=args.importance)
        print(f"saved room: {slug}")
    elif args.cmd == "search":
        for r in c.retriever.retrieve(args.query, k=args.k):
            print(f"{r['score']:>7}  {r['slug']}  ({r['kind']}, "
                  f"imp {r['importance']:.2f})")
    elif args.cmd == "step":
        print(c.step(args.prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
