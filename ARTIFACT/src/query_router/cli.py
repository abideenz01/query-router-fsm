"""Command line entry point.

  python -m query_router ask "Who owns the billing-api?" --role analyst
  python -m query_router eval --queries examples/eval_queries.jsonl --out report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import load_config
from .evaluate import (CostModel, FanOutStrategy, RouterStrategy, VectorOnlyStrategy,
                       evaluate, load_queries, render_markdown)
from .fsm import QueryRouter
from .proposers import HeuristicProposer
from .tools import build_registry

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "examples" / "data"


def _build(args):
    cfg = load_config(args.config)
    data_dir = Path(args.data_dir or os.environ.get("ROUTER_DATA_DIR") or DEFAULT_DATA)
    if not data_dir.is_dir():
        raise SystemExit(f"data directory not found: {data_dir}")
    registry = build_registry(data_dir, cfg)
    if args.proposer == "llm":
        from .llm_adapter import OpenAICompatibleProposer
        try:
            proposer = OpenAICompatibleProposer(registry, cfg)
        except RuntimeError as exc:
            raise SystemExit(f"llm proposer not configured: {exc}") from None
    else:
        proposer = HeuristicProposer(cfg, list(registry.get("directory").by_key))
    log_text = os.environ.get("ROUTER_LOG_QUERY_TEXT") == "1"
    router = QueryRouter(registry, proposer, cfg, log_query_text=log_text)
    return cfg, registry, proposer, router


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="query_router")
    ap.add_argument("--config", help="JSON override for default config")
    ap.add_argument("--data-dir", help="directory with orders.csv, docs.jsonl, directory.json")
    ap.add_argument("--proposer", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--log-level", default=os.environ.get("ROUTER_LOG_LEVEL", "WARNING"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="route one query and print the trace")
    a.add_argument("query")
    a.add_argument("--role", default="analyst")
    a.add_argument("--no-items", action="store_true", help="omit result rows from output")
    e = sub.add_parser("eval", help="run the evaluation harness")
    e.add_argument("--queries", required=True)
    e.add_argument("--out", help="write the JSON report here")
    e.add_argument("--markdown", help="write a markdown summary here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr, format="%(message)s")

    cfg, registry, proposer, router = _build(args)
    if args.cmd == "ask":
        res = router.handle(args.query, args.role)
        print(json.dumps(res.to_dict(include_items=not args.no_items), indent=2))
        return 0 if res.status in ("ANSWERED", "PARTIAL") else 2

    if not isinstance(proposer, HeuristicProposer):
        heuristic = HeuristicProposer(cfg, list(registry.get("directory").by_key))
    else:
        heuristic = proposer
    roles = {k: set(v) for k, v in cfg["roles"].items()}
    strategies = [RouterStrategy(router), VectorOnlyStrategy(registry, roles),
                  FanOutStrategy(registry, heuristic, roles)]
    report = evaluate(load_queries(args.queries), strategies, CostModel(cfg))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(report))
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
