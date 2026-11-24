"""Configuration loading: packaged defaults, optional JSON override, env overrides."""
from __future__ import annotations

import copy
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

ENV_CONFIG_PATH = "ROUTER_CONFIG"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults, then merge a JSON override from `path` or $ROUTER_CONFIG."""
    raw = resources.files("query_router").joinpath("default_config.json").read_text()
    cfg = json.loads(raw)
    override_path = path or os.environ.get(ENV_CONFIG_PATH)
    if override_path:
        p = Path(override_path)
        if not p.is_file():
            raise FileNotFoundError(f"config override not found: {p}")
        cfg = _deep_merge(cfg, json.loads(p.read_text()))
    _check(cfg)
    return cfg


def _check(cfg: dict) -> None:
    r = cfg["routing"]
    if not 0 < r["min_confidence"] <= 1:
        raise ValueError("routing.min_confidence must be in (0, 1]")
    if r["max_subtasks"] < 1:
        raise ValueError("routing.max_subtasks must be >= 1")
    b = cfg["budgets"]
    for key in ("max_tool_calls", "max_transitions"):
        if b[key] < 1:
            raise ValueError(f"budgets.{key} must be >= 1")
    if b["max_retries"] < 0:
        raise ValueError("budgets.max_retries must be >= 0")
    known = set(cfg["fallbacks"])
    for src, targets in cfg["fallbacks"].items():
        for t in targets:
            if t not in known:
                raise ValueError(f"fallback {src}->{t} targets unknown route")
            if t == src:
                raise ValueError(f"fallback {src}->{t} would loop")
    _check_fallback_cycles(cfg["fallbacks"])


def _check_fallback_cycles(graph: dict[str, list[str]]) -> None:
    """Reject fallback chains that can revisit a route (A->B->A)."""
    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node: str, trail: list[str]) -> None:
        if node in done:
            return
        if node in visiting:
            raise ValueError("fallback cycle: " + " -> ".join(trail + [node]))
        visiting.add(node)
        for nxt in graph.get(node, []):
            walk(nxt, trail + [node])
        visiting.discard(node)
        done.add(node)

    for start in graph:
        walk(start, [])
