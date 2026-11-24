"""Evaluation harness: score the state-machine router against two baselines.

Strategies
  router       proposer + state machine (this package)
  vector_only  every query goes to document search ("RAG for everything")
  fan_out      every query calls every tool the role may use, then synthesises

Baselines are scored generously. Fan-out is assumed to have a perfect
synthesiser that finds the right evidence in whatever it retrieved, and both
baselines get the same sufficiency check and ACLs as the router. Neither
baseline can clarify or refuse, because nothing in their design has a state
for it. That is the property being measured, not a handicap added for effect.

Cost and latency are modelled from the cost_model config block (illustrative
values). Replace them with your own measurements before quoting numbers.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import schema
from .fsm import QueryRouter, RouteResult
from .proposers import HeuristicProposer
from .tools import ToolRegistry, ToolResult

ANSWERABLE = "ANSWERED"


@dataclass
class Evidence:
    route: str
    args: dict
    items: list[dict]


@dataclass
class Outcome:
    status: str
    evidence: list[Evidence] = field(default_factory=list)
    tool_routes: list[str] = field(default_factory=list)
    router_call: bool = False
    parallel_tools: bool = False
    label: str = ""


def canonical_metric_args(args: dict) -> tuple:
    return (args.get("metric"), args.get("period_start"), args.get("period_end"),
            tuple(sorted(args.get("group_by") or [])), args.get("region"),
            args.get("product_line"))


def part_matched(part: dict, ev: Evidence) -> bool:
    if part["route"] != ev.route or not ev.items:
        return False
    if part["route"] == "metrics":
        return canonical_metric_args(part["args"]) == canonical_metric_args(ev.args)
    if part["route"] == "policy":
        return any(i.get("doc_id") == part["doc_id"] for i in ev.items)
    if part["route"] == "directory":
        return any(i.get("entity") == part["entity"] and i.get("field") == part["field"]
                   for i in ev.items)
    return False


def relevant_items(parts: list[dict], ev: Evidence) -> int:
    n = 0
    for item in ev.items:
        for part in parts:
            if part["route"] != ev.route:
                continue
            if part["route"] == "metrics" and part_matched(part, ev):
                n += 1
                break
            if part["route"] == "policy" and item.get("doc_id") == part["doc_id"]:
                n += 1
                break
            if (part["route"] == "directory" and item.get("entity") == part["entity"]
                    and item.get("field") == part["field"]):
                n += 1
                break
    return n


# --------------------------------------------------------------------------- strategies

class RouterStrategy:
    name = "router"

    def __init__(self, router: QueryRouter) -> None:
        self.router = router
        self.last: RouteResult | None = None

    def run(self, query: str, role: str) -> Outcome:
        res = self.router.handle(query, role)
        self.last = res
        ev = [Evidence(s.route_path[-1], s.args, s.items)
              for s in res.subtasks if s.state.value == "ANSWERED"]
        routes = [c["tool"] for s in res.subtasks for c in s.tool_calls]
        if len(res.subtasks) > 1:
            label = "compound"
        elif res.status == "ANSWERED":
            label = res.subtasks[0].route_path[-1]
        else:
            label = res.status
        return Outcome(res.status, ev, routes, router_call=True, label=label)


class VectorOnlyStrategy:
    name = "vector_only"

    def __init__(self, registry: ToolRegistry, roles: dict) -> None:
        self.tool, self.roles = registry.get("policy"), roles

    def run(self, query: str, role: str) -> Outcome:
        if "policy" not in self.roles.get(role, ()):
            return Outcome("NO_ANSWER", label="NO_ANSWER")
        res: ToolResult = self.tool.run({"query": query[:300]}, role)
        if res.sufficient:
            return Outcome("ANSWERED", [Evidence("policy", {"query": query}, res.items)],
                           ["policy_search"], label="policy")
        return Outcome("NO_ANSWER", [], ["policy_search"], label="NO_ANSWER")


class FanOutStrategy:
    name = "fan_out"

    def __init__(self, registry: ToolRegistry, proposer: HeuristicProposer, roles: dict) -> None:
        self.registry, self.proposer, self.roles = registry, proposer, roles

    def run(self, query: str, role: str) -> Outcome:
        granted = self.roles.get(role, set())
        evidence, called = [], []
        for route in self.registry.routes():
            tool = self.registry.get(route)
            args, _ = self.proposer.extract(route, query)
            if not tool.required_scopes(args) <= set(granted):
                continue
            if schema.validate(args, tool.spec.parameters):
                continue
            try:
                tool.check_args(args)
            except ValueError:
                continue
            called.append(tool.spec.name)
            res = tool.run(args, role)
            if res.sufficient:
                evidence.append(Evidence(route, args, res.items))
        status = "ANSWERED" if evidence else "NO_ANSWER"
        label = evidence[0].route if len(evidence) == 1 else ("fan_out" if evidence else status)
        return Outcome(status, evidence, called, parallel_tools=True, label=label)


# --------------------------------------------------------------------------- cost model

class CostModel:
    def __init__(self, cfg: dict) -> None:
        self.c = cfg["cost_model"]
        self.tool_by_name = {"metrics_query": "metrics", "policy_search": "policy",
                             "directory_lookup": "directory"}

    def _usd(self, tin: int, tout: int) -> float:
        return tin * self.c["usd_per_1m_input_tokens"] / 1e6 + tout * self.c["usd_per_1m_output_tokens"] / 1e6

    def price(self, out: Outcome) -> dict:
        usd, latency, tin, routing_usd = 0.0, 0.0, 0, 0.0
        if out.router_call:
            routing_usd = self._usd(self.c["router_input_tokens"], self.c["router_output_tokens"])
            usd += routing_usd
            latency += self.c["router_latency_ms"]
        tool_lat = []
        for name in out.tool_routes:
            t = self.c["tools"][self.tool_by_name[name]]
            usd += t["usd_per_call"]
            tool_lat.append(t["latency_ms"])
        latency += (max(tool_lat) if out.parallel_tools else sum(tool_lat)) if tool_lat else 0
        if out.status in ("ANSWERED", "PARTIAL"):
            evidence_tokens = sum(max(1, len(json.dumps(e.items)) // 4) for e in out.evidence)
            tin = self.c["synthesis_base_input_tokens"] + evidence_tokens
            usd += self._usd(tin, self.c["synthesis_output_tokens"])
            latency += self.c["synthesis_latency_ms"] + self.c["synthesis_latency_ms_per_1k_tokens"] * tin / 1000
        elif out.status == "CLARIFY" and out.router_call:
            usd += self._usd(0, self.c["clarify_output_tokens"])
        return {"usd": usd, "usd_without_llm_routing": usd - routing_usd,
                "latency_ms": latency, "synthesis_input_tokens": tin}


# --------------------------------------------------------------------------- harness

def load_queries(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "role", "query", "gold"):
                if key not in row:
                    raise ValueError(f"line {n}: missing {key}")
            if "outcome" not in row["gold"]:
                raise ValueError(f"line {n}: gold.outcome missing")
            if row["gold"]["outcome"] == ANSWERABLE and not row["gold"].get("parts"):
                raise ValueError(f"line {n}: ANSWERED gold needs parts")
            rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query ids")
    return rows


def gold_label(row: dict) -> str:
    g = row["gold"]
    if g["outcome"] != ANSWERABLE:
        return g["outcome"]
    return "compound" if len(g["parts"]) > 1 else g["parts"][0]["route"]


def score(row: dict, out: Outcome) -> tuple[bool, str]:
    g = row["gold"]
    if out.status != g["outcome"]:
        return False, f"status {out.status} != {g['outcome']}"
    if g["outcome"] != ANSWERABLE:
        return True, "ok"
    for part in g["parts"]:
        if not any(part_matched(part, ev) for ev in out.evidence):
            return False, f"missing evidence for {part['route']}"
    return True, "ok"


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def evaluate(rows: list[dict], strategies: list, cost: CostModel) -> dict:
    report: dict = {"n_queries": len(rows), "strategies": {}}
    for strat in strategies:
        correct, answerable_ok, non_ok = 0, 0, 0
        n_ans = sum(1 for r in rows if r["gold"]["outcome"] == ANSWERABLE)
        usd, usd_nr, lat, tokens, calls, precisions = [], [], [], [], [], []
        answered_wrongly = 0
        by_cat: dict = defaultdict(lambda: [0, 0])
        confusion: Counter = Counter()
        misses = []
        for row in rows:
            out = strat.run(row["query"], row["role"])
            ok, why = score(row, out)
            price = cost.price(out)
            usd.append(price["usd"])
            usd_nr.append(price["usd_without_llm_routing"])
            if row["gold"]["outcome"] != ANSWERABLE and out.status in ("ANSWERED", "PARTIAL"):
                answered_wrongly += 1
            lat.append(price["latency_ms"])
            tokens.append(price["synthesis_input_tokens"])
            calls.append(len(out.tool_routes))
            cat = row.get("category", "uncategorised")
            by_cat[cat][1] += 1
            if ok:
                correct += 1
                by_cat[cat][0] += 1
                if row["gold"]["outcome"] == ANSWERABLE:
                    answerable_ok += 1
                else:
                    non_ok += 1
            else:
                misses.append({"id": row["id"], "why": why})
            if out.status in ("ANSWERED", "PARTIAL") and out.evidence:
                total = sum(len(e.items) for e in out.evidence)
                rel = sum(relevant_items(row["gold"].get("parts", []), e) for e in out.evidence)
                precisions.append(rel / total if total else 0.0)
            confusion[(gold_label(row), out.label)] += 1
        n = len(rows)
        report["strategies"][strat.name] = {
            "outcome_accuracy": round(correct / n, 3),
            "answerable_accuracy": round(answerable_ok / n_ans, 3) if n_ans else None,
            "non_answerable_accuracy": round(non_ok / (n - n_ans), 3) if n > n_ans else None,
            "tool_calls_per_query": round(sum(calls) / n, 2),
            "answered_when_it_should_not": answered_wrongly,
            "usd_per_1k_queries": round(1000 * sum(usd) / n, 3),
            "usd_per_1k_queries_without_llm_routing": round(1000 * sum(usd_nr) / n, 3),
            "synthesis_input_tokens_per_query": round(sum(tokens) / n, 1),
            "latency_ms_p50": round(_pct(lat, 0.5), 1),
            "latency_ms_p95": round(_pct(lat, 0.95), 1),
            "context_precision_mean": round(statistics.mean(precisions), 3) if precisions else None,
            "accuracy_by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
            "confusion": {f"{g} -> {p}": c for (g, p), c in sorted(confusion.items())},
            "misses": misses,
        }
    return report


def render_markdown(report: dict) -> str:
    cols = ["outcome_accuracy", "answerable_accuracy", "non_answerable_accuracy",
            "answered_when_it_should_not", "tool_calls_per_query", "usd_per_1k_queries",
            "usd_per_1k_queries_without_llm_routing", "synthesis_input_tokens_per_query",
            "latency_ms_p50", "latency_ms_p95", "context_precision_mean"]
    lines = [f"Queries evaluated: {report['n_queries']} (synthetic, labeled)", "",
             "| metric | " + " | ".join(report["strategies"]) + " |",
             "|---|" + "---|" * len(report["strategies"])]
    for c in cols:
        vals = [str(s[c]) for s in report["strategies"].values()]
        lines.append(f"| {c} | " + " | ".join(vals) + " |")
    lines.append("")
    for name, s in report["strategies"].items():
        lines.append(f"{name} by category: " + ", ".join(f"{k} {v}" for k, v in s["accuracy_by_category"].items()))
    return "\n".join(lines) + "\n"
