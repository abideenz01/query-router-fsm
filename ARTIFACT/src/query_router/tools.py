"""Tool registry and the three specialist tools.

Each tool publishes a JSON Schema argument contract, the scope a caller needs,
and a sufficiency check that decides whether its result is good enough to
answer from. The state machine never trusts a tool call it has not validated.
"""
from __future__ import annotations

import csv
import difflib
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .text import content_tokens, normalise


class ArgumentError(ValueError):
    """Arguments passed schema checks but violate a semantic rule."""


class TransientToolError(RuntimeError):
    """Retryable failure (timeout, throttling)."""


class PermanentToolError(RuntimeError):
    """Non-retryable failure (bad state, missing backend)."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    route: str
    description: str
    parameters: dict
    scope: str


@dataclass
class ToolResult:
    items: list[dict]
    sufficient: bool
    reason: str
    notes: list[str] = field(default_factory=list)

    @property
    def evidence_tokens(self) -> int:
        return max(1, len(json.dumps(self.items)) // 4) if self.items else 0


class Tool(Protocol):
    spec: ToolSpec

    def check_args(self, args: dict) -> None: ...

    def required_scopes(self, args: dict) -> set[str]: ...

    def run(self, args: dict, role: str) -> ToolResult: ...


# --------------------------------------------------------------------------- metrics

METRIC_SQL = {
    "revenue": "SUM(revenue_usd)",
    "orders": "SUM(orders)",
    "refunds": "SUM(refunds)",
    "refund_rate": "ROUND(CAST(SUM(refunds) AS REAL) / NULLIF(SUM(orders), 0), 4)",
    "avg_order_value": "ROUND(CAST(SUM(revenue_usd) AS REAL) / NULLIF(SUM(orders), 0), 2)",
}
DIMENSIONS = ("region", "product_line", "month")
REGIONS = ("UAE", "KSA", "Pakistan", "UK")
PRODUCT_LINES = ("Cloud", "Support", "Licenses")
MONTH_PATTERN = r"20[0-9]{2}-(0[1-9]|1[0-2])"


class MetricsTool:
    """Governed metric templates over a SQL table. No free-form SQL."""

    spec = ToolSpec(
        name="metrics_query",
        route="metrics",
        description=(
            "Compute a governed business metric (revenue, orders, refunds, refund_rate, "
            "avg_order_value) over a month range, optionally grouped and filtered."
        ),
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["metric", "period_start", "period_end"],
            "properties": {
                "metric": {"type": "string", "enum": list(METRIC_SQL)},
                "period_start": {"type": "string", "pattern": MONTH_PATTERN},
                "period_end": {"type": "string", "pattern": MONTH_PATTERN},
                "group_by": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(DIMENSIONS)},
                    "maxItems": 2,
                    "uniqueItems": True,
                },
                "region": {"type": "string", "enum": list(REGIONS)},
                "product_line": {"type": "string", "enum": list(PRODUCT_LINES)},
            },
        },
        scope="metrics",
    )

    def __init__(self, csv_path: Path, data_start: str, data_end: str,
                 fault: Callable[[], None] | None = None) -> None:
        self.data_start, self.data_end = data_start, data_end
        self._fault = fault
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute(
            "CREATE TABLE orders (month TEXT, region TEXT, product_line TEXT, "
            "orders INTEGER, revenue_usd INTEGER, refunds INTEGER)"
        )
        with Path(csv_path).open(newline="") as f:
            rows = [
                (r["month"], r["region"], r["product_line"], int(r["orders"]),
                 int(r["revenue_usd"]), int(r["refunds"]))
                for r in csv.DictReader(f)
            ]
        if not rows:
            raise PermanentToolError(f"no rows in {csv_path}")
        self._db.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", rows)

    def check_args(self, args: dict) -> None:
        if args["period_start"] > args["period_end"]:
            raise ArgumentError("period_start is after period_end")
        if args["period_end"] < self.data_start or args["period_start"] > self.data_end:
            raise ArgumentError(
                f"period outside available data {self.data_start}..{self.data_end}"
            )

    def required_scopes(self, args: dict) -> set[str]:
        return {self.spec.scope}

    def build_sql(self, args: dict) -> tuple[str, list[Any]]:
        group = list(args.get("group_by") or [])
        select = ", ".join(group + [f"{METRIC_SQL[args['metric']]} AS value"])
        where, params = ["month BETWEEN ? AND ?"], [args["period_start"], args["period_end"]]
        for col in ("region", "product_line"):
            if args.get(col):
                where.append(f"{col} = ?")
                params.append(args[col])
        sql = f"SELECT {select} FROM orders WHERE {' AND '.join(where)}"
        if group:
            sql += f" GROUP BY {', '.join(group)} ORDER BY {', '.join(group)}"
        return sql, params

    def run(self, args: dict, role: str) -> ToolResult:
        if self._fault:
            self._fault()
        sql, params = self.build_sql(args)
        cur = self._db.execute(sql, params)
        cols = [c[0] for c in cur.description]
        items = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
        items = [i for i in items if i.get("value") is not None]
        if not items:
            return ToolResult([], False, "metric returned no rows for the filters")
        return ToolResult(items, True, f"{len(items)} row(s)")


# --------------------------------------------------------------------------- policy search

class PolicySearchTool:
    """BM25 over policy documents with ACL filtering applied before ranking."""

    spec = ToolSpec(
        name="policy_search",
        route="policy",
        description="Search internal policy and runbook documents the caller may read.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 3, "maxLength": 300},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
        },
        scope="policy",
    )

    def __init__(self, docs_path: Path, top_k: int = 3, min_score: float = 2.0,
                 min_term_coverage: float = 0.5, k1: float = 1.5, b: float = 0.75) -> None:
        self.top_k, self.min_score, self.min_cov = top_k, min_score, min_term_coverage
        self.k1, self.b = k1, b
        self.docs: list[dict] = []
        with Path(docs_path).open() as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                doc = json.loads(line)
                missing = {"id", "title", "text", "acl"} - doc.keys()
                if missing:
                    raise PermanentToolError(f"doc line {line_no} missing {sorted(missing)}")
                doc["_tf"] = Counter(content_tokens(doc["title"] + " " + doc["text"]))
                doc["_len"] = sum(doc["_tf"].values())
                self.docs.append(doc)
        if not self.docs:
            raise PermanentToolError(f"no documents in {docs_path}")
        self.avg_len = sum(d["_len"] for d in self.docs) / len(self.docs)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d["_tf"].keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def check_args(self, args: dict) -> None:
        if not content_tokens(args["query"]):
            raise ArgumentError("query has no searchable terms")

    def required_scopes(self, args: dict) -> set[str]:
        return {self.spec.scope}

    def _score(self, q: list[str], doc: dict) -> float:
        s = 0.0
        for t in q:
            tf = doc["_tf"].get(t, 0)
            if tf:
                denom = tf + self.k1 * (1 - self.b + self.b * doc["_len"] / self.avg_len)
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
        return s

    def run(self, args: dict, role: str) -> ToolResult:
        q = list(dict.fromkeys(content_tokens(args["query"])))
        visible = [d for d in self.docs if role in d["acl"]]
        ranked = sorted(((self._score(q, d), d) for d in visible),
                        key=lambda x: (-x[0], x[1]["id"]))
        top = [(s, d) for s, d in ranked[: args.get("top_k", self.top_k)] if s > 0]
        items = [{"doc_id": d["id"], "title": d["title"], "score": round(s, 3),
                  "text": d["text"]} for s, d in top]
        if not items:
            return ToolResult([], False, "no matching document visible to caller")
        best = top[0][1]
        coverage = sum(1 for t in q if t in best["_tf"]) / len(q)
        if top[0][0] < self.min_score or coverage < self.min_cov:
            return ToolResult(items, False,
                              f"weak match score={top[0][0]:.2f} coverage={coverage:.2f}")
        return ToolResult(items, True, f"top={best['id']} coverage={coverage:.2f}")


# --------------------------------------------------------------------------- directory

DIRECTORY_FIELDS = ("owner_team", "owner_name", "oncall_channel", "cost_center",
                    "escalation_email", "personal_phone")


class DirectoryTool:
    """Exact and close-match lookup of system ownership records."""

    spec = ToolSpec(
        name="directory_lookup",
        route="directory",
        description="Look up who owns a system, its on-call channel, cost center or escalation contact.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["entity", "field"],
            "properties": {
                "entity": {"type": "string", "minLength": 2, "maxLength": 80},
                "field": {"type": "string", "enum": list(DIRECTORY_FIELDS)},
            },
        },
        scope="directory",
    )

    def __init__(self, path: Path, match_cutoff: float = 0.85) -> None:
        raw = json.loads(Path(path).read_text())
        self.restricted = set(raw.get("restricted_fields", []))
        self.cutoff = match_cutoff
        self.by_key: dict[str, dict] = {}
        for ent in raw["entities"]:
            for key in [ent["id"], *ent.get("aliases", [])]:
                self.by_key[normalise(key)] = ent

    def resolve(self, name: str) -> dict | None:
        key = normalise(name)
        if key in self.by_key:
            return self.by_key[key]
        close = difflib.get_close_matches(key, list(self.by_key), n=1, cutoff=self.cutoff)
        return self.by_key[close[0]] if close else None

    def check_args(self, args: dict) -> None:
        return None

    def required_scopes(self, args: dict) -> set[str]:
        scopes = {self.spec.scope}
        if args["field"] in self.restricted:
            scopes.add("directory:restricted")
        return scopes

    def run(self, args: dict, role: str) -> ToolResult:
        ent = self.resolve(args["entity"])
        if ent is None:
            return ToolResult([], False, "entity not found in directory")
        item = {"entity": ent["id"], "field": args["field"], "value": ent[args["field"]]}
        return ToolResult([item], True, f"resolved {ent['id']}")


# --------------------------------------------------------------------------- registry

class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self.by_route = {t.spec.route: t for t in tools}
        self.by_name = {t.spec.name: t for t in tools}
        if len(self.by_route) != len(tools) or len(self.by_name) != len(tools):
            raise ValueError("duplicate tool route or name")

    def routes(self) -> list[str]:
        return list(self.by_route)

    def get(self, route: str) -> Tool:
        return self.by_route[route]


def build_registry(data_dir: Path, cfg: dict,
                   metrics_fault: Callable[[], None] | None = None) -> ToolRegistry:
    data_dir = Path(data_dir)
    ps = cfg["policy_search"]
    return ToolRegistry([
        MetricsTool(data_dir / "orders.csv", cfg["data_start_month"], cfg["data_end_month"],
                    fault=metrics_fault),
        PolicySearchTool(data_dir / "docs.jsonl", ps["top_k"], ps["min_score"],
                         ps["min_term_coverage"]),
        DirectoryTool(data_dir / "directory.json"),
    ])
