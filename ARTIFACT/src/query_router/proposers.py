"""Route proposers.

A proposer turns a query into a Plan: route it, ask for clarification, or mark
it out of scope. Proposers only *propose*. Arguments are validated, scopes are
checked and transitions are enforced by the state machine.

HeuristicProposer is deterministic and needs no model. LLM-backed proposers
live in llm_adapter.py and return the same Plan type.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .text import content_tokens, normalise


@dataclass
class SubTask:
    route: str
    args: dict
    text: str
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Plan:
    decision: str  # "route" | "clarify" | "out_of_scope"
    subtasks: list[SubTask]
    reason: str
    scores: dict[str, float] = field(default_factory=dict)


class Proposer(Protocol):
    def propose(self, query: str) -> Plan: ...


class ScriptedProposer:
    """Returns pre-built plans. Used in tests and for replaying recorded decisions."""

    def __init__(self, plans: dict[str, Plan]) -> None:
        self.plans = plans

    def propose(self, query: str) -> Plan:
        if query not in self.plans:
            return Plan("out_of_scope", [], "no scripted plan")
        return self.plans[query]


# --------------------------------------------------------------------------- lexicons

_W = list[tuple[re.Pattern, float]]


def _c(pairs: list[tuple[str, float]]) -> _W:
    return [(re.compile(p), w) for p, w in pairs]


INTENT_CUES: dict[str, _W] = {
    "metrics": _c([
        (r"\b(revenue|sales|turnover|top line)\b", 3),
        (r"\brefund (rate|ratio|percentage)\b", 4),
        (r"\b(average order value|aov|basket size)\b", 4),
        (r"\b(how many|number of|count of|total) (orders|refunds)\b", 4),
        (r"\borders\b", 2),
        (r"\brefunds? (issued|processed|count)\b", 3),
        (r"\brefunds?\b", 1),
        (r"\b(by|per|each|across|which) (region|product|product line|month)s?\b", 1.5),
        (r"\b(monthly|trend|breakdown|month over month)\b", 1.5),
        (r"\b(q[1-4]|h1|h2|ytd|year to date|last quarter|last month|first half)\b", 1),
        (r"\b(total|sum|how much|compare|highest|lowest|best|worst)\b", 1),
        (r"\b(performance|how are we doing|numbers|kpis?)\b", 1),
    ]),
    "policy": _c([
        (r"\b(policy|policies|procedure|process|guideline|rules?|standard|runbook|checklist)\b", 3),
        (r"\b(how do i|how to|how can i|can i|am i allowed|allowed|required|need approval|approval|approve)\b", 2),
        (r"\b(how many days|how long|deadline|within how|what is the limit|cap|capped)\b", 2),
        (r"\b(how does|what happens|when does)\b", 1.5),
        (r"\b(vpn|leave|vacation|expenses?|hotel|travel|flights?|retention|retained|password|"
         r"laptop|vendors?|procurement|purchases?|severity|sev[123]|postmortem|pii|"
         r"personal data|salary|compensation|carry over|rotation|credentials?|meals?)\b", 1.5),
        (r"\brefunds?\b", 1),
    ]),
    "directory": _c([
        (r"\b(who owns|owner of|owned by|who is responsible|responsible for)\b", 3),
        (r"\b(owner|owns|which team)\b", 2),
        (r"\b(on[- ]?call channel|oncall|channel|escalat\w*|contact|cost cent(er|re)|"
         r"phone|mobile number|email)\b", 1.5),
    ]),
}

METRIC_PATTERNS = [
    ("refund_rate", r"\brefund (rate|ratio|percentage)\b"),
    ("avg_order_value", r"\b(average order value|aov|basket size)\b"),
    ("refunds", r"\b(how many|number of|count of|total) refunds\b|\brefunds? (issued|processed|count)\b"),
    ("orders", r"\borders\b"),
    ("revenue", r"\b(revenue|sales|turnover|top line)\b"),
]
GROUP_PATTERNS = [
    ("region", r"\b(by|per|each|across|which|every) regions?\b|\bregional breakdown\b"),
    ("product_line", r"\b(by|per|each|across|which|every) (product|product line)s?\b"),
    ("month", r"\b(by|per|each) month\b|\bmonthly\b|\btrend\b|\bmonth over month\b"),
]
REGION_ALIASES = {
    "uae": "UAE", "emirates": "UAE", "ksa": "KSA", "saudi": "KSA", "saudi arabia": "KSA",
    "pakistan": "Pakistan", "uk": "UK", "united kingdom": "UK", "britain": "UK",
}
PRODUCT_ALIASES = {"cloud": "Cloud", "support": "Support", "licenses": "Licenses",
                   "license": "Licenses", "licence": "Licenses", "licences": "Licenses"}
FIELD_PATTERNS = [
    ("personal_phone", r"\b(phone|mobile|cell)\b"),
    ("oncall_channel", r"\b(on[- ]?call|channel)\b"),
    ("cost_center", r"\bcost cent(er|re)\b"),
    ("escalation_email", r"\b(escalat\w*|email|contact)\b"),
    ("owner_team", r"\bwhich team\b|\bteam (owns|behind)\b"),
    ("owner_name", r"\b(who owns|owner|owns|responsible)\b"),
]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
_SPLIT = re.compile(
    r"\?\s*|;\s*|,?\s+and\s+(?=(?:who|what|which|how|where|when|is|are|does|do|can|also)\b)"
)


class HeuristicProposer:
    """Deterministic lexical router with argument extraction."""

    def __init__(self, cfg: dict, entity_keys: list[str]) -> None:
        r = cfg["routing"]
        self.min_conf, self.min_margin = r["min_confidence"], r["min_margin"]
        self.max_subtasks = r["max_subtasks"]
        self.as_of = cfg["as_of_month"]
        # longest aliases first so "payments gateway" wins over "payments"
        self.entity_keys = sorted({normalise(k) for k in entity_keys}, key=len, reverse=True)

    # ------------------------------------------------------------------ scoring
    def score(self, text: str) -> dict[str, float]:
        t = normalise(text)
        scores = {}
        for intent, cues in INTENT_CUES.items():
            scores[intent] = sum(w for pat, w in cues if pat.search(t))
        if self._find_entity(t):
            scores["directory"] += 3
        return scores

    def _decide(self, scores: dict[str, float]) -> tuple[str, str | None, float, str]:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        (top, s1), (_, s2) = ranked[0], ranked[1]
        total = sum(v for v in scores.values() if v > 0)
        if s1 == 0:
            return "out_of_scope", None, 0.0, "no routable intent detected"
        conf = s1 / total
        if s1 - s2 < self.min_margin or conf < self.min_conf:
            return "clarify", None, conf, f"ambiguous intent margin={s1 - s2:.1f} conf={conf:.2f}"
        return "route", top, conf, f"intent={top} conf={conf:.2f}"

    # ------------------------------------------------------------------ planning
    def propose(self, query: str) -> Plan:
        whole = self.score(query)
        fragments = [f.strip() for f in _SPLIT.split(query)
                     if f and f.strip() and len(content_tokens(f)) >= 2]
        if len(fragments) > 1:
            decided = [(f, self._decide(self.score(f))) for f in fragments]
            if all(d[0] == "route" for _, d in decided):
                if len(decided) > self.max_subtasks:
                    return Plan("clarify", [], f"{len(decided)} questions exceed max_subtasks", whole)
                subtasks = [self._subtask(route, f, conf) for f, (_, route, conf, _) in decided]
                return Plan("route", subtasks, f"compound query with {len(subtasks)} parts", whole)
        decision, route, conf, reason = self._decide(whole)
        if decision != "route":
            return Plan(decision, [], reason, whole)
        return Plan("route", [self._subtask(route, query, conf)], reason, whole)

    def extract(self, route: str, text: str) -> tuple[dict, list[str]]:
        """Build arguments for `route` from free text (also used by baselines)."""
        builder = {"metrics": self._metrics_args, "policy": self._policy_args,
                   "directory": self._directory_args}[route]
        return builder(text)

    def _subtask(self, route: str, text: str, conf: float) -> SubTask:
        args, notes = self.extract(route, text)
        return SubTask(route, args, text, conf, notes)

    # ------------------------------------------------------------------ extractors
    def _policy_args(self, text: str) -> tuple[dict, list[str]]:
        return {"query": text.strip()[:300]}, []

    def _find_entity(self, t: str) -> str | None:
        for key in self.entity_keys:
            if re.search(rf"\b{re.escape(key)}\b", t):
                return key
        return None

    def _directory_args(self, text: str) -> tuple[dict, list[str]]:
        t = normalise(text)
        entity = self._find_entity(t)
        notes = []
        if entity is None:
            m = re.search(r"\b(?:owns|owner of|responsible for|for|of)\s+(?:the\s+)?(.+)$", t)
            entity = (m.group(1) if m else t).strip()
            notes.append("entity not in alias list, passed through as free text")
        fld = next((f for f, p in FIELD_PATTERNS if re.search(p, t)), "owner_name")
        return {"entity": entity, "field": fld}, notes

    def _metrics_args(self, text: str) -> tuple[dict, list[str]]:
        t = normalise(text)
        args: dict = {}
        notes: list[str] = []
        metric = next((m for m, p in METRIC_PATTERNS if re.search(p, t)), None)
        if metric:
            args["metric"] = metric
        group = [g for g, p in GROUP_PATTERNS if re.search(p, t)]
        if group:
            args["group_by"] = group[:2]
        if "region" not in group:
            region = self._alias(t, REGION_ALIASES)
            if region:
                args["region"] = region
        if "product_line" not in group:
            product = self._alias(t, PRODUCT_ALIASES)
            if product:
                args["product_line"] = product
        period = self._period(t)
        if period is None:
            year = self.as_of[:4]
            end = _prev_month(self.as_of)
            period = (f"{year}-01", end if end[:4] == year else f"{year}-01")
            notes.append(f"no period given, assumed year to date {period[0]}..{period[1]}")
        args["period_start"], args["period_end"] = period
        return args, notes

    @staticmethod
    def _alias(t: str, table: dict[str, str]) -> str | None:
        for key in sorted(table, key=len, reverse=True):
            if re.search(rf"\b{re.escape(key)}\b", t):
                return table[key]
        return None

    def _period(self, t: str) -> tuple[str, str] | None:
        year_m = re.search(r"\b(20\d{2})\b", t)
        year = year_m.group(1) if year_m else self.as_of[:4]
        q = re.search(r"\bq([1-4])\b", t)
        if q:
            n = int(q.group(1))
            return f"{year}-{3 * n - 2:02d}", f"{year}-{3 * n:02d}"
        if re.search(r"\b(h1|first half)\b", t):
            return f"{year}-01", f"{year}-06"
        if re.search(r"\b(h2|second half)\b", t):
            return f"{year}-07", f"{year}-12"
        if "last quarter" in t:
            y, m = int(self.as_of[:4]), int(self.as_of[5:])
            cq = (m - 1) // 3  # 0-based current quarter
            pq, py = (cq - 1) % 4, y - (1 if cq == 0 else 0)
            return f"{py}-{3 * pq + 1:02d}", f"{py}-{3 * pq + 3:02d}"
        if "last month" in t:
            pm = _prev_month(self.as_of)
            return pm, pm
        if re.search(r"\b(ytd|year to date|this year|so far)\b", t):
            return f"{year}-01", _prev_month(self.as_of)
        found = []
        for i, name in enumerate(MONTHS, 1):
            if name == "may":  # "may" is also a modal verb; require a preposition
                hit = re.search(r"\b(in|of|during|for|to|and|from|between) may\b", t)
            else:
                hit = re.search(rf"\b({name}|{name[:3]})\b", t)
            if hit:
                found.append(i)
        if found:
            lo, hi = min(found), max(found)
            return f"{year}-{lo:02d}", f"{year}-{hi:02d}"
        return None


def _prev_month(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[5:])
    return f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"
