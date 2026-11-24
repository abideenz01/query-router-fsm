"""Explicit state machine that owns every routing hop.

The proposer suggests a route and arguments. This module decides what happens
next, and only along edges listed in TRANSITIONS. An edge that is not in the
table raises IllegalTransition, so an unexpected hop is a bug you can test for,
not behaviour you discover in production.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from . import schema
from .proposers import Plan, Proposer, SubTask
from .tools import ArgumentError, PermanentToolError, ToolRegistry, ToolResult, TransientToolError

log = logging.getLogger("query_router")


class State(str, Enum):
    RECEIVED = "RECEIVED"
    ROUTING = "ROUTING"
    VALIDATING = "VALIDATING"
    AUTHORIZING = "AUTHORIZING"
    EXECUTING = "EXECUTING"
    RETRYING = "RETRYING"
    CHECKING = "CHECKING"
    FALLBACK = "FALLBACK"
    ANSWERED = "ANSWERED"
    CLARIFY = "CLARIFY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    REFUSED = "REFUSED"
    NO_ANSWER = "NO_ANSWER"
    FAILED = "FAILED"


S = State
TERMINAL = frozenset({S.ANSWERED, S.CLARIFY, S.OUT_OF_SCOPE, S.REFUSED, S.NO_ANSWER, S.FAILED})
TRANSITIONS: dict[State, frozenset[State]] = {
    S.RECEIVED: frozenset({S.ROUTING, S.FAILED}),
    S.ROUTING: frozenset({S.VALIDATING, S.CLARIFY, S.OUT_OF_SCOPE, S.FAILED}),
    S.VALIDATING: frozenset({S.AUTHORIZING, S.CLARIFY, S.FAILED}),
    S.AUTHORIZING: frozenset({S.EXECUTING, S.REFUSED, S.FAILED}),
    S.EXECUTING: frozenset({S.CHECKING, S.RETRYING, S.FAILED}),
    S.RETRYING: frozenset({S.EXECUTING, S.FAILED}),
    S.CHECKING: frozenset({S.ANSWERED, S.FALLBACK, S.NO_ANSWER, S.FAILED}),
    S.FALLBACK: frozenset({S.VALIDATING, S.FAILED}),
}

# How to build arguments for a fallback route from the original sub-question.
FALLBACK_ARGS: dict[str, Callable[[SubTask], dict]] = {
    "policy": lambda st: {"query": st.text.strip()[:300]},
}


class IllegalTransition(RuntimeError):
    pass


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class SubtaskRun:
    text: str
    route_path: list[str]
    args: dict
    state: State = S.RECEIVED
    reason: str = ""
    items: list[dict] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self, include_items: bool = True) -> dict:
        d = {
            "status": self.state.value, "reason": self.reason,
            "route_path": self.route_path, "args": self.args,
            "confidence": round(self.confidence, 3), "notes": self.notes,
            "tool_calls": self.tool_calls, "transitions": self.transitions,
        }
        if include_items:
            d["items"] = self.items
        return d


@dataclass
class RouteResult:
    request_id: str
    role: str
    status: str
    plan_decision: str
    plan_reason: str
    scores: dict
    subtasks: list[SubtaskRun]

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.tool_calls) for s in self.subtasks)

    def to_dict(self, include_items: bool = True) -> dict:
        return {
            "request_id": self.request_id, "role": self.role, "status": self.status,
            "plan": {"decision": self.plan_decision, "reason": self.plan_reason,
                     "scores": self.scores},
            "subtasks": [s.to_dict(include_items) for s in self.subtasks],
        }


class _Budget:
    def __init__(self, max_tool_calls: int, max_transitions: int) -> None:
        self.tool_calls_left = max_tool_calls
        self.transitions_left = max_transitions


class QueryRouter:
    """Runs a Plan through the state machine, one sub-machine per sub-question."""

    def __init__(self, registry: ToolRegistry, proposer: Proposer, cfg: dict,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 log_query_text: bool = False) -> None:
        self.registry, self.proposer, self.cfg = registry, proposer, cfg
        self.sleep, self.clock = sleep, clock
        self.log_query_text = log_query_text
        self.roles: dict[str, set[str]] = {k: set(v) for k, v in cfg["roles"].items()}
        self.fallbacks: dict[str, list[str]] = cfg["fallbacks"]
        b = cfg["budgets"]
        self.max_tool_calls, self.max_transitions = b["max_tool_calls"], b["max_transitions"]
        self.max_retries = b["max_retries"]

    # ------------------------------------------------------------------ public
    def handle(self, query: str, role: str, request_id: str | None = None) -> RouteResult:
        rid = request_id or uuid.uuid4().hex[:12]
        qsha = _sha(query)
        t0 = self.clock()
        budget = _Budget(self.max_tool_calls, self.max_transitions)
        ctx = {"rid": rid, "qsha": qsha, "t0": t0, "budget": budget}
        if self.log_query_text:
            log.info(json.dumps({"event": "query", "request_id": rid, "text": query}))

        if role not in self.roles:
            run = SubtaskRun(query, [], {})
            self._move(run, S.ROUTING, "received", ctx)
            self._move(run, S.FAILED, f"unknown role {role!r}", ctx)
            return self._result(rid, role, Plan("route", [], "unknown role"), [run])

        try:
            plan = self.proposer.propose(query)
        except Exception as exc:  # proposer outage must not crash the caller
            run = SubtaskRun(query, [], {})
            self._move(run, S.ROUTING, "received", ctx)
            self._move(run, S.FAILED, f"proposer error {type(exc).__name__}", ctx)
            return self._result(rid, role, Plan("route", [], "proposer error"), [run])

        if plan.decision != "route" or not plan.subtasks:
            run = SubtaskRun(query, [], {})
            self._move(run, S.ROUTING, "received", ctx)
            target = S.OUT_OF_SCOPE if plan.decision == "out_of_scope" else S.CLARIFY
            self._move(run, target, plan.reason, ctx)
            return self._result(rid, role, plan, [run])

        runs = [self._run_subtask(st, role, ctx) for st in plan.subtasks]
        return self._result(rid, role, plan, runs)

    # ------------------------------------------------------------------ core loop
    def _run_subtask(self, st: SubTask, role: str, ctx: dict) -> SubtaskRun:
        run = SubtaskRun(st.text, [st.route], dict(st.args), notes=list(st.notes),
                         confidence=st.confidence)
        self._move(run, S.ROUTING, "received", ctx)
        self._move(run, S.VALIDATING, f"route={st.route}", ctx)
        tried = {st.route}
        attempt = 0
        result: ToolResult | None = None

        while run.state not in TERMINAL:
            route = run.route_path[-1]
            if run.state is S.VALIDATING:
                ok, why = self._validate(route, run.args)
                self._move(run, S.AUTHORIZING if ok else S.CLARIFY, why, ctx)
            elif run.state is S.AUTHORIZING:
                tool = self.registry.get(route)
                missing = tool.required_scopes(run.args) - self.roles[role]
                if missing:
                    self._move(run, S.REFUSED, f"role lacks scope {sorted(missing)}", ctx)
                else:
                    attempt = 0
                    self._move(run, S.EXECUTING, "authorized", ctx)
            elif run.state is S.EXECUTING:
                if ctx["budget"].tool_calls_left <= 0:
                    self._move(run, S.FAILED, "tool call budget exhausted", ctx)
                    continue
                ctx["budget"].tool_calls_left -= 1
                attempt += 1
                tool = self.registry.get(route)
                call = {"tool": tool.spec.name, "args_sha": _sha(run.args), "attempt": attempt}
                try:
                    result = tool.run(run.args, role)
                    call["ok"] = True
                    run.tool_calls.append(call)
                    self._move(run, S.CHECKING, result.reason, ctx)
                except TransientToolError as exc:
                    call["ok"], call["error"] = False, type(exc).__name__
                    run.tool_calls.append(call)
                    if attempt <= self.max_retries:
                        self._move(run, S.RETRYING, f"transient error attempt={attempt}", ctx)
                    else:
                        self._move(run, S.FAILED, "retries exhausted", ctx)
                except (PermanentToolError, Exception) as exc:  # noqa: BLE001
                    call["ok"], call["error"] = False, type(exc).__name__
                    run.tool_calls.append(call)
                    self._move(run, S.FAILED, f"tool error {type(exc).__name__}", ctx)
            elif run.state is S.RETRYING:
                self.sleep(min(2.0, 0.2 * 2 ** (attempt - 1)))
                self._move(run, S.EXECUTING, "retry", ctx)
            elif run.state is S.CHECKING:
                assert result is not None
                if result.sufficient:
                    run.items = result.items
                    self._move(run, S.ANSWERED, "sufficient evidence", ctx)
                    continue
                nxt = next((r for r in self.fallbacks.get(route, [])
                            if r not in tried and r in FALLBACK_ARGS), None)
                if nxt is None:
                    run.items = []
                    self._move(run, S.NO_ANSWER, f"insufficient: {result.reason}", ctx)
                else:
                    tried.add(nxt)
                    run.route_path.append(nxt)
                    run.args = FALLBACK_ARGS[nxt](st)
                    self._move(run, S.FALLBACK, f"{route} insufficient, trying {nxt}", ctx)
            elif run.state is S.FALLBACK:
                self._move(run, S.VALIDATING, f"route={run.route_path[-1]}", ctx)
            else:  # pragma: no cover - guarded by TRANSITIONS
                raise IllegalTransition(f"no handler for {run.state}")
        return run

    def _validate(self, route: str, args: dict) -> tuple[bool, str]:
        if route not in self.registry.by_route:
            return False, f"unknown tool or route {route!r}"
        tool = self.registry.get(route)
        errors = schema.validate(args, tool.spec.parameters)
        if errors:
            return False, "invalid arguments: " + "; ".join(errors[:4])
        try:
            tool.check_args(args)
        except ArgumentError as exc:
            return False, f"invalid arguments: {exc}"
        return True, "arguments valid"

    # ------------------------------------------------------------------ helpers
    def _move(self, run: SubtaskRun, to: State, reason: str, ctx: dict) -> None:
        if run.state in TERMINAL:
            raise IllegalTransition(f"{run.state.value} is terminal")
        if to not in TRANSITIONS[run.state]:
            raise IllegalTransition(f"{run.state.value} -> {to.value} is not allowed")
        budget: _Budget = ctx["budget"]
        if budget.transitions_left <= 0 and to is not S.FAILED:
            to, reason = S.FAILED, "transition budget exhausted"
        budget.transitions_left -= 1
        rec = {"from": run.state.value, "to": to.value, "reason": reason,
               "t_ms": round((self.clock() - ctx["t0"]) * 1000, 2)}
        run.transitions.append(rec)
        log.debug(json.dumps({"event": "transition", "request_id": ctx["rid"],
                              "query_sha": ctx["qsha"], **rec}))
        run.state, run.reason = to, reason

    @staticmethod
    def _result(rid: str, role: str, plan: Plan, runs: list[SubtaskRun]) -> RouteResult:
        states = [r.state for r in runs]
        if all(s is S.ANSWERED for s in states):
            status = "ANSWERED"
        elif any(s is S.ANSWERED for s in states):
            status = "PARTIAL"
        else:
            priority = [S.FAILED, S.REFUSED, S.CLARIFY, S.NO_ANSWER, S.OUT_OF_SCOPE]
            status = next(p.value for p in priority if p in states)
        log.info(json.dumps({"event": "routed", "request_id": rid, "role": role,
                             "status": status, "decision": plan.decision,
                             "subtasks": len(runs)}))
        return RouteResult(rid, role, status, plan.decision, plan.reason, plan.scores, runs)
