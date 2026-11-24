import json
import logging

import pytest

from query_router import IllegalTransition, Plan, QueryRouter, ScriptedProposer, State, SubTask, TRANSITIONS
from query_router.fsm import TERMINAL, SubtaskRun, _Budget
from query_router.tools import PermanentToolError, TransientToolError, build_registry
from conftest import DATA


def _router(cfg, registry, plans, **kw):
    return QueryRouter(registry, ScriptedProposer(plans), cfg, sleep=lambda s: None, **kw)


def _states(sub):
    return [t["to"] for t in sub.transitions]


def test_every_state_can_reach_a_terminal_and_terminals_are_sinks():
    for s in State:
        if s in TERMINAL:
            assert s not in TRANSITIONS
            continue
        seen, stack = set(), [s]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(TRANSITIONS.get(cur, ()))
        assert seen & TERMINAL, s


def test_illegal_and_terminal_transitions_raise(router):
    ctx = {"rid": "r", "qsha": "q", "t0": 0.0, "budget": _Budget(5, 50)}
    run = SubtaskRun("x", ["policy"], {})
    with pytest.raises(IllegalTransition):
        router._move(run, State.EXECUTING, "skip validation", ctx)
    router._move(run, State.ROUTING, "", ctx)
    router._move(run, State.OUT_OF_SCOPE, "", ctx)
    with pytest.raises(IllegalTransition, match="terminal"):
        router._move(run, State.ROUTING, "", ctx)


def test_happy_path_transition_sequence(router):
    res = router.handle("Who owns the billing-api?", "analyst")
    assert res.status == "ANSWERED"
    assert _states(res.subtasks[0]) == ["ROUTING", "VALIDATING", "AUTHORIZING", "EXECUTING",
                                        "CHECKING", "ANSWERED"]
    assert res.subtasks[0].items[0]["value"] == "Sara Malik"


def test_fallback_runs_only_on_insufficient_result(router):
    res = router.handle("Who owns the legacy fax gateway?", "support")
    sub = res.subtasks[0]
    assert res.status == "ANSWERED" and sub.route_path == ["directory", "policy"]
    assert "FALLBACK" in _states(sub) and sub.items[0]["doc_id"] == "pol-legacy"


def test_refusal_never_falls_back(router):
    res = router.handle("What is the personal phone of the payments gateway owner?", "analyst")
    sub = res.subtasks[0]
    assert res.status == "REFUSED" and sub.route_path == ["directory"] and not sub.tool_calls


def test_hr_can_read_restricted_field(router):
    res = router.handle("What is the phone number for the billing-api owner?", "hr")
    assert res.status == "ANSWERED"


def test_invalid_model_arguments_become_clarify(cfg, registry):
    plans = {"q": Plan("route", [SubTask("metrics", {"metric": "ebitda", "period_start": "2026-01",
                                                      "period_end": "2026-02"}, "q")], "s")}
    res = _router(cfg, registry, plans).handle("q", "analyst")
    assert res.status == "CLARIFY" and "invalid arguments" in res.subtasks[0].reason
    assert not res.subtasks[0].tool_calls


def test_hallucinated_tool_is_caught(cfg, registry):
    plans = {"q": Plan("route", [SubTask("unknown:drop_table", {}, "q")], "s")}
    res = _router(cfg, registry, plans).handle("q", "analyst")
    assert res.status == "CLARIFY" and "unknown tool" in res.subtasks[0].reason


def test_unknown_role_fails_closed(router):
    res = router.handle("Who owns the billing-api?", "contractor")
    assert res.status == "FAILED"


def test_proposer_exception_fails_closed(cfg, registry):
    class Boom:
        def propose(self, q):
            raise TimeoutError("model down")
    res = QueryRouter(registry, Boom(), cfg).handle("q", "analyst")
    assert res.status == "FAILED" and "TimeoutError" in res.subtasks[0].reason


def _flaky(fail_times, exc=TransientToolError):
    state = {"n": 0}

    def fault():
        state["n"] += 1
        if state["n"] <= fail_times:
            raise exc("injected")
    return fault


METRIC_Q = "What was total revenue in Q1 2026?"


def _router_with_fault(cfg, proposer, fault, sleeps=None):
    reg = build_registry(DATA, cfg, metrics_fault=fault)
    return QueryRouter(reg, proposer, cfg,
                       sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_transient_error_is_retried_with_backoff(cfg, proposer):
    sleeps = []
    res = _router_with_fault(cfg, proposer, _flaky(1), sleeps).handle(METRIC_Q, "analyst")
    sub = res.subtasks[0]
    assert res.status == "ANSWERED" and "RETRYING" in _states(sub)
    assert [c["ok"] for c in sub.tool_calls] == [False, True] and sleeps == [0.2]


def test_retries_exhausted_fails(cfg, proposer):
    res = _router_with_fault(cfg, proposer, _flaky(5)).handle(METRIC_Q, "analyst")
    assert res.status == "FAILED" and res.subtasks[0].reason == "retries exhausted"


def test_permanent_error_is_not_retried(cfg, proposer):
    res = _router_with_fault(cfg, proposer, _flaky(1, PermanentToolError)).handle(METRIC_Q, "analyst")
    assert res.status == "FAILED" and len(res.subtasks[0].tool_calls) == 1


def test_tool_call_budget_is_shared_across_subtasks(cfg, registry, proposer):
    cfg["budgets"]["max_tool_calls"] = 1
    r = QueryRouter(registry, proposer, cfg, sleep=lambda s: None)
    res = r.handle("What is the refund rate in UAE for Q2 and who owns the billing-api?", "analyst")
    assert res.status == "PARTIAL"
    assert res.subtasks[1].reason == "tool call budget exhausted"


def test_transition_budget_forces_failure(cfg, registry, proposer):
    cfg["budgets"]["max_transitions"] = 3
    res = QueryRouter(registry, proposer, cfg).handle("Who owns the billing-api?", "analyst")
    assert res.status == "FAILED" and res.subtasks[0].reason == "transition budget exhausted"


def test_logs_omit_query_text_and_result_values(router, caplog):
    caplog.set_level(logging.DEBUG, logger="query_router")
    router.handle("What is the phone number for the billing-api owner?", "hr")
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "phone number" not in blob and "+000-0000-1001" not in blob
    assert all("query_sha" in json.loads(r.getMessage()) for r in caplog.records
               if json.loads(r.getMessage())["event"] == "transition")


def test_same_input_same_decisions(router):
    a = router.handle(METRIC_Q, "analyst").to_dict()
    b = router.handle(METRIC_Q, "analyst").to_dict()

    def strip(d):
        return [(t["from"], t["to"], t["reason"]) for s in d["subtasks"] for t in s["transitions"]]

    assert strip(a) == strip(b) and a["subtasks"][0]["items"] == b["subtasks"][0]["items"]
