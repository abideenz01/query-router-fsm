import json

import pytest

from query_router import QueryRouter
from query_router.llm_adapter import OpenAICompatibleProposer

ENV = {"ROUTER_LLM_URL": "https://llm.example.invalid/v1/chat/completions",
       "ROUTER_LLM_MODEL": "router-model", "ROUTER_LLM_API_KEY": "sk-test"}


def _resp(*calls):
    return {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}}]}


def _proposer(registry, cfg, response, seen=None, env=ENV):
    def transport(url, headers, body, timeout):
        if seen is not None:
            seen.update(url=url, headers=headers, body=body)
        return response
    return OpenAICompatibleProposer(registry, cfg, transport=transport, env=env)


def test_request_carries_tool_contracts(registry, cfg):
    seen = {}
    p = _proposer(registry, cfg, _resp(("out_of_scope", {"reason": "x"})), seen)
    p.propose("hello")
    names = [t["function"]["name"] for t in seen["body"]["tools"]]
    assert {"metrics_query", "policy_search", "directory_lookup",
            "ask_clarification", "out_of_scope"} == set(names)
    assert seen["headers"] == {"Authorization": "Bearer sk-test"}
    assert seen["body"]["model"] == "router-model"


def test_api_key_header_variant(registry, cfg):
    seen = {}
    env = {**ENV, "ROUTER_LLM_AUTH_HEADER": "api-key"}
    _proposer(registry, cfg, _resp(("out_of_scope", {"reason": "x"})), seen, env).propose("q")
    assert seen["headers"] == {"api-key": "sk-test"}


def test_missing_env_raises(registry, cfg):
    with pytest.raises(RuntimeError, match="ROUTER_LLM_URL"):
        OpenAICompatibleProposer(registry, cfg, transport=lambda *a: {}, env={})


def test_parallel_tool_calls_become_subtasks_and_execute(registry, cfg):
    resp = _resp(("metrics_query", {"metric": "orders", "period_start": "2026-01", "period_end": "2026-03"}),
                 ("directory_lookup", {"entity": "crm", "field": "owner_name"}))
    router = QueryRouter(registry, _proposer(registry, cfg, resp), cfg)
    res = router.handle("orders in q1 and who owns crm", "analyst")
    assert res.status == "ANSWERED" and [s.route_path[-1] for s in res.subtasks] == ["metrics", "directory"]


@pytest.mark.parametrize("call,status", [
    (("metrics_query", "{not json"), "CLARIFY"),
    (("metrics_query", {"metric": "orders"}), "CLARIFY"),
    (("delete_everything", {}), "CLARIFY"),
    (("ask_clarification", {"question": "Which region?"}), "CLARIFY"),
    (("out_of_scope", {"reason": "weather"}), "OUT_OF_SCOPE"),
    (("directory_lookup", {"entity": "crm", "field": "personal_phone"}), "REFUSED"),
])
def test_model_output_is_still_governed(registry, cfg, call, status):
    router = QueryRouter(registry, _proposer(registry, cfg, _resp(call)), cfg)
    assert router.handle("q", "analyst").status == status


def test_malformed_response_and_no_calls(registry, cfg):
    assert _proposer(registry, cfg, {"oops": 1}).propose("q").decision == "clarify"
    empty = {"choices": [{"message": {"content": "I think it is revenue"}}]}
    assert _proposer(registry, cfg, empty).propose("q").decision == "clarify"


def test_too_many_calls_is_clarify(registry, cfg):
    call = ("policy_search", {"query": "vpn access"})
    assert _proposer(registry, cfg, _resp(call, call, call)).propose("q").decision == "clarify"
