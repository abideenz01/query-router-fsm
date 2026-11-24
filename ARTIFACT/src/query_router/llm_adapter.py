"""Optional LLM proposer for OpenAI-compatible Chat Completions tool calling.

The model sees the tool contracts plus two control tools (ask_clarification,
out_of_scope) and returns tool calls. Every call is converted to a SubTask and
then goes through the same validation, authorization and transition rules as
the heuristic proposer. Nothing the model returns is executed directly.

Configuration (environment variables, nothing is hard-coded):
  ROUTER_LLM_URL          full chat completions URL
  ROUTER_LLM_MODEL        model or deployment name
  ROUTER_LLM_API_KEY      secret, read at call time, never logged
  ROUTER_LLM_AUTH_HEADER  "Authorization" (default, sends "Bearer <key>") or "api-key"
  ROUTER_LLM_TIMEOUT_S    request timeout, default 20

Only the request and response shapes are exercised by the test suite, through
an injected transport. Validate against your own endpoint before relying on it.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from .proposers import Plan, SubTask
from .tools import ToolRegistry

SYSTEM_PROMPT = (
    "You route enterprise search questions to tools. Call exactly one tool per distinct "
    "question in the user message, at most {max_subtasks}. Use metrics_query only for numbers "
    "that come from governed metrics. Use policy_search for rules, processes and documents. "
    "Use directory_lookup for system ownership and contacts. If the question is ambiguous or "
    "lacks a required detail, call ask_clarification. If no tool fits, call out_of_scope. "
    "Never invent metric names, months or fields outside the tool schemas."
)

CONTROL_TOOLS = [
    {"type": "function", "function": {
        "name": "ask_clarification",
        "description": "Ask the user one short question when intent or a required detail is unclear.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["question"],
                       "properties": {"question": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "out_of_scope",
        "description": "The request is not answerable by any available tool.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["reason"],
                       "properties": {"reason": {"type": "string"}}}}},
]

Transport = Callable[[str, dict, dict, float], dict]


def _urllib_transport(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL from config
        return json.loads(resp.read().decode())


def tool_definitions(registry: ToolRegistry) -> list[dict]:
    defs = [{"type": "function", "function": {
        "name": t.spec.name, "description": t.spec.description,
        "parameters": t.spec.parameters}} for t in registry.by_name.values()]
    return defs + CONTROL_TOOLS


class OpenAICompatibleProposer:
    def __init__(self, registry: ToolRegistry, cfg: dict,
                 transport: Transport = _urllib_transport, env: dict | None = None) -> None:
        self.registry = registry
        self.max_subtasks = cfg["routing"]["max_subtasks"]
        self.transport = transport
        self.env = env if env is not None else os.environ
        self.url = self._need("ROUTER_LLM_URL")
        self.model = self._need("ROUTER_LLM_MODEL")
        self.header = self.env.get("ROUTER_LLM_AUTH_HEADER", "Authorization")
        self.timeout = float(self.env.get("ROUTER_LLM_TIMEOUT_S", "20"))

    def _need(self, key: str) -> str:
        val = self.env.get(key)
        if not val:
            raise RuntimeError(f"missing environment variable {key}")
        return val

    def _headers(self) -> dict:
        key = self._need("ROUTER_LLM_API_KEY")
        if self.header.lower() == "authorization":
            return {"Authorization": f"Bearer {key}"}
        return {self.header: key}

    def build_request(self, query: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(max_subtasks=self.max_subtasks)},
                {"role": "user", "content": query},
            ],
            "tools": tool_definitions(self.registry),
            "tool_choice": "required",
            "temperature": 0,
        }

    def propose(self, query: str) -> Plan:
        response = self.transport(self.url, self._headers(), self.build_request(query), self.timeout)
        return self.parse(response, query)

    def parse(self, response: dict, query: str) -> Plan:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return Plan("clarify", [], "malformed model response")
        calls = message.get("tool_calls") or []
        if not calls:
            return Plan("clarify", [], "model returned no tool call")
        subtasks: list[SubTask] = []
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": True}
            if not isinstance(args, dict):
                args = {"_non_object_arguments": True}
            if name == "out_of_scope":
                return Plan("out_of_scope", [], "model: out of scope")
            if name == "ask_clarification":
                return Plan("clarify", [], "model asked for clarification")
            tool = self.registry.by_name.get(name)
            route = tool.spec.route if tool else f"unknown:{name}"
            subtasks.append(SubTask(route, args, query, 1.0, [f"proposed by model as {name}"]))
        if len(subtasks) > self.max_subtasks:
            return Plan("clarify", [], f"model proposed {len(subtasks)} calls, limit {self.max_subtasks}")
        return Plan("route", subtasks, f"model proposed {len(subtasks)} call(s)")
