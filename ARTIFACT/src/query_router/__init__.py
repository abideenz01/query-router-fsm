"""Deterministic state-machine router for agentic enterprise search."""
from .config import load_config
from .fsm import QueryRouter, RouteResult, State, TRANSITIONS, IllegalTransition
from .proposers import HeuristicProposer, Plan, ScriptedProposer, SubTask
from .tools import build_registry

__all__ = ["load_config", "QueryRouter", "RouteResult", "State", "TRANSITIONS",
           "IllegalTransition", "HeuristicProposer", "Plan", "ScriptedProposer",
           "SubTask", "build_registry"]
__version__ = "0.1.0"
