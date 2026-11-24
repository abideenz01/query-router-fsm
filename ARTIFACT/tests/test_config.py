import json

import pytest

from query_router.config import load_config


def test_defaults_load(cfg):
    assert cfg["budgets"]["max_tool_calls"] >= 1
    assert "analyst" in cfg["roles"]


def test_override_deep_merges(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"routing": {"min_margin": 2.5}}))
    cfg = load_config(p)
    assert cfg["routing"]["min_margin"] == 2.5
    assert cfg["routing"]["max_subtasks"] == 2  # untouched key survives


def test_env_override(tmp_path, monkeypatch):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"budgets": {"max_retries": 3}}))
    monkeypatch.setenv("ROUTER_CONFIG", str(p))
    assert load_config()["budgets"]["max_retries"] == 3


def test_fallback_cycle_rejected(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"fallbacks": {"directory": ["policy"], "policy": ["directory"]}}))
    with pytest.raises(ValueError, match="cycle"):
        load_config(p)


def test_invalid_values_rejected(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"routing": {"min_confidence": 0}}))
    with pytest.raises(ValueError):
        load_config(p)


def test_missing_override_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/override.json")
