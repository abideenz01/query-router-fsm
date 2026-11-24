import json

import pytest

from conftest import ROOT
from query_router.evaluate import (CostModel, FanOutStrategy, RouterStrategy, VectorOnlyStrategy,
                                   canonical_metric_args, evaluate, load_queries, render_markdown)

QUERIES = ROOT / "examples" / "eval_queries.jsonl"


def test_canonical_args_ignore_group_order():
    a = {"metric": "orders", "period_start": "2026-01", "period_end": "2026-02", "group_by": ["month", "region"]}
    b = dict(a, group_by=["region", "month"])
    assert canonical_metric_args(a) == canonical_metric_args(b)


def test_loader_rejects_bad_rows(tmp_path):
    p = tmp_path / "q.jsonl"
    row = {"id": "a", "role": "analyst", "query": "x", "gold": {"outcome": "ANSWERED", "parts": []}}
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="needs parts"):
        load_queries(p)
    row["gold"] = {"outcome": "CLARIFY"}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_queries(p)


@pytest.fixture()
def report(cfg, registry, router, proposer):
    roles = {k: set(v) for k, v in cfg["roles"].items()}
    strategies = [RouterStrategy(router), VectorOnlyStrategy(registry, roles),
                  FanOutStrategy(registry, proposer, roles)]
    return evaluate(load_queries(QUERIES), strategies, CostModel(cfg))


def test_regression_guard_on_synthetic_set(report):
    r = report["strategies"]
    assert r["router"]["outcome_accuracy"] >= 0.85
    assert r["router"]["outcome_accuracy"] > r["fan_out"]["outcome_accuracy"]
    assert r["router"]["outcome_accuracy"] > r["vector_only"]["outcome_accuracy"]
    assert r["router"]["answered_when_it_should_not"] <= 1


def test_report_matches_committed_expected_output(report):
    expected = json.loads((ROOT / "examples" / "expected_eval_report.json").read_text())
    for name, s in expected["strategies"].items():
        assert report["strategies"][name]["outcome_accuracy"] == s["outcome_accuracy"]
        assert report["strategies"][name]["usd_per_1k_queries"] == s["usd_per_1k_queries"]


def test_markdown_renders(report):
    md = render_markdown(report)
    assert "outcome_accuracy" in md and "router" in md
