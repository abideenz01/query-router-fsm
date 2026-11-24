import csv
import json

import pytest

from conftest import DATA
from query_router.schema import validate
from query_router.tools import ArgumentError, PermanentToolError, PolicySearchTool


def _manual(metric_fn, **flt):
    rows = list(csv.DictReader((DATA / "orders.csv").open()))
    sel = [r for r in rows if all(r[k] == v for k, v in flt.items() if k not in ("lo", "hi"))
           and flt["lo"] <= r["month"] <= flt["hi"]]
    return metric_fn(sel)


def test_metrics_match_hand_computation(registry):
    t = registry.get("metrics")
    res = t.run({"metric": "revenue", "period_start": "2026-01", "period_end": "2026-03",
                 "region": "UAE"}, "analyst")
    expected = _manual(lambda s: sum(int(r["revenue_usd"]) for r in s), region="UAE",
                       lo="2026-01", hi="2026-03")
    assert res.sufficient and res.items == [{"value": expected}]


def test_refund_rate_is_ratio_of_sums(registry):
    t = registry.get("metrics")
    res = t.run({"metric": "refund_rate", "period_start": "2026-04", "period_end": "2026-06"},
                "analyst")
    exp = _manual(lambda s: round(sum(int(r["refunds"]) for r in s) / sum(int(r["orders"]) for r in s), 4),
                  lo="2026-04", hi="2026-06")
    assert res.items[0]["value"] == exp


def test_group_by_returns_one_row_per_group(registry):
    t = registry.get("metrics")
    res = t.run({"metric": "orders", "period_start": "2026-01", "period_end": "2026-06",
                 "group_by": ["region"]}, "analyst")
    assert [i["region"] for i in res.items] == sorted(["UAE", "KSA", "Pakistan", "UK"])


def test_sql_uses_placeholders_for_values(registry):
    sql, params = registry.get("metrics").build_sql(
        {"metric": "orders", "period_start": "2026-01", "period_end": "2026-02", "region": "UAE"})
    assert "UAE" not in sql and params == ["2026-01", "2026-02", "UAE"]


def test_injection_in_filter_is_rejected_by_contract(registry):
    t = registry.get("metrics")
    bad = {"metric": "orders", "period_start": "2026-01", "period_end": "2026-02",
           "region": "UAE' OR '1'='1"}
    assert validate(bad, t.spec.parameters)


def test_metrics_semantic_checks(registry):
    t = registry.get("metrics")
    with pytest.raises(ArgumentError, match="after"):
        t.check_args({"metric": "orders", "period_start": "2026-05", "period_end": "2026-02"})
    with pytest.raises(ArgumentError, match="outside"):
        t.check_args({"metric": "orders", "period_start": "2026-07", "period_end": "2026-09"})


def test_policy_acl_hides_restricted_document(registry):
    t = registry.get("policy")
    q = {"query": "How are salary bands reviewed?"}
    assert not t.run(q, "analyst").sufficient
    hr = t.run(q, "hr")
    assert hr.sufficient and hr.items[0]["doc_id"] == "pol-comp-bands"
    assert all(i["doc_id"] != "pol-comp-bands" for i in t.run(q, "analyst").items)


def test_policy_weak_match_is_insufficient(registry):
    res = registry.get("policy").run({"query": "policy on bringing pets to the office"}, "analyst")
    assert not res.sufficient


def test_policy_rejects_stopword_only_query(registry):
    with pytest.raises(ArgumentError):
        registry.get("policy").check_args({"query": "what is the"})


def test_policy_malformed_doc_file(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"id": "x", "text": "t"}) + "\n")
    with pytest.raises(PermanentToolError, match="missing"):
        PolicySearchTool(p)


def test_directory_alias_fuzzy_and_unknown(registry):
    d = registry.get("directory")
    assert d.resolve("Billing API")["id"] == "billing-api"
    assert d.resolve("paymnets gateway")["id"] == "payments-gateway"
    assert d.resolve("fax gateway") is None


def test_directory_restricted_field_requires_extra_scope(registry):
    d = registry.get("directory")
    assert d.required_scopes({"entity": "crm", "field": "owner_name"}) == {"directory"}
    assert "directory:restricted" in d.required_scopes({"entity": "crm", "field": "personal_phone"})
