import pytest


@pytest.mark.parametrize("q,route", [
    ("What was total revenue in Q1 2026?", "metrics"),
    ("What is the refund policy for licenses?", "policy"),
    ("Who owns the billing-api?", "directory"),
    ("How does the on call rotation work?", "policy"),
])
def test_single_intent_routes(proposer, q, route):
    plan = proposer.propose(q)
    assert plan.decision == "route" and plan.subtasks[0].route == route


@pytest.mark.parametrize("q,decision", [
    ("Tell me about refunds", "clarify"),
    ("How are we doing in KSA?", "clarify"),
    ("Translate good morning into Urdu", "out_of_scope"),
])
def test_non_routable(proposer, q, decision):
    assert proposer.propose(q).decision == decision


@pytest.mark.parametrize("text,period", [
    ("revenue in q2", ("2026-04", "2026-06")),
    ("revenue h1 2026", ("2026-01", "2026-06")),
    ("revenue last quarter", ("2026-04", "2026-06")),
    ("revenue last month", ("2026-06", "2026-06")),
    ("revenue between april and june", ("2026-04", "2026-06")),
    ("revenue in march", ("2026-03", "2026-03")),
])
def test_period_parsing(proposer, text, period):
    args, _ = proposer.extract("metrics", text)
    assert (args["period_start"], args["period_end"]) == period


def test_may_as_modal_verb_is_not_a_month(proposer):
    args, notes = proposer.extract("metrics", "revenue that may be off")
    assert args["period_start"] == "2026-01" and any("assumed" in n for n in notes)


def test_missing_metric_is_left_out_for_validation(proposer):
    args, _ = proposer.extract("metrics", "numbers for Cloud in Q1")
    assert "metric" not in args and args["product_line"] == "Cloud"


def test_compound_query_splits(proposer):
    plan = proposer.propose("What is the refund rate in UAE for Q2 and who owns the billing-api?")
    assert [s.route for s in plan.subtasks] == ["metrics", "directory"]
    assert plan.subtasks[0].args["region"] == "UAE"


def test_too_many_parts_asks_for_clarification(proposer):
    plan = proposer.propose("What was revenue in Q1? Who owns the CRM? How do I request VPN access?")
    assert plan.decision == "clarify" and "max_subtasks" in plan.reason


def test_restricted_field_extraction(proposer):
    args, _ = proposer.extract("directory", "mobile number for the owner of the data lake")
    assert args == {"entity": "data lake", "field": "personal_phone"}
