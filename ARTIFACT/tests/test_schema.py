from query_router.schema import SchemaError, validate

import pytest

S = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {
        "a": {"type": "string", "enum": ["x", "y"]},
        "n": {"type": "integer", "minimum": 1, "maximum": 5},
        "m": {"type": "string", "pattern": r"20[0-9]{2}-(0[1-9]|1[0-2])"},
        "g": {"type": "array", "items": {"type": "string", "enum": ["p", "q"]},
              "maxItems": 2, "uniqueItems": True},
    },
}


def test_valid_instance_has_no_errors():
    assert validate({"a": "x", "n": 3, "m": "2026-04", "g": ["p"]}, S) == []


def test_missing_required_and_unexpected_field():
    errs = validate({"b": 1}, S)
    assert any("a: required" in e for e in errs)
    assert any("b: unexpected" in e for e in errs)


def test_enum_pattern_and_bounds():
    errs = validate({"a": "z", "n": 9, "m": "2026-13"}, S)
    assert len(errs) == 3


def test_boolean_is_not_an_integer():
    assert validate({"a": "x", "n": True}, S)


def test_array_items_max_and_unique():
    assert validate({"a": "x", "g": ["p", "p"]}, S)
    assert validate({"a": "x", "g": ["p", "q", "p"]}, S)
    assert validate({"a": "x", "g": ["r"]}, S)


def test_wrong_top_level_type():
    assert validate([], S) == ["$: expected object, got list"]


def test_unsupported_type_is_a_schema_error():
    with pytest.raises(SchemaError):
        validate(1, {"type": "decimal"})
