"""Minimal JSON Schema subset validator (stdlib only).

Supports the keywords used by tool argument contracts: type, properties,
required, additionalProperties (bool), enum, pattern, minLength, maxLength,
minimum, maximum, items, maxItems, uniqueItems.

It exists so tool arguments are checked in code even when the model provider
claims schema adherence. Provider-side strict modes depend on the model and
request configuration; this check does not.
"""
from __future__ import annotations

import re
from typing import Any

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


class SchemaError(ValueError):
    """Raised for an invalid schema definition (developer error)."""


def validate(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable violations. Empty list means valid."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        if expected not in _TYPES:
            raise SchemaError(f"unsupported type {expected!r} at {path}")
        py_type = _TYPES[expected]
        is_bool = isinstance(instance, bool)
        if expected in ("integer", "number") and is_bool:
            errors.append(f"{path}: expected {expected}, got boolean")
            return errors
        if not isinstance(instance, py_type):
            errors.append(f"{path}: expected {expected}, got {type(instance).__name__}")
            return errors

    if "enum" in schema and instance not in schema["enum"]:
        allowed = ", ".join(map(str, schema["enum"]))
        errors.append(f"{path}: {instance!r} is not one of [{allowed}]")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], instance):
            errors.append(f"{path}: {instance!r} does not match {schema['pattern']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    if isinstance(instance, list):
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and len(set(map(repr, instance))) != len(instance):
            errors.append(f"{path}: items must be unique")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}.{name}: required field missing")
        if schema.get("additionalProperties") is False:
            for name in instance:
                if name not in props:
                    errors.append(f"{path}.{name}: unexpected field")
        for name, sub in props.items():
            if name in instance:
                errors.extend(validate(instance[name], sub, f"{path}.{name}"))
    return errors
