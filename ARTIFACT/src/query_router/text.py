"""Shared text normalisation used by search and routing."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the is are was were be been of to in on for and or with by at from as
    what whats which who whom how do does did can could should would will i we our
    us my me you your it its this that these those there here any all about tell
    please much many get show give need want know
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def _stem(tok: str) -> str:
    """Very light plural stripping; enough for a small English corpus."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def normalise(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))
