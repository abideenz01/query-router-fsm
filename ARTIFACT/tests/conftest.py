import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_router import HeuristicProposer, QueryRouter, build_registry, load_config  # noqa: E402

DATA = ROOT / "examples" / "data"


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def registry(cfg):
    return build_registry(DATA, cfg)


@pytest.fixture()
def proposer(cfg, registry):
    return HeuristicProposer(cfg, list(registry.get("directory").by_key))


@pytest.fixture()
def router(cfg, registry, proposer):
    return QueryRouter(registry, proposer, cfg, sleep=lambda s: None)
