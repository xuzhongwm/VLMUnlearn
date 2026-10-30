"""Milestone J as unit tests. GPU (Colab) only: needs the processed data and the vanilla adapter
of the config, exactly like scripts/check_hooks.py, and runs the same checks (vlmu/mech/checks.py).

  VLMU_TEST_CONFIG=configs/qwen2vl/grid.yaml [VLMU_TEST_DEBUG=1] python -m pytest -q -s tests/test_milestone_j.py

-s shows the J6 alignment printout (the patched tokens of one pair) for the manual check.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import checks  # noqa: E402


@pytest.fixture(scope="module")
def j():
    argv = ["--config", os.environ.get("VLMU_TEST_CONFIG", "configs/qwen2vl/grid.yaml")]
    if os.environ.get("VLMU_TEST_DEBUG"):
        argv.append("--debug")
    return checks.setup(resolve(base_parser(__doc__).parse_args(argv)))


@pytest.mark.parametrize("name", list(checks.CHECKS))
def test_milestone_j(j, name):
    res = checks.CHECKS[name](j)
    failed = {k: v for k, v in res.items() if not v["passed"]}
    assert res and not failed, failed
