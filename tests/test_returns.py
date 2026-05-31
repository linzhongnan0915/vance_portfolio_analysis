"""Return calculation sanity checks."""

from __future__ import annotations

import numpy as np

from src.preprocessing import prices_to_returns
from tests.fixtures import make_synthetic_prices


def test_simple_returns_first_row_dropped(synthetic_prices):
    rets = prices_to_returns(synthetic_prices)
    assert rets.index[0] > synthetic_prices.index[0]


def test_returns_finite_and_bounded(synthetic_prices):
    rets = prices_to_returns(synthetic_prices)
    assert np.isfinite(rets.values).all()
    assert rets.abs().max().max() < 0.5, "Daily returns should be <50% for synthetic fixtures"
