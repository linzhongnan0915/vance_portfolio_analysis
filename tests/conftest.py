"""Shared pytest fixtures (offline, deterministic)."""

from __future__ import annotations

import pytest

from tests.fixtures import make_synthetic_daily_returns, make_synthetic_prices


@pytest.fixture
def synthetic_prices():
    return make_synthetic_prices()


@pytest.fixture
def synthetic_daily_returns():
    return make_synthetic_daily_returns()
