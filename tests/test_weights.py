"""Weight normalization and mandate constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_WEIGHTS
from src.mandate_constraints import MandateConstraints, apply_mandate_constraints
from src.signals import make_weight_functions


def test_target_weights_sum_to_one():
    assert abs(TARGET_WEIGHTS.sum() - 1.0) < 1e-9


def test_make_weight_functions_fixed_long_only():
    tickers = list(TARGET_WEIGHTS.index)
    fns = make_weight_functions(tickers, TARGET_WEIGHTS)
    train = pd.DataFrame(0.001, index=pd.date_range("2020-01-01", periods=60), columns=tickers)
    w, _ = fns["fixed_baseline"](train)
    assert (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-9


def test_mandate_shy_cap_no_negative_weights():
    w = pd.Series({"QQQ": 0.1, "SPY": 0.1, "DIA": 0.1, "GLD": 0.1, "TLT": 0.1, "SHY": 0.5})
    mandate = MandateConstraints(shy_cap=0.10, min_equity_total=0.60)
    out, _ = apply_mandate_constraints(w, mandate)
    assert (out >= -1e-12).all()
    assert abs(out.sum() - 1.0) < 1e-6
    assert out["SHY"] <= 0.10 + 1e-6
