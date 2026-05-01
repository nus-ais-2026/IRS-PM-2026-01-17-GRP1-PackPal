"""
Unit tests for historical weather prediction helpers.

Tests exponential weight calculation, Theil‑Sen prediction,
and weather‑code mode.

Usage:
    pytest tests/test_historical_forecast.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from historical_forecast import _predict_continuous, _predict_code, _exp_weights


def test_exp_weights():
    """Exponential weights should sum to 1 and decrease."""
    w = _exp_weights(5)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[-1]


def test_predict_continuous_theil_sen():
    """Theil‑Sen prediction must return a finite number."""
    years = np.arange(2015, 2025, dtype=float)
    values = np.array([30, 32, 28, 31, 33, 29, 30, 32, 31, 30], dtype=float)
    result = _predict_continuous(years, values, 2025, method="theil_sen")
    assert np.isfinite(result)


def test_predict_code():
    """Weather‑code prediction must return an integer."""
    years = np.arange(2015, 2025)
    codes = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    result = _predict_code(years, codes)
    assert isinstance(result, (int, np.integer))