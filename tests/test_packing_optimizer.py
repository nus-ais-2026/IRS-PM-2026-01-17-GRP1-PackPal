"""
Unit tests for the packing optimizer.

Tests the knapsack solver under simple weight constraints.

Usage:
    pytest tests/test_packing_optimizer.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from packing_optimizer import _knapsack_select


def test_knapsack_basic():
    """Should select light items and exclude heavy ones."""
    items = ["a", "b", "c"]
    comfort = np.array([0.9, 0.5, 0.7])
    weights = np.array([2.0, 1.0, 3.0])
    selected = _knapsack_select(items, comfort, weights, weight_limit=4.0)
    # 'a' (2kg) + 'b' (1kg) = 3kg → fits; 'c' alone is 3kg but less comfort
    assert "a" in selected
    assert "b" in selected
    assert "c" not in selected


def test_knapsack_empty():
    """Empty item list should return an empty selection."""
    selected = _knapsack_select([], np.array([]), np.array([]), 10.0)
    assert selected == []