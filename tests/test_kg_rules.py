"""
Unit tests for Knowledge Graph rules (fallback logic).

Tests CLO calculation and layering recommendations for
extreme temperature scenarios using the fallback path
(does not require a live Neo4j connection).

Usage:
    pytest tests/test_kg_rules.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import kg_rules
from models import DayForecast


def test_clo_cold_weather():
    """Sub‑zero temperatures should require high CLO."""
    forecasts = [
        DayForecast("2026-12-10", 2, 80, -5, 0, 0, 10, 0)
    ]
    clo = kg_rules.calculate_base_weather_clo(forecasts)
    assert clo >= 0.8


def test_clo_hot_weather():
    """Very hot weather should require low CLO."""
    forecasts = [
        DayForecast("2026-07-10", 10, 10, 30, 38, 0, 5, 0)
    ]
    clo = kg_rules.calculate_base_weather_clo(forecasts)
    assert clo <= 0.3


def test_layering_fallback_not_empty():
    """Layering fallback should return a non‑empty string for valid temperatures."""
    forecasts = [
        DayForecast("2026-12-10", 2, 80, -5, 0, 0, 10, 0)
    ]
    advice = kg_rules.recommend_layering(forecasts, 1.0)
    assert isinstance(advice, str)
    assert len(advice) > 0