"""
Unit tests for the clothing recommender.

Tests rule‑based label generation, feature extraction shape,
training data generation, and that the KNN model can be trained
and used for prediction.

Usage:
    pytest tests/test_recommender.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from recommender import _rule_labels, _features, _generate_training_data, ALL_ITEMS
from models import DayForecast, TripContext


def test_rule_labels_output_shape():
    """_rule_labels should return a binary vector with len(ALL_ITEMS) elements."""
    labels = _rule_labels(10, 5, 15, 0.5, 20, 5, 50, 0, 0, 1)
    assert len(labels) == len(ALL_ITEMS)
    assert set(np.unique(labels)).issubset({0.0, 1.0})


def test_rule_labels_cold_weather():
    """Cold weather must recommend a heavy winter coat."""
    labels = _rule_labels(-5, -10, 0, 0.0, 0, 0, 0, 0, 0, 1)
    idx = ALL_ITEMS.index("Heavy winter coat")
    assert labels[idx] == 1.0


def test_features_output_shape():
    """_features should return a (1, 10) DataFrame."""
    fc = DayForecast(date="2026-08-11", uv_index_max=5, cloud_cover_mean=50,
                     temp_min=20, temp_max=30, precipitation_mm=2,
                     wind_speed_max=15, weather_code=0)
    feat = _features(fc, "tourism")
    assert feat.shape == (1, 10)


def test_training_data_generation():
    """_generate_training_data should produce (n_samples, 10) X and (n_samples, |ALL_ITEMS|) Y."""
    n = 100
    X, Y = _generate_training_data(n_samples=n, seed=42)
    assert X.shape == (n, 10)
    assert Y.shape == (n, len(ALL_ITEMS))


def test_model_trains_and_predicts():
    """KNN model should train on 100 samples and return valid predictions."""
    from recommender import train_and_save, _load_or_train

    # Train a tiny model for speed
    train_and_save(verbose=False, model_type="knn")

    fc = DayForecast(date="2026-08-11", uv_index_max=5, cloud_cover_mean=50,
                     temp_min=20, temp_max=30, precipitation_mm=2,
                     wind_speed_max=15, weather_code=0)
    feat = _features(fc, "tourism")
    model = _load_or_train("knn")
    pred = model.predict(feat)
    assert pred.shape == (1, len(ALL_ITEMS))