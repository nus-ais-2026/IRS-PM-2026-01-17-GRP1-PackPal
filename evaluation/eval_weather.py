"""
Evaluate weather forecasting – Theil‑Sen vs OLS RMSE on synthetic data.

Simulates 10 years of historical observations with a linear trend +
noise, withholds one year, and compares predictions.

Output: RMSE for Theil‑Sen and OLS, plus error variance reduction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from historical_forecast import _predict_continuous

rng = np.random.default_rng(42)
years = np.arange(2015, 2025, dtype=float)       # 10 historical years
true_slope = 0.03                                 # gentle warming trend
true_intercept = 25.0
true_values = true_intercept + true_slope * (years - 2015)

# Add realistic noise
noise = rng.normal(0, 1.5, size=len(years))
obs = true_values + noise

# Withhold 2024 as test year, predict using 2015-2023
train_idx = years < 2024
test_year = 2024.0
actual_2024 = obs[years == test_year][0] if test_year in years else np.interp(test_year, years, obs)

pred_theil = _predict_continuous(years[train_idx], obs[train_idx], test_year, method="theil_sen")
pred_ols   = _predict_continuous(years[train_idx], obs[train_idx], test_year, method="ewm_ols")

rmse_theil = np.sqrt((pred_theil - actual_2024)**2)
rmse_ols   = np.sqrt((pred_ols - actual_2024)**2)

print(f"Theil‑Sen prediction: {pred_theil:.2f}, actual: {actual_2024:.2f}, RMSE: {rmse_theil:.3f}")
print(f"OLS prediction:     {pred_ols:.2f}, actual: {actual_2024:.2f}, RMSE: {rmse_ols:.3f}")
if rmse_ols > 0:
    reduction = (1 - rmse_theil / rmse_ols) * 100
    print(f"Error variance reduction: {reduction:.1f}%")
else:
    print("Error variance reduction: n/a (OLS RMSE is zero)")