"""
Historical Climate-Based Weather Predictor
==========================================
Refactored to use:
  - pandas  : store multi-year observations as DataFrames
  - numpy   : weighted mean (np.average), OLS trend (np.polyfit), std
  - pyarrow : cache each year's API response as Parquet on disk;
              subsequent runs load from cache instead of re-fetching

Algorithm (unchanged):
  1. Fetch same calendar window from past N years (archive API / parquet cache)
  2. Exponentially weighted mean — recent years weighted higher (α = 0.85)
  3. OLS linear trend via np.polyfit — extrapolated to target year
  4. Adaptive blend: trust trend less when variability >> trend signal
  5. UV proxy when satellite records are sparse (< 3 valid years)
  6. Weather code: exponentially weighted mode (categorical)
  7. Physical constraints clamp all values to valid ranges

Prediction method options (pass `method=` to get_historical_forecast):
  "ewm_ols"   (default) Exponential weighted mean + OLS trend, adaptively blended.
  "holt_des"  Holt's Double Exponential Smoothing — principled level + trend
              decomposition with separate smoothing params (alpha, beta).
  "theil_sen" Theil-Sen estimator — median of all pairwise slopes, robust to
              outlier years (e.g. El Niño, volcanic winters).
  "gpr"       Gaussian Process Regression (sklearn Matern kernel) — non-linear
              trend with built-in noise handling; slowest but most flexible.
"""

import urllib.request
import urllib.parse
import json
from datetime import date, timedelta
from calendar import isleap
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 — ensures parquet engine is available

from models import DayForecast

CACHE_DIR = Path(__file__).parent / "cache"
ALPHA = 0.85
TREND_WEIGHT = 0.25
MIN_OBS_FOR_TREND = 3

DAILY_VARS = [
    "uv_index_max",
    "cloud_cover_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "weather_code",
]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(lat: float, lon: float, year: int, mmdd_start: str, mmdd_end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{lat:.2f}_{lon:.2f}_{year}_{mmdd_start}_{mmdd_end}.parquet"


# ── Date helpers ──────────────────────────────────────────────────────────────

def _shift_year(d: date, target_year: int) -> date:
    if d.month == 2 and d.day == 29 and not isleap(target_year):
        return date(target_year, 2, 28)
    return date(target_year, d.month, d.day)


# ── API fetch ─────────────────────────────────────────────────────────────────

def _fetch_from_api(lat: float, lon: float, timezone: str,
                    start: str, end: str) -> pd.DataFrame:
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "timezone": timezone,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS),
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())
    daily = data.get("daily", {})
    if not daily.get("time"):
        return pd.DataFrame()
    return pd.DataFrame(daily).rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum":  "precipitation_mm",
        "wind_speed_10m_max": "wind_speed_max",
    })


def _fetch_year(lat: float, lon: float, timezone: str,
                hist_start: date, hist_end: date,
                target_start: date, target_end: date) -> pd.DataFrame:
    """Return DataFrame for one historical year, using parquet cache when available."""
    mmdd_start = target_start.strftime("%m%d")
    mmdd_end   = target_end.strftime("%m%d")
    cache_file = _cache_path(lat, lon, hist_start.year, mmdd_start, mmdd_end)

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    try:
        df = _fetch_from_api(lat, lon, timezone,
                             hist_start.isoformat(), hist_end.isoformat())
        if not df.empty:
            df.to_parquet(cache_file, index=False)
        return df
    except Exception:
        return pd.DataFrame()


# ── Data collection ───────────────────────────────────────────────────────────

def _collect(lat: float, lon: float, timezone: str,
             target_start: date, target_end: date,
             n_years: int) -> pd.DataFrame:
    """
    Fetch same calendar window from past n_years.
    Returns a DataFrame with columns:
      year, day_offset, temp_max, temp_min, precipitation_mm,
      wind_speed_max, uv_index_max, cloud_cover_mean, weather_code
    Rows are sorted by (day_offset, year desc).
    """
    target_year = target_start.year
    n_days = (target_end - target_start).days + 1
    frames = []

    for offset in range(1, n_years + 1):
        hist_year = target_year - offset
        if hist_year < 1940:
            break
        hist_start = _shift_year(target_start, hist_year)
        hist_end   = _shift_year(target_end,   hist_year)
        df = _fetch_year(lat, lon, timezone, hist_start, hist_end,
                         target_start, target_end)
        if df.empty or len(df) < n_days:
            continue
        df = df.head(n_days).copy()
        df["year"]       = hist_year
        df["day_offset"] = range(n_days)
        frames.append(df)

    if not frames:
        raise ValueError("Could not retrieve any historical data for this location.")

    all_df = pd.concat(frames, ignore_index=True)
    years_fetched = sorted(all_df["year"].unique())
    print(f"  [Historical] Used {len(years_fetched)} years of data "
          f"({years_fetched[0]}–{years_fetched[-1]})")
    return all_df


# ── Prediction helpers (numpy-based) ─────────────────────────────────────────

def _exp_weights(n: int) -> np.ndarray:
    """Normalised exponential decay; index 0 = most recent year."""
    w = ALPHA ** np.arange(n, dtype=float)
    return w / w.sum()


def _predict_ewm_ols(y: np.ndarray, v: np.ndarray, target_year: int) -> float:
    """
    Exponentially weighted mean + OLS linear trend, adaptively blended.
    y/v sorted most-recent-first, NaNs already removed.
    """
    n = len(v)
    w = _exp_weights(n)
    mu_w    = np.average(v, weights=w)
    sigma_w = np.sqrt(np.average((v - mu_w) ** 2, weights=w))

    if n >= MIN_OBS_FOR_TREND:
        slope, intercept = np.polyfit(y.astype(float), v, 1)
        std_v = np.std(v)
        slope = np.clip(slope, -2 * std_v, 2 * std_v)
        x_trend = intercept + slope * target_year

        trend_signal      = abs(slope * 5)
        trend_reliability = trend_signal / (trend_signal + sigma_w + 1e-9)
        eff_tw = TREND_WEIGHT * min(1.0, trend_reliability)
    else:
        x_trend = mu_w
        eff_tw  = 0.0

    return float((1.0 - eff_tw) * mu_w + eff_tw * x_trend)


def _predict_holt_des(y: np.ndarray, v: np.ndarray, target_year: int,
                      alpha: float = 0.85, beta: float = 0.15) -> float:
    """
    Holt's Double Exponential Smoothing.
    Maintains a level and a trend state, each with its own decay parameter.
    More principled than ewm_ols for level+trend decomposition.
    alpha: smoothing for level (high = trust recent observations more)
    beta:  smoothing for trend (low = smooth out year-to-year slope noise)
    """
    n = len(v)
    if n == 1:
        return float(v[0])

    # Reverse to chronological order (oldest → newest) for the forward pass
    v_chron = v[::-1]
    y_chron = y[::-1]

    level = v_chron[0]
    trend = v_chron[1] - v_chron[0]

    for val in v_chron[1:]:
        prev_level = level
        level = alpha * val + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend

    # Forecast h steps ahead from the last observed year
    h = target_year - int(y_chron[-1])
    return float(level + h * trend)


def _predict_theil_sen(y: np.ndarray, v: np.ndarray, target_year: int) -> float:
    """
    Theil-Sen estimator: slope = median of all pairwise slopes.
    Robust to outlier years (e.g. El Niño, volcanic winters) — up to ~29%
    of observations can be outliers without distorting the result.
    Pure numpy, no extra dependency.
    """
    n = len(v)
    if n == 1:
        return float(v[0])

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dy = float(y[j] - y[i])
            if dy != 0:
                slopes.append((v[j] - v[i]) / dy)

    if not slopes:
        return float(np.median(v))

    slope     = float(np.median(slopes))
    intercept = float(np.median(v) - slope * np.median(y))
    return intercept + slope * target_year


def _predict_gpr(y: np.ndarray, v: np.ndarray, target_year: int) -> float:
    """
    Gaussian Process Regression with a Matern(nu=1.5) + WhiteKernel.
    Handles small datasets (5-15 points) without overfitting, and naturally
    accounts for year-to-year noise via WhiteKernel.
    Slower (~5-15 ms per call) but most flexible; good for non-linear trends.
    Requires scikit-learn (already in requirements.txt).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    n = len(v)
    if n == 1:
        return float(v[0])

    X = y.reshape(-1, 1).astype(float)
    kernel = Matern(length_scale=5.0, nu=1.5) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, normalize_y=True
    )
    gpr.fit(X, v)
    pred, _ = gpr.predict([[float(target_year)]], return_std=True)
    return float(pred[0])


_PREDICT_METHODS = {
    "ewm_ols":   _predict_ewm_ols,
    "holt_des":  _predict_holt_des,
    "theil_sen": _predict_theil_sen,
    "gpr":       _predict_gpr,
}


def _predict_continuous(years: np.ndarray, values: np.ndarray,
                        target_year: int,
                        method: str = "theil_sen") -> float:
    """
    Dispatch to the selected prediction method.
    years/values must be sorted most-recent-first.
    """
    mask = ~np.isnan(values)
    if not mask.any():
        return 0.0
    y, v = years[mask], values[mask]

    fn = _PREDICT_METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {list(_PREDICT_METHODS)}"
        )
    return fn(y, v, target_year)


def _predict_code(years: np.ndarray, codes: np.ndarray) -> int:
    """Exponentially weighted mode — calmer code wins ties."""
    mask = ~np.isnan(codes)
    if not mask.any():
        return 0
    y, c = years[mask], codes[mask].astype(int)
    w = _exp_weights(len(c))
    score: dict = {}
    for wi, ci in zip(w, c):
        score[ci] = score.get(ci, 0.0) + wi
    return min(score, key=lambda x: (-score[x], x))


def _uv_proxy(lat: float, cloud: float) -> float:
    base = max(0.0, 12.0 - abs(lat) * 0.15)
    return base * (1.0 - 0.7 * (cloud / 100.0))


# ── Public API ────────────────────────────────────────────────────────────────

def get_historical_forecast(lat: float, lon: float, timezone: str,
                             start_date: str, end_date: str,
                             n_years: int = 10,
                             method: str = "theil_sen") -> list:
    """
    Predict daily weather for a future date range from past climate data.
    Uses parquet cache to avoid re-fetching on repeated runs.
    Returns list[DayForecast].

    method: prediction algorithm for continuous variables (temp, precip, etc.)
      "ewm_ols"   (default) Exponential weighted mean + OLS trend blend
      "holt_des"  Holt's Double Exponential Smoothing (level + trend)
      "theil_sen" Theil-Sen robust linear trend (outlier-resistant)
      "gpr"       Gaussian Process Regression (non-linear, slowest)
    """
    target_start = date.fromisoformat(start_date)
    target_end   = date.fromisoformat(end_date)
    target_year  = target_start.year
    n_days       = (target_end - target_start).days + 1

    print(f"  [Historical] Fetching up to {n_years} years of archive data...")
    df = _collect(lat, lon, timezone, target_start, target_end, n_years)

    forecasts = []
    for day_i in range(n_days):
        day_df = (
            df[df["day_offset"] == day_i]
            .sort_values("year", ascending=False)
            .reset_index(drop=True)
        )
        years = day_df["year"].to_numpy(dtype=float)

        def col(name):
            return day_df[name].to_numpy(dtype=float) if name in day_df else np.array([np.nan])

        temp_max = _predict_continuous(years, col("temp_max"),         target_year, method)
        temp_min = _predict_continuous(years, col("temp_min"),         target_year, method)
        cloud    = _predict_continuous(years, col("cloud_cover_mean"), target_year, method)
        precip   = _predict_continuous(years, col("precipitation_mm"), target_year, method)
        wind     = _predict_continuous(years, col("wind_speed_max"),   target_year, method)
        code     = _predict_code(years,       col("weather_code"))

        uv_vals = col("uv_index_max")
        valid_uv = uv_vals[~np.isnan(uv_vals)]
        if len(valid_uv) >= MIN_OBS_FOR_TREND:
            uv = _predict_continuous(years, uv_vals, target_year, method)
        else:
            uv = _uv_proxy(lat, cloud)

        # Physical constraints
        temp_max = float(np.clip(temp_max, -80, 60))
        temp_min = float(np.clip(temp_min, -80, 60))
        if temp_max < temp_min + 0.5:
            temp_max = temp_min + 0.5
        cloud  = float(np.clip(cloud,  0, 100))
        precip = float(max(0.0, precip))
        wind   = float(max(0.0, wind))
        uv     = float(np.clip(uv, 0, 16))

        current_date = (target_start + timedelta(days=day_i)).isoformat()
        forecasts.append(DayForecast(
            date=current_date,
            uv_index_max=round(uv, 1),
            cloud_cover_mean=round(cloud, 1),
            temp_min=round(temp_min, 1),
            temp_max=round(temp_max, 1),
            precipitation_mm=round(precip, 1),
            wind_speed_max=round(wind, 1),
            weather_code=int(code),
        ))

    return forecasts
