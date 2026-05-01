# Automated Test Suite (Pytest)

This directory contains unit tests for the core reasoning modules of **PackPal**.
The tests validate the logical correctness of individual functions, ensuring that
each AI component behaves as expected. They are designed to run quickly and do not
require any external services or API keys.

## Purpose in the Project

The automated test suite serves several purposes:

- **Correctness verification** – confirms that algorithmic building blocks
  (Theil‑Sen prediction, rule‑based label generation, knapsack solver, etc.)
  produce mathematically sound outputs.
- **Regression prevention** – future code changes can be validated against these
  tests to avoid introducing bugs.
- **Evaluation support** – the tests form part of the **Offline Ranking** and
  **Historical Simulation** evaluation methods described in the project report.
  While they do not compute predictive performance metrics like RMSE or Precision@K
  (those are in the `evaluation/` folder), they prove that the underlying logic
  is reliable.

## Relationship to the Project Report

The test results are referenced in **Section 6 (Evaluation)** and detailed in
**Appendix F (Automated Test Suite)** of the project report. Each test file
corresponds to the module it validates, and the pass/fail outcomes are
indicative of the system's robustness.

## How to Run

From the **project root** (where `streamlit_app.py` lives), execute:

    pip install pytest
    pytest tests/ --tb=short -v

The output will show a list of test names and a summary like `6 passed in 2.34s`.
If a test fails, the traceback will indicate the exact assertion that failed.

## Test File Descriptions

| File | Module(s) Tested | What It Validates | Evaluation Method | Metric |
|------|------------------|-------------------|-------------------|--------|
| `test_slot_detection.py` | `slot_detection.py` (fallback) | Correct extraction of destination, dates, purpose; missing slot reporting | Offline Ranking (Precision@1) | Correct / total assertions |
| `test_recommender.py` | `recommender.py` | Rule‑label shape, cold‑weather recommendation, feature shape, training data dimensions, model prediction | Historical Simulation / Offline Ranking | Pass/fail assertions |
| `test_kg_rules.py` | `kg_rules.py` (fallback) | CLO bounds for extreme temperatures, non‑empty layering advice | Offline simulation (temperature bands) | Pass/fail assertions |
| `test_historical_forecast.py` | `historical_forecast.py` (helpers) | Exponential weight sum, Theil‑Sen finiteness, weather‑code integer return | Historical simulation (synthetic data) | Pass/fail assertions |
| `test_packing_optimizer.py` | `packing_optimizer.py` (knapsack) | Weight‑constraint compliance, empty‑list handling | Constraint satisfaction (simulated trips) | Pass/fail assertions |
| `test_xai.py` | `xai_explain.py`, `lime_explainer.py` | LIME background data shape/columns, SHAP output keys (when model available) | Unit test / sanity check | Pass/fail assertions |

### `test_slot_detection.py`

- **Purpose:** Validate the regex‑based fallback extraction used when the Groq LLM is unavailable.
- **Key tests:**
  - Extracts destination from capitalised words (e.g., "Tokyo").
  - Parses YYYY‑MM‑DD dates for start and end.
  - Recognises tourism‑related keywords and maps them to `purpose="tourism"`.
  - Correctly reports missing slots when information is incomplete.
  - Always returns a `TripSlots` object, even for empty input.
- **Evaluation method:** Offline Ranking (Precision@1 on canned utterances).

### `test_recommender.py`

- **Purpose:** Test the clothing recommendation engine's rule‑based label generation and the KNN model training pipeline.
- **Key tests:**
  - `_rule_labels()` returns a binary vector of length 38 (the full item vocabulary).
  - Cold‑weather logic recommends `Heavy winter coat`.
  - `_features()` returns a (1, 10) DataFrame.
  - `_generate_training_data()` produces the correct dimensions.
  - A fresh KNN model can be trained on 100 samples and produce predictions.
- **Evaluation method:** Historical Simulation / Offline Ranking (supports accuracy metrics by proving training pipeline works).

### `test_kg_rules.py`

- **Purpose:** Verify the Knowledge Graph fallback logic for CLO calculation and layering advice.
- **Key tests:**
  - `calculate_base_weather_clo()` returns high CLO (≥0.8) for sub‑zero temperatures.
  - Returns low CLO (≤0.3) for very hot temperatures.
  - `recommend_layering()` returns a non‑empty string containing contextual advice.
- **Note:** These tests use the fallback path; they do not require a live Neo4j connection.
- **Evaluation method:** Offline simulation against known temperature bands.

### `test_historical_forecast.py`

- **Purpose:** Test the numerical helpers used in weather prediction.
- **Key tests:**
  - Exponential weights decay correctly and sum to 1.
  - `_predict_continuous()` with Theil‑Sen returns a finite number.
  - `_predict_code()` returns an integer weather code.
- **Evaluation method:** Historical simulation (synthetic data).

### `test_packing_optimizer.py`

- **Purpose:** Validate the knapsack solver's constraint satisfaction.
- **Key tests:**
  - The solver selects items that stay within the weight limit.
  - An empty item list yields an empty selection.
- **Evaluation method:** Constraint satisfaction (simulated trips).

### `test_xai.py`

- **Purpose:** Sanity checks for the explainability modules.
- **Key tests:**
  - LIME background data has 1000 rows and the correct column names.
  - SHAP explanation (when the personalisation model is available) returns a dictionary with expected keys.
- **Evaluation method:** Unit test / sanity check.

## Interpreting Results

- **All tests passing** indicates that the core logic is healthy and ready for
  evaluation.
- **A test failure** should be investigated immediately—it may indicate a regression
  caused by a recent code change or a mismatch between the model and the current
  item vocabulary (e.g., stale `.joblib` files).

## Dependencies

- pytest
- numpy, pandas, scikit‑learn, lightgbm (already in the project's `requirements.txt`)