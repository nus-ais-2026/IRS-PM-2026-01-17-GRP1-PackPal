"""
Evaluate KG layering logic – correctness for known temperature bands.

Tests that the fallback layering advice contains expected keywords
for five distinct temperature ranges.

Output: accuracy (number of correct temperature bands out of 5).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kg_rules
from models import DayForecast

test_temps = [
    (-10, ["Heavy winter coat", "Insulated boots"]),
    (5,   ["Heavy winter coat", "Warm sweater"]),
    (15,  ["Light jacket", "Long-sleeve"]),
    (25,  ["T-shirt", "Shorts"]),
    (35,  ["Lightweight", "Shorts"]),
]

correct = 0
for temp, expected_keywords in test_temps:
    fc = DayForecast("2026-01-01", 0, 0, temp, temp + 5, 0, 0, 0)
    advice = kg_rules.recommend_layering([fc], 0.5)
    if all(kw.lower() in advice.lower() for kw in expected_keywords):
        correct += 1

print(f"KG Layering Accuracy: {correct}/{len(test_temps)}")