"""
Evaluate packing optimizer – constraint satisfaction rate.

Generates 50 random trip configurations and verifies that the knapsack
solver returns a set of items whose total weight does not exceed the weight limit.

Output: success rate as a percentage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import numpy as np
from packing_optimizer import _knapsack_select, _item_weight, _item_comfort_score
from models import DayForecast, TripContext
from recommender import ALL_ITEMS

random.seed(123)
np.random.seed(123)

trials = 50
successes = 0
for _ in range(trials):
    # Random trip: 3-10 days, random purpose
    days = random.randint(3, 10)
    temp = random.uniform(-5, 35)
    forecasts = [DayForecast("2026-01-01", 5, 50, temp, temp+8,
                             random.uniform(0, 15), random.uniform(0, 45), 0)
                 for _ in range(days)]
    ctx = TripContext(random.choice(["business", "tourism", "visiting"]),
                      "TestCity", "TestCountry")
    # Select 10-20 random items
    n_items = random.randint(10, 20)
    items = random.sample(ALL_ITEMS, n_items)
    weights = np.array([_item_weight(it) for it in items])
    weight_limit = sum(weights) * random.uniform(0.4, 1.0)
    comfort = np.array([_item_comfort_score(it, forecasts, ctx) for it in items])

    selected = _knapsack_select(items, comfort, weights, weight_limit)
    total_w = sum(_item_weight(it) for it in selected)
    if total_w <= weight_limit + 0.01:   # tiny tolerance for rounding
        successes += 1

rate = successes / trials * 100
print(f"Optimizer constraint satisfaction rate: {successes}/{trials} ({rate:.1f}%)")