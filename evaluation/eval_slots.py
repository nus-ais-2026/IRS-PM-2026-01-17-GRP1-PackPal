"""
Evaluate slot detection – Precision@1 on a small test set.

Runs the regex‑based fallback extraction on known utterances.

Output: Precision@1 (how often the extracted slot matches expected value).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_detection import _fallback_extraction

test_cases = [
    ("I'm going to Tokyo from 2026-08-11 to 2026-08-20 for tourism",
     {"destination": "Tokyo", "start_date": "2026-08-11", "end_date": "2026-08-20", "purpose": "tourism"}),
    ("Business trip to Singapore 2026-05-01 2026-05-05",
     {"destination": "Singapore", "start_date": "2026-05-01", "end_date": "2026-05-05", "purpose": "business"}),
    ("Visiting family in London next month",
     {"destination": "London", "purpose": "visiting"}),
]

correct = 0
total = 0
for utterance, expected in test_cases:
    result = _fallback_extraction([], utterance)
    for field, exp_val in expected.items():
        total += 1
        if getattr(result, field) == exp_val:
            correct += 1

precision = correct / total if total else 0.0
print(f"Slot Detection Precision@1 (fallback): {precision:.2f} ({correct}/{total})")