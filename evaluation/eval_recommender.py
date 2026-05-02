"""
Evaluate clothing recommender – mean accuracy and F1 of KNN model.

Trains on a subset of synthetic data and tests on a held‑out split.

Output: mean accuracy and mean F1 across all 37 labels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from recommender import _generate_training_data, ALL_ITEMS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score

# Generate a small dataset
X, Y = _generate_training_data(n_samples=500, seed=123)
split = 400
X_train, Y_train = X.iloc[:split], Y[:split]
X_test,  Y_test  = X.iloc[split:], Y[split:]

# Train KNN model
model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, weights="distance"))
model.fit(X_train, Y_train)

# Predict on test set
preds = model.predict(X_test)

# Compute metrics per label, then average
accs = []
f1s = []
for i in range(Y_test.shape[1]):
    accs.append(accuracy_score(Y_test[:, i], preds[:, i]))
    f1s.append(f1_score(Y_test[:, i], preds[:, i], zero_division=0))

mean_acc = np.mean(accs)
mean_f1  = np.mean(f1s)

print(f"KNN Recommender – mean accuracy: {mean_acc:.3f}, mean F1: {mean_f1:.3f}")