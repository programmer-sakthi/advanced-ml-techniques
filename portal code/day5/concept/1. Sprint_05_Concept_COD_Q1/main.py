import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. File Input and Loading
# -------------------------------
try:
    data = pd.read_csv(os.path.join(sys.path[0], input().strip()))
except:
    print("Error: Unable to read file.")
    sys.exit(1)

# -------------------------------
# 2. Data Validation
# -------------------------------
if "target" not in data.columns:
    print("Error: Target column 'target' not found.")
    sys.exit(1)

# Separate features and target
X = data.iloc[:, :-1].values
y = data["target"].values

# -------------------------------
# 3. Cross-Validation Setup
# -------------------------------
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# -------------------------------
# 4. Cross-Validation Execution
# -------------------------------
all_preds = []
all_true = []

for train_idx, test_idx in rskf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, oob_score=True, n_jobs=-1, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    all_preds.append(preds)
    all_true.append(y_test)

# -------------------------------
# 5. Accuracy Calculation
# -------------------------------
all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)

accuracy = accuracy_score(all_true, all_preds)

# -------------------------------
# 6. OOB Score (Full Dataset)
# -------------------------------
final_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, oob_score=True, n_jobs=-1, random_state=42
)

final_model.fit(X, y)
oob_score = final_model.oob_score_

# -------------------------------
# 7. Output
# -------------------------------
print("=================================")
print(f"Accuracy: {accuracy:.3f}")
print(f"OOB Score: {oob_score:.3f}")
print("=================================")
