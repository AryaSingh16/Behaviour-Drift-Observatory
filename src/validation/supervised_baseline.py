import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------------------------------
# Load behavioral representations 
# -------------------------------------------------
df = pd.read_csv("data/behavior_representations.csv")

"""
Expected columns:
user_id
day
session_count_mean_14d
avg_session_duration_mean_14d
active_hours_entropy_mean_14d
action_type_entropy_mean_14d
inter_day_variability_mean_14d
"""

# -------------------------------------------------
# Synthetic labels (ONLY for supervised comparison)
# Rule: Drift if behavior changes significantly after day 60
# -------------------------------------------------
df["synthetic_drift_label"] = (
    (df["day"] > 60) &
    (
        df["inter_day_variability_mean_14d"] >
        df["inter_day_variability_mean_14d"].median()
    )
).astype(int)

# -------------------------------------------------
# Feature matrix
# -------------------------------------------------
FEATURE_COLS = [
    "session_count_mean_14d",
    "avg_session_duration_mean_14d",
    "active_hours_entropy_mean_14d",
    "action_type_entropy_mean_14d",
    "inter_day_variability_mean_14d",
]

X = df[FEATURE_COLS]
y = df["synthetic_drift_label"]

# -------------------------------------------------
# Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Train supervised model
# -------------------------------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate
# -------------------------------------------------
y_pred = clf.predict(X_test)

print("\n=== Supervised Drift Classifier (Baseline) ===\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# Feature importance (for contrast with BDO)
# -------------------------------------------------
importance = pd.Series(
    clf.coef_[0],
    index=FEATURE_COLS
).sort_values(ascending=False)

print("\nFeature importance (supervised):")
print(importance)
