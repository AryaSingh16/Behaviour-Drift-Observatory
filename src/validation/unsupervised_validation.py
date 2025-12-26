import pandas as pd
import numpy as np

# -------------------------------------------------
# Load drift scores produced by BDO
# -------------------------------------------------
drift_scores = pd.read_csv("data/drift_scores.csv")
"""
Expected columns:
user_id, day, drift_score
"""

# -------------------------------------------------
# Synthetic ground-truth drift injection point
# (known only to us, NOT used by the model)
# -------------------------------------------------
INJECTED_DRIFT_DAY = 60

# -------------------------------------------------
# Stability validation (pre-drift)
# -------------------------------------------------
pre_drift = drift_scores[drift_scores["day"] < INJECTED_DRIFT_DAY]

pre_drift_mean = pre_drift.groupby("user_id")["drift_score"].mean()
pre_drift_std = pre_drift.groupby("user_id")["drift_score"].std()

print("\n--- Stability Check (Before Drift Injection) ---")
print(f"Mean drift score (should be low): {pre_drift_mean.mean():.4f}")
print(f"Std deviation (should be small): {pre_drift_std.mean():.4f}")

# -------------------------------------------------
# Sensitivity validation (post-drift)
# -------------------------------------------------
post_drift = drift_scores[drift_scores["day"] >= INJECTED_DRIFT_DAY]

post_drift_mean = post_drift.groupby("user_id")["drift_score"].mean()

print("\n--- Sensitivity Check (After Drift Injection) ---")
print(f"Mean drift score after injection: {post_drift_mean.mean():.4f}")

# -------------------------------------------------
# Onset alignment validation
# -------------------------------------------------
detected_onsets = []

for user_id, df in drift_scores.groupby("user_id"):
    df = df.sort_values("day")
    above = df["drift_score"] > 0.15

    for i in range(len(above) - 3):
        if above.iloc[i:i+3].all():
            detected_onsets.append(df.iloc[i]["day"])
            break

detected_onsets = pd.Series(detected_onsets)

print("\n--- Drift Onset Alignment ---")
print(detected_onsets.describe())
print(f"Expected onset ≈ day {INJECTED_DRIFT_DAY}")
