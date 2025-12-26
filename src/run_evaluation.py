import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
DRIFT_PATH = "data/drift_scores.csv"
EXPLAIN_PATH = "data/drift_explanations.csv"

# -----------------------------
# Basic validation
# -----------------------------
assert os.path.exists(DRIFT_PATH), "Missing drift_scores.csv"
assert os.path.exists(EXPLAIN_PATH), "Missing drift_explanations.csv"

print("✔ Required files found")

# -----------------------------
# Load data
# -----------------------------
drift_df = pd.read_csv(DRIFT_PATH)
explain_df = pd.read_csv(EXPLAIN_PATH)

print(f"✔ Drift scores shape: {drift_df.shape}")
print(f"✔ Drift explanations shape: {explain_df.shape}")

# =====================================================
# 1. STABILITY CHECK
# =====================================================
print("\n--- Stability Check (Low-drift users) ---")

stats = drift_df.groupby("user_id")["drift_score"].agg(["mean", "std"])
stable_users = stats.sort_values("mean").head(3).index.tolist()

plt.figure()
for user in stable_users:
    u = drift_df[drift_df["user_id"] == user]
    plt.plot(u["day"], u["drift_score"], label=user)

plt.title("Stability Test: Low Drift Users")
plt.xlabel("Day")
plt.ylabel("Drift Score")
plt.legend()
plt.show()

print("Expected: flat, near-zero drift curves")

# =====================================================
# 2. SENSITIVITY CHECK
# =====================================================
print("\n--- Sensitivity Check (High-drift users) ---")

high_users = (
    drift_df.groupby("user_id")["drift_score"]
    .max()
    .sort_values(ascending=False)
    .head(3)
    .index.tolist()
)

plt.figure()
for user in high_users:
    u = drift_df[drift_df["user_id"] == user]
    plt.plot(u["day"], u["drift_score"], label=user)

plt.title("Sensitivity Test: High Drift Users")
plt.xlabel("Day")
plt.ylabel("Drift Score")
plt.legend()
plt.show()

print("Expected: clear rise and sustained elevation")

# =====================================================
# 3. ONSET SANITY CHECK
# =====================================================
print("\n--- Drift Onset Sanity ---")

THRESHOLD = 0.15
onsets = []

for user, u_df in drift_df.groupby("user_id"):
    above = u_df[u_df["drift_score"] > THRESHOLD]
    if len(above) >= 3:
        onsets.append(above.iloc[0]["day"])

if onsets:
    onset_series = pd.Series(onsets)
    print(onset_series.describe())
else:
    print("No strong drift onsets detected")

print("Expected: onsets not clustered at very early days")

# =====================================================
# 4. EXPLANATION SANITY CHECK
# =====================================================
print("\n--- Explanation Sanity Check ---")

top_explanations = (
    explain_df
    .reindex(explain_df["contribution"].abs().sort_values(ascending=False).index)
    .head(10)
)

print(top_explanations)

print("\nExpected:")
print("- Large positive/negative contributions")
print("- Clear direction (increase/decrease)")
print("- Interpretable features")

print("\n✔ Phase 7 evaluation completed successfully")
