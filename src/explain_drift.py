import pandas as pd
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
REPRESENTATION_PATH = "data/behavior_representations.csv"
DRIFT_PATH = "data/drift_scores.csv"
OUTPUT_PATH = "data/drift_explanations.csv"

REFERENCE_WINDOW = 30
CURRENT_WINDOW = 14
EPSILON = 1e-8
TOP_K = 3

FEATURE_COLUMNS = [
    "session_count_mean_14d",
    "avg_session_duration_mean_14d",
    "active_hours_entropy_mean_14d",
    "action_type_entropy_mean_14d",
    "inter_day_variability_mean_14d"
]

# -----------------------------
# Load data
# -----------------------------
rep_df = pd.read_csv(REPRESENTATION_PATH)
drift_df = pd.read_csv(DRIFT_PATH)

rep_df = rep_df.sort_values(by=["user_id", "day"]).reset_index(drop=True)

# -----------------------------
# Build explanation
# -----------------------------
explanation_rows = []

for user_id, user_rep in rep_df.groupby("user_id"):
    user_rep = user_rep.reset_index(drop=True)

    user_drift = drift_df[drift_df["user_id"] == user_id]
    if user_drift.empty:
        continue

    for _, drift_row in user_drift.iterrows():
        day = int(drift_row["day"])
        idx = user_rep[user_rep["day"] == day].index

        if len(idx) == 0:
            continue

        idx = idx[0]

        ref_start = idx - CURRENT_WINDOW - REFERENCE_WINDOW + 1
        ref_end = idx - CURRENT_WINDOW + 1
        cur_start = idx - CURRENT_WINDOW + 1
        cur_end = idx + 1

        if ref_start < 0:
            continue

        ref_window = user_rep.iloc[ref_start:ref_end]
        cur_window = user_rep.iloc[cur_start:cur_end]

        mu_ref = ref_window[FEATURE_COLUMNS].mean()
        mu_cur = cur_window[FEATURE_COLUMNS].mean()

        contributions = (mu_cur - mu_ref) / (mu_ref.abs() + EPSILON)

        top_features = contributions.abs().sort_values(ascending=False).head(TOP_K)

        for feature in top_features.index:
            explanation_rows.append({
                "user_id": user_id,
                "day": day,
                "feature": feature.replace("_mean_14d", ""),
                "contribution": contributions[feature],
                "direction": "increase" if contributions[feature] > 0 else "decrease"
            })

# -----------------------------
# Save explanations
# -----------------------------
explain_df = pd.DataFrame(explanation_rows)
explain_df.to_csv(OUTPUT_PATH, index=False)

print(
    f"Drift explanations generated: {explain_df.shape}\n"
    f"Saved to: {OUTPUT_PATH}"
)
