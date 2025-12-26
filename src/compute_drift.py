import pandas as pd
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
REPRESENTATION_PATH = "data/behavior_representations.csv"
OUTPUT_PATH = "data/drift_scores.csv"

REFERENCE_WINDOW = 30
CURRENT_WINDOW = 14
EPSILON = 1e-8

FEATURE_COLUMNS = [
    "session_count_mean_14d",
    "avg_session_duration_mean_14d",
    "active_hours_entropy_mean_14d",
    "action_type_entropy_mean_14d",
    "inter_day_variability_mean_14d"
]

# -----------------------------
# Load representations
# -----------------------------
df = pd.read_csv(REPRESENTATION_PATH)
df = df.sort_values(by=["user_id", "day"]).reset_index(drop=True)

# -----------------------------
# Drift computation
# -----------------------------
drift_rows = []

for user_id, user_df in df.groupby("user_id"):
    user_df = user_df.reset_index(drop=True)

    min_required = REFERENCE_WINDOW + CURRENT_WINDOW
    if len(user_df) < min_required:
        continue

    for idx in range(min_required - 1, len(user_df)):
        ref_start = idx - CURRENT_WINDOW - REFERENCE_WINDOW + 1
        ref_end = idx - CURRENT_WINDOW + 1
        cur_start = idx - CURRENT_WINDOW + 1
        cur_end = idx + 1

        ref_window = user_df.iloc[ref_start:ref_end]
        cur_window = user_df.iloc[cur_start:cur_end]

        mu_ref = ref_window[FEATURE_COLUMNS].mean().values
        mu_cur = cur_window[FEATURE_COLUMNS].mean().values

        # Normalized L2 drift score
        drift_score = np.linalg.norm(mu_cur - mu_ref) / (
            np.linalg.norm(mu_ref) + EPSILON
        )

        drift_rows.append({
            "user_id": user_id,
            "day": int(user_df.loc[idx, "day"]),
            "drift_score": drift_score
        })

# -----------------------------
# Save drift scores
# -----------------------------
drift_df = pd.DataFrame(drift_rows)
drift_df.to_csv(OUTPUT_PATH, index=False)

print(
    f"Drift scores computed: {drift_df.shape}\n"
    f"Saved to: {OUTPUT_PATH}"
)
