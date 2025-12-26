import pandas as pd
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/synthetic_behavior.csv"
OUTPUT_PATH = "data/behavior_representations.csv"

WINDOW_SIZE = 14

FEATURE_COLUMNS = [
    "session_count",
    "avg_session_duration",
    "active_hours_entropy",
    "action_type_entropy",
    "inter_day_variability"
]

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Ensure correct ordering
df = df.sort_values(by=["user_id", "day"]).reset_index(drop=True)

# -----------------------------
# Build rolling representations
# -----------------------------
representation_rows = []

for user_id, user_df in df.groupby("user_id"):
    user_df = user_df.reset_index(drop=True)

    for current_day in range(WINDOW_SIZE - 1, len(user_df)):
        window_df = user_df.iloc[current_day - WINDOW_SIZE + 1 : current_day + 1]

        rep = {
            "user_id": user_id,
            "day": int(user_df.loc[current_day, "day"])
        }

        # Compute window means
        for feature in FEATURE_COLUMNS:
            rep[f"{feature}_mean_{WINDOW_SIZE}d"] = window_df[feature].mean()

        representation_rows.append(rep)

# -----------------------------
# Save representations
# -----------------------------
rep_df = pd.DataFrame(representation_rows)
rep_df.to_csv(OUTPUT_PATH, index=False)

print(
    f"Behavior representations generated: {rep_df.shape}\n"
    f"Saved to: {OUTPUT_PATH}"
)
