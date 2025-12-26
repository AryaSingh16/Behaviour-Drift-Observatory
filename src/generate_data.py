import numpy as np
import pandas as pd

# -----------------------------
# Global configuration
# -----------------------------
SEED = 42
NUM_USERS = 500
NUM_DAYS = 90
DRIFT_RATIO = 0.3  # 30% of users drift

np.random.seed(SEED)

# -----------------------------
# Helper functions
# -----------------------------
def entropy_like(value, noise=0.05):
    """Clamp entropy-like values between 0 and 1"""
    return np.clip(value + np.random.normal(0, noise), 0.01, 0.99)

def positive(value, noise=0.1):
    """Ensure positive continuous values"""
    return max(0.01, value + np.random.normal(0, noise))

# -----------------------------
# User baseline generator
# -----------------------------
def generate_user_baseline():
    return {
        "session_count": np.random.uniform(2, 6),
        "avg_session_duration": np.random.uniform(8, 18),
        "active_hours_entropy": np.random.uniform(0.3, 0.6),
        "action_type_entropy": np.random.uniform(0.4, 0.7),
        "inter_day_variability": np.random.uniform(0.05, 0.15),
    }

# -----------------------------
# Drift configuration per user
# -----------------------------
def assign_drift():
    if np.random.rand() > DRIFT_RATIO:
        return None

    drift_type = np.random.choice(["sudden", "gradual"])
    drift_start = np.random.randint(25, 60)
    affected = np.random.choice(
        ["session_count", "avg_session_duration",
         "active_hours_entropy", "action_type_entropy"],
        size=np.random.randint(1, 3),
        replace=False
    )
    strength = np.random.uniform(0.2, 0.6)

    return {
        "type": drift_type,
        "start_day": drift_start,
        "features": affected,
        "strength": strength
    }

# -----------------------------
# Main data generation
# -----------------------------
rows = []

for user_id in range(NUM_USERS):
    baseline = generate_user_baseline()
    drift = assign_drift()

    prev_day_values = baseline.copy()

    for day in range(NUM_DAYS):
        values = {}

        for feature, base_value in baseline.items():
            current_value = base_value

            # Apply drift if applicable
            if drift and day >= drift["start_day"] and feature in drift["features"]:
                if drift["type"] == "sudden":
                    current_value *= (1 + drift["strength"])
                elif drift["type"] == "gradual":
                    progress = (day - drift["start_day"]) / (NUM_DAYS - drift["start_day"])
                    current_value *= (1 + drift["strength"] * progress)

            # Noise + constraints
            if "entropy" in feature:
                values[feature] = entropy_like(current_value)
            else:
                values[feature] = positive(current_value)

        # Inter-day variability depends on yesterday
        variability = np.mean([
            abs(values[f] - prev_day_values[f]) / (prev_day_values[f] + 1e-6)
            for f in values
        ])
        values["inter_day_variability"] = np.clip(variability, 0.01, 1.0)

        prev_day_values = values.copy()

        rows.append({
            "user_id": f"user_{user_id}",
            "day": day,
            **values
        })

# -----------------------------
# Save dataset
# -----------------------------
df = pd.DataFrame(rows)
df.to_csv("data/synthetic_behavior.csv", index=False)

print("Synthetic dataset generated:", df.shape)
