import pandas as pd
import matplotlib.pyplot as plt

DRIFT_PATH = "data/drift_scores.csv"

df = pd.read_csv(DRIFT_PATH)

# Compute per-user drift stats
stats = df.groupby("user_id")["drift_score"].agg(["mean", "std"]).reset_index()

# Select most stable users
stable_users = stats.sort_values("mean").head(5)["user_id"]

# Plot drift timelines
plt.figure()
for user in stable_users:
    user_df = df[df["user_id"] == user]
    plt.plot(user_df["day"], user_df["drift_score"], label=user)

plt.xlabel("Day")
plt.ylabel("Drift Score")
plt.title("Stability Test: Low-Drift Users")
plt.legend()
plt.show()
