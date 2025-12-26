import pandas as pd
import matplotlib.pyplot as plt

DRIFT_PATH = "data/drift_scores.csv"

df = pd.read_csv(DRIFT_PATH)

# Find users with strongest drift
stats = df.groupby("user_id")["drift_score"].max().reset_index()
drifting_users = stats.sort_values("drift_score", ascending=False).head(5)["user_id"]

plt.figure()
for user in drifting_users:
    user_df = df[df["user_id"] == user]
    plt.plot(user_df["day"], user_df["drift_score"], label=user)

plt.xlabel("Day")
plt.ylabel("Drift Score")
plt.title("Sensitivity Test: High-Drift Users")
plt.legend()
plt.show()
