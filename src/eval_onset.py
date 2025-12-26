import pandas as pd

DRIFT_PATH = "data/drift_scores.csv"
THRESHOLD = 0.15

df = pd.read_csv(DRIFT_PATH)

onsets = []

for user_id, user_df in df.groupby("user_id"):
    above = user_df[user_df["drift_score"] > THRESHOLD]
    if len(above) >= 3:
        onset_day = above.iloc[0]["day"]
        onsets.append(onset_day)

print("Onset day statistics:")
print(pd.Series(onsets).describe())
