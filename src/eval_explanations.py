import pandas as pd

EXPLAIN_PATH = "data/drift_explanations.csv"

df = pd.read_csv(EXPLAIN_PATH)

# Look at strongest contributions
top = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)

print("Top explanation samples:")
print(top.head(10))
