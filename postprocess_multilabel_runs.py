# postprocess_multilabel_runs.py
import pandas as pd
import numpy as np

# Require N consecutive ON minutes to confirm "true ON"
N = 2
INPUT = "multilabel_predictions.csv"
OUTPUT = "multilabel_predictions_pp.csv"

df = pd.read_csv(INPUT, parse_dates=[0], index_col=0)

# Columns that end in "_on" (per appliance)
on_cols = [c for c in df.columns if c.endswith("_on")]

def clean_runs(series, n=N):
    s = series.astype(int).copy()
    runs = (s != s.shift()).cumsum()
    sizes = runs.map(runs.value_counts())
    return s.where(~((s == 1) & (sizes < n)), 0)

for c in on_cols:
    df[c] = clean_runs(df[c])

df.to_csv(OUTPUT)
print(f"✅ Wrote {OUTPUT}")
