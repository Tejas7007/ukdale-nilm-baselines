# predict_multilabel.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path

CFG = "multilabel_thresholds.json"       # produced by the tuner
OUT = "multilabel_predictions.csv"       # output (probs + ON/OFF for each label)

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["mains_t"] = df["mains_w"]
    for k in range(1,6):
        X[f"lag{k}"] = df["mains_w"].shift(k)
    for w in [3,5,9,15]:
        X[f"mean{w}"] = df["mains_w"].rolling(w, min_periods=1).mean()
        X[f"std{w}"]  = df["mains_w"].rolling(w, min_periods=1).std().fillna(0.0)
    return X

with open(CFG) as f:
    cfg = json.load(f)
thr = cfg["thresholds"]
models = cfg["models"]
csv = cfg["csv"]

df = pd.read_csv(csv, parse_dates=[0], index_col=0)
X = make_features(df).dropna()
ts = X.index

out = pd.DataFrame(index=ts)

for a, model_path in models.items():
    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(X.values)[:,1]
    pred  = (proba >= thr[a]).astype(int)
    key = a.replace(" ", "_")
    out[f"{key}_proba"] = proba
    out[f"{key}_on"]    = pred

# keep mains for context if present at aligned timestamps
if "mains_w" in df.columns:
    out = out.join(df[["mains_w"]], how="left")

out.to_csv(OUT)
print(f"✅ Wrote {OUT} with per-label probabilities and ON/OFF predictions.")
