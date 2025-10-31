# predict_kettle.py
import json, joblib, pandas as pd, numpy as np

MODEL_PATH = "mlp_kettle_14d.joblib"
THRESHOLD_PATH = "mlp_kettle_14d_threshold.json"
ON_THRESHOLD_W = 10.0

def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["mains_t"] = df["mains_w"]
    for k in range(1, 6):
        X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
    for w in [3, 5]:
        X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
        X[f"mains_rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)
    return X

# load model + thr
pipe = joblib.load(MODEL_PATH)
thr = json.load(open(THRESHOLD_PATH))["threshold"]

# example: predict on the same CSV (or any CSV with mains_w)
CSV = "b1_kettle_14d_1min.csv"
df = pd.read_csv(CSV, parse_dates=[0], index_col=0)
X = make_features(df).dropna()
proba = pipe.predict_proba(X.values)[:, 1]
pred_on = (proba >= thr).astype(int)

# if ground-truth exists, report quick metrics
if "kettle_w" in df.columns:
    y = (df.loc[X.index, "kettle_w"] >= ON_THRESHOLD_W).astype(int).values
    from sklearn.metrics import classification_report
    print(classification_report(y, pred_on, digits=3))

# save predictions with timestamps
out = pd.DataFrame({"proba": proba, "pred_on": pred_on}, index=X.index)
out.to_csv("kettle_predictions.csv")
print("✅ Wrote kettle_predictions.csv")
