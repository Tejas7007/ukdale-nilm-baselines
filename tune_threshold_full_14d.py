import json, pandas as pd, numpy as np, joblib
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV = "b1_kettle_14d_1min.csv"
MODEL = "mlp_kettle_14d.joblib"
THR_JSON = "mlp_kettle_14d_threshold.json"
ON_W = 10.0
RECALL_MIN = 0.85

# load data + model
df = pd.read_csv(CSV, parse_dates=[0], index_col=0)
pipe = joblib.load(MODEL)

# features (must match training)
def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["mains_t"] = df["mains_w"]
    for k in range(1,6):
        X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
    for w in [3,5]:
        X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
        X[f"mains_rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)
    return X

y = (df["kettle_w"] >= ON_W).astype(int)
X = make_features(df).dropna()
y = y.loc[X.index].values
proba = pipe.predict_proba(X.values)[:,1]

# sweep thresholds
cands = []
for thr in np.linspace(0.01, 0.50, 100):
    pred = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    cands.append((thr, p, r, f1, tn, fp, fn, tp))

feasible = [c for c in cands if c[2] >= RECALL_MIN]
best = max(feasible, key=lambda x: x[1]) if feasible else max(cands, key=lambda x: x[3])

thr,p,r,f1,tn,fp,fn,tp = best
auc = roc_auc_score(y, proba) if len(np.unique(y))==2 else float("nan")
print(f"AUC={auc:.4f}")
print(f"Best precision with recall≥{RECALL_MIN}: thr={thr:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

# save new threshold
with open(THR_JSON, "w") as f:
    json.dump({"threshold": float(thr), "csv": CSV}, f)
print(f"✅ Updated {THR_JSON} with threshold {thr:.2f}")
