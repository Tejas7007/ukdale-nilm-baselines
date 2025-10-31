# multilabel_mlp_tune.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.base import clone

CSV = "b1_MULTI_14d_1min.csv"
APPS = ["kettle", "microwave", "fridge freezer", "toaster"]

# Per-appliance ON thresholds (watts). Tweak if needed.
ON_W = {"default": 10.0, "kettle":10.0, "microwave":30.0, "fridge freezer":60.0, "toaster":25.0}

# Per-appliance recall targets for threshold tuning
RECALL_MIN = {"kettle":0.85, "microwave":0.70, "fridge freezer":0.75, "toaster":0.70}

# -------- utils --------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["mains_t"] = df["mains_w"]
    # lags
    for k in range(1,6):
        X[f"lag{k}"] = df["mains_w"].shift(k)
    # rolling stats
    for w in [3,5,9,15]:
        X[f"mean{w}"] = df["mains_w"].rolling(w, min_periods=1).mean()
        X[f"std{w}"]  = df["mains_w"].rolling(w, min_periods=1).std().fillna(0.0)
    return X

# -------- load data --------
df = pd.read_csv(CSV, parse_dates=[0], index_col=0)
X = make_features(df)
Y = pd.DataFrame(index=df.index)
for a in APPS:
    col = f"{a}_w"
    th = ON_W.get(a, ON_W["default"])
    Y[a] = ((df[col] if col in df.columns else 0.0) >= th).astype(int)

xy = pd.concat([X, Y], axis=1).dropna()
X = xy[X.columns].values
Y = xy[APPS].values
TS = xy.index

# time-ordered split 80/20
split = int(len(xy)*0.8)
X_tr, X_te = X[:split], X[split:]
Y_tr, Y_te = Y[:split], Y[split:]
TS_te = TS[split:]

# base pipeline
base = Pipeline([
    ("sc", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64,32),
        activation="relu",
        solver="adam",
        max_iter=500,
        shuffle=False,
        random_state=42
    ))
])

models = {}
thresholds = {}
rows = []

Path("models_multilabel").mkdir(exist_ok=True)

print("\nPer-label MLP results with tuned thresholds (maximize precision s.t. recall ≥ target):")
for i, a in enumerate(APPS):
    ytr = Y_tr[:, i]; yte = Y_te[:, i]
    # need both classes in train & test
    if ytr.sum()==0 or ytr.sum()==len(ytr) or yte.sum()==0 or yte.sum()==len(yte):
        print(f"- {a}: skipped (insufficient class variety in train/test)")
        continue

    pipe = clone(base)
    pipe.fit(X_tr, ytr)
    proba = pipe.predict_proba(X_te)[:,1]
    auc = roc_auc_score(yte, proba)

    # sweep thresholds
    best = None
    tgt_recall = RECALL_MIN.get(a, 0.7)
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (proba >= thr).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        tn,fp,fn,tp = confusion_matrix(yte, pred, labels=[0,1]).ravel()
        if r >= tgt_recall:
            if best is None or p > best[1]:
                best = (thr, p, r, f1, tn, fp, fn, tp)

    # fallback to best F1 if recall target not achievable
    if best is None:
        best = (0.5, 0, 0, 0, 0, 0, 0, 0)
        for thr in np.linspace(0.01, 0.99, 99):
            pred = (proba >= thr).astype(int)
            p,r,f1,_ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
            tn,fp,fn,tp = confusion_matrix(yte, pred, labels=[0,1]).ravel()
            if f1 > best[3]:
                best = (thr, p, r, f1, tn, fp, fn, tp)

    thr,p,r,f1,tn,fp,fn,tp = best
    print(f"- {a}: AUC={auc:.3f}  thr={thr:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}")

    # save model & threshold
    out_model = Path("models_multilabel") / f"mlp_{a.replace(' ','_')}.joblib"
    joblib.dump(pipe, out_model)
    models[a] = str(out_model)
    thresholds[a] = float(thr)

    rows.append({
        "appliance": a, "AUC": auc, "thr": thr,
        "precision": p, "recall": r, "F1": f1,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    })

# save summary + thresholds
summary = pd.DataFrame(rows)
summary.to_csv("multilabel_mlp_summary.csv", index=False)
with open("multilabel_thresholds.json", "w") as f:
    json.dump({"thresholds": thresholds, "models": models, "csv": CSV}, f, indent=2)

print("\n✅ Saved:")
print("- models in models_multilabel/")
print("- per-label thresholds in multilabel_thresholds.json")
print("- summary table in multilabel_mlp_summary.csv")
