# multilabel_logistic_tune.py
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

CSV = "b1_MULTI_14d_1min.csv"
APPS = ["kettle", "microwave", "fridge freezer", "toaster"]
ON_W = {"default": 10.0, "kettle":10.0, "microwave":20.0, "fridge freezer":50.0, "toaster":20.0}
RECALL_MIN = {"kettle":0.85, "microwave":0.70, "fridge freezer":0.70, "toaster":0.70}  # tweak if you like

# --- load
df = pd.read_csv(CSV, parse_dates=[0], index_col=0)

# --- features from mains
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1,6):
    X[f"lag{k}"] = df["mains_w"].shift(k)
for w in [3,5,9,15]:
    X[f"mean{w}"] = df["mains_w"].rolling(w, min_periods=1).mean()
    X[f"std{w}"]  = df["mains_w"].rolling(w, min_periods=1).std().fillna(0)

# --- labels matrix
Y = pd.DataFrame(index=df.index)
for a in APPS:
    th = ON_W.get(a, ON_W["default"])
    col = f"{a}_w"
    Y[a] = ((df[col] if col in df.columns else 0) >= th).astype(int)

xy = pd.concat([X, Y], axis=1).dropna()
X = xy[X.columns].values
Y = xy[APPS].values
split = int(len(X)*0.8)  # time-ordered split
X_tr, X_te = X[:split], X[split:]
Y_tr, Y_te = Y[:split], Y[split:]
Y_te_df = xy[APPS].iloc[split:]

# --- fit OvR logistic with scaling
pipe = Pipeline([
    ("sc", StandardScaler()),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced")))
])
pipe.fit(X_tr, Y_tr)
# probs per label
probs = pipe.predict_proba(X_te)  # shape [n, L]

print("\nPer-label AUC / tuned thresholds (maximize precision s.t. recall ≥ target):")
for i, a in enumerate(APPS):
    y = Y_te[:, i]
    pr = probs[:, i]
    if y.sum()==0 or (len(y)-y.sum())==0:
        print(f"- {a}: not enough class variety in test")
        continue

    auc = roc_auc_score(y, pr)

    # sweep thresholds
    best = None
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (pr >= thr).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        tn,fp,fn,tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        if r >= RECALL_MIN.get(a, 0.7):
            if best is None or p > best[1]:
                best = (thr, p, r, f1, tn, fp, fn, tp)

    # fallback to best F1 if recall target not achievable
    if best is None:
        for thr in np.linspace(0.01, 0.99, 99):
            pred = (pr >= thr).astype(int)
            p,r,f1,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            tn,fp,fn,tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            if best is None or f1 > best[2]:
                best_f1 = (thr, p, r, f1, tn, fp, fn, tp)
        best = best_f1

    thr,p,r,f1,tn,fp,fn,tp = best
    print(f"- {a}: AUC={auc:.3f}  thr={thr:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}")
