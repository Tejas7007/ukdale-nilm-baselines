# logistic_kettle_threshold_rule.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0
RECALL_MIN = 0.85  # constraint

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
y_on = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# --- features: mains with lags 1..5
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)

xy = pd.concat([X, y_on.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values

# --- time-ordered split
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- train
clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

# --- threshold search: max precision with recall >= RECALL_MIN
candidates = []
for thr in np.linspace(0.05, 0.9, 36):
    pred = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    candidates.append((thr, p, r, f1, tn, fp, fn, tp))

# filter by recall constraint, then choose highest precision
feasible = [row for row in candidates if row[2] >= RECALL_MIN]
best = max(feasible, key=lambda x: x[1]) if feasible else max(candidates, key=lambda x: x[3])  # fallback to best F1

thr,p,r,f1,tn,fp,fn,tp = best
print(f"AUC: {auc:.4f}")
print(f"Best by precision with recall ≥ {RECALL_MIN}:")
print(f"  thr={thr:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# optional: save a curve for precision/recall vs threshold
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thrs, ps, rs = zip(*[(t, pr, rc) for t,pr,rc,_,_,_,_,_ in candidates])
    plt.figure(figsize=(10,4))
    plt.plot(thrs, ps, label="Precision")
    plt.plot(thrs, rs, label="Recall")
    plt.axhline(RECALL_MIN, linestyle="--", label=f"Recall min {RECALL_MIN}")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title("Threshold selection — maximize precision with recall constraint")
    plt.legend(); plt.tight_layout()
    plt.savefig("logreg_kettle_threshold_rule.png", dpi=150)
    print("✅ Saved: logreg_kettle_threshold_rule.png")
except Exception as e:
    print(f"(Plot skipped: {e})")

