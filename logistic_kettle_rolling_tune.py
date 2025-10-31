# logistic_kettle_rolling_tune.py
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0
ROLLS = [3, 5]
RECALL_MIN = 0.80  # keep at least 0.80 recall

# load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# features: mains with lags + rolling stats
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
for w in ROLLS:
    X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
    X[f"mains_rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)

xy = pd.concat([X, y.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values

# time-ordered split
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# fit
clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

# sweep thresholds; choose highest precision s.t. recall >= RECALL_MIN
cands = []
for thr in np.linspace(0.1, 0.9, 33):
    pred = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cands.append((thr, p, r, f1, tn, fp, fn, tp))

feasible = [c for c in cands if c[2] >= RECALL_MIN]
best = max(feasible, key=lambda x: x[1]) if feasible else max(cands, key=lambda x: x[3])

thr, p, r, f1, tn, fp, fn, tp = best
print(f"AUC: {auc:.4f}")
print(f"Best by precision with recall ≥ {RECALL_MIN}:")
print(f"  thr={thr:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# (optional) save curve
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thrs, ps, rs, fs = zip(*[(t, P, R, F) for t,P,R,F,_,_,_,_ in cands])
    plt.figure(figsize=(10,4))
    plt.plot(thrs, ps, label="Precision")
    plt.plot(thrs, rs, label="Recall")
    plt.plot(thrs, fs, label="F1")
    plt.axhline(RECALL_MIN, ls="--", label=f"Recall min {RECALL_MIN}")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.legend()
    plt.title("Rolling features — threshold tuning")
    plt.tight_layout(); plt.savefig("logreg_kettle_rolling_tuning.png", dpi=150)
    print("✅ Saved: logreg_kettle_rolling_tuning.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
