# logistic_kettle_tune.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0

df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
y_on = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)

xy = pd.concat([X, y_on.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values

split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

best = None
for thr in np.linspace(0.1, 0.9, 17):
    pred = (proba >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    score = f1  # optimize F1 (change to your preference)
    row = (thr, prec, rec, f1, tn, fp, fn, tp)
    if (best is None) or (score > best[2]):
        best = row

print(f"AUC: {auc:.4f}")
print("Best threshold tuning (by F1):")
print(f"  thr={best[0]:.2f}  precision={best[1]:.3f}  recall={best[2]:.3f}  F1={best[3]:.3f}")
print(f"  TN={best[4]}  FP={best[5]}  FN={best[6]}  TP={best[7]}")

# Optional: save a small threshold–metrics table
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thrs, ps, rs, fs = [], [], [], []
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (proba >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        thrs.append(thr); ps.append(p); rs.append(r); fs.append(f)

    plt.figure(figsize=(10,4))
    plt.plot(thrs, ps, label="Precision")
    plt.plot(thrs, rs, label="Recall")
    plt.plot(thrs, fs, label="F1")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title("Threshold Tuning — Logistic (Kettle)")
    plt.legend(); plt.tight_layout()
    plt.savefig("logreg_kettle_threshold_tuning.png", dpi=150)
    print("✅ Saved: logreg_kettle_threshold_tuning.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
