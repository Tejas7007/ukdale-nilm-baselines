# logistic_kettle_rolling.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0
ROLLS = [3, 5]  # 3-min and 5-min windows

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)

# target: ON/OFF
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# base features: mains with lags 1..5
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)

# rolling stats on mains
for w in ROLLS:
    X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
    X[f"mains_rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)

# align & drop NaNs from shifts
xy = pd.concat([X, y.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values

# time-ordered split
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# model
clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
clf.fit(X_train, y_train)

# evaluate at threshold tuned earlier (0.61). You can sweep again if you like.
proba = clf.predict_proba(X_test)[:, 1]
thr = 0.61
y_pred = (proba >= thr).astype(int)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
auc = roc_auc_score(y_test, proba)
cm = confusion_matrix(y_test, y_pred)

print("AUC:", round(auc, 4))
print(f"thr={thr}  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  F1={f1:.4f}")
print("Confusion matrix:\n", cm)

# quick viz (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(300, len(y_test))
    xs = np.arange(n)
    plt.figure(figsize=(12,4))
    plt.plot(xs, y_test[:n], label="Actual ON(1)/OFF(0)")
    plt.plot(xs, proba[:n], label="Pred prob (ON)")
    plt.legend(); plt.title("Logistic + Rolling Features — Kettle (test subset)")
    plt.tight_layout(); plt.savefig("logreg_kettle_rolling.png", dpi=150)
    print("✅ Saved: logreg_kettle_rolling.png")
except Exception as e:
    print(f"(Plot skipped: {e})")

