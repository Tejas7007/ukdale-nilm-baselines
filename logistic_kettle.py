# logistic_kettle.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0  # ON if kettle_w >= 10 W

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)

# --- target: ON/OFF
y_on = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# --- features: mains with lags (t, t-1..t-5)
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)

# align + drop NaNs from shifting
xy = pd.concat([X, y_on.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values

# --- time-ordered split: 80/20
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- model
clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
clf.fit(X_train, y_train)

# --- eval
proba = clf.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
try:
    auc = roc_auc_score(y_test, proba)
except Exception:
    auc = float("nan")
cm = confusion_matrix(y_test, y_pred)

print("Accuracy :", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("F1       :", round(f1, 4))
print("ROC AUC  :", round(auc, 4))
print("Confusion matrix:\n", cm)

# optional: save a quick overlay plot of probabilities vs label
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(300, len(y_test))
    xs = np.arange(n)
    plt.figure(figsize=(12,4))
    plt.plot(xs, y_test[:n], label="Actual ON(1)/OFF(0)")
    plt.plot(xs, proba[:n], label="Pred prob(ON)")
    plt.legend()
    plt.title("Logistic Regression — Kettle ON/OFF (test subset)")
    plt.tight_layout()
    plt.savefig("logreg_kettle_probs.png", dpi=150)
    print("✅ Saved: logreg_kettle_probs.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
