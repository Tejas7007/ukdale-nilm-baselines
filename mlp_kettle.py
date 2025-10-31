# mlp_kettle.py
import pandas as pd, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0
ROLLS = [3, 5]
RECALL_MIN = 0.80
RANDOM_STATE = 42

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)

# target: ON/OFF
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# features: mains + lags + rolling stats
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
for w in ROLLS:
    X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
    X[f"mains_rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)

# align
xy = pd.concat([X, y.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values

# time-ordered split (80/20)
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# pipeline: Standardize -> MLP
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=64,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False
    ))
])

pipe.fit(X_train, y_train)

# probabilities
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

# evaluate at default 0.5 threshold
y_pred_05 = (proba >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_05)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_05, average="binary", zero_division=0)
cm = confusion_matrix(y_test, y_pred_05)
print("=== MLP @ thr=0.50 ===")
print(f"AUC={auc:.4f}  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  F1={f1:.4f}")
print("Confusion matrix:\n", cm)

# simple threshold sweep: best precision subject to recall >= RECALL_MIN
cands = []
for thr in np.linspace(0.1, 0.9, 33):
    pred = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cands.append((thr, p, r, f1, tn, fp, fn, tp))

feasible = [c for c in cands if c[2] >= RECALL_MIN]
best = max(feasible, key=lambda x: x[1]) if feasible else max(cands, key=lambda x: x[3])

thr, p, r, f1, tn, fp, fn, tp = best
print("\n=== MLP tuned threshold ===")
print(f"Best by precision with recall ≥ {RECALL_MIN}:")
print(f"thr={thr:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  AUC={auc:.4f}")
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# quick viz of probabilities
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(300, len(y_test))
    xs = np.arange(n)
    plt.figure(figsize=(12,4))
    plt.plot(xs, y_test[:n], label="Actual ON(1)/OFF(0)")
    plt.plot(xs, proba[:n], label="Pred prob (ON)")
    plt.legend(); plt.title("MLP — Kettle ON/OFF (test subset)")
    plt.tight_layout(); plt.savefig("mlp_kettle_probs.png", dpi=150)
    print("✅ Saved: mlp_kettle_probs.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
