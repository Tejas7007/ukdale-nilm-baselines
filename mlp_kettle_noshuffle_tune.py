# mlp_kettle_noshuffle_tune.py
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0
RECALL_MIN = 0.80
RANDOM_STATE = 42

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# --- features: mains + lags(1..5) + rolling(3,5)
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
for w in [3, 5]:
    X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
    X[f"mains_rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)

xy = pd.concat([X, y.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values

# --- time-ordered split
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=64,
        max_iter=500,
        early_stopping=False,
        shuffle=False,
        random_state=RANDOM_STATE,
        verbose=False
    ))
])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"AUC={auc:.4f}")

# --- dense threshold sweep
cands = []
for thr in np.linspace(0.01, 0.9, 90):
    pred = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cands.append((thr, p, r, f1, tn, fp, fn, tp))

# 1) best precision subject to recall >= RECALL_MIN
feasible = [c for c in cands if c[2] >= RECALL_MIN]
best_prec = max(feasible, key=lambda x: x[1]) if feasible else None

# 2) best F1 (for reference)
best_f1 = max(cands, key=lambda x: x[3])

def pretty(tag, row):
    thr,p,r,f1,tn,fp,fn,tp = row
    print(f"{tag}: thr={thr:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}")

if best_prec:
    pretty(f"Best precision (recall≥{RECALL_MIN})", best_prec)
else:
    print(f"No threshold reached recall ≥ {RECALL_MIN}; showing best F1 instead.")
pretty("Best F1 (unconstrained)", best_f1)
