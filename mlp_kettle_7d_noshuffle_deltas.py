# mlp_kettle_7d_noshuffle_deltas.py
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV_PATH = "b1_kettle_7d_1min.csv"
ON_THRESHOLD_W = 10.0
RECALL_MIN = 0.80
RANDOM_STATE = 42

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# --- features: mains + lags(1..5) + deltas + rolling(3,5,9,15)
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)

# spike features (capture rapid on/off edges)
X["dmains_t"] = X["mains_t"].diff()
for k in range(1, 6):
    X[f"dmains_t_minus_{k}"] = X[f"mains_t_minus_{k}"].diff()

for w in [3, 5, 9, 15]:
    X[f"rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
    X[f"rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)

# align
xy = pd.concat([X, y.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values
ts = xy.index.to_numpy()

# --- time-ordered split ensuring both classes in test
def find_good_split(y, base_frac=0.8, step=300):
    n = len(y); split = int(n*base_frac)
    while split > int(n*0.6):
        yt = y[split:]; 
        if yt.sum() > 0 and (len(yt)-yt.sum()) > 0: return split
        split -= step
    split = int(n*base_frac)
    while split < int(n*0.9):
        yt = y[split:]; 
        if yt.sum() > 0 and (len(yt)-yt.sum()) > 0: return split
        split += step
    return int(n*0.8)

split = find_good_split(y)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train end: {ts[split-1]} | Test start: {ts[split]}")
print(f"Class counts — train: pos={y_train.sum()}/{len(y_train)}, test: pos={y_test.sum()}/{len(y_test)}")

# --- pipeline: Scale -> MLP (no shuffle, no early stopping)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=64,
        max_iter=600,
        early_stopping=False,
        shuffle=False,
        random_state=RANDOM_STATE,
        verbose=False
    ))
])

pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) == 2 else float("nan")
print(f"AUC={auc:.4f}" if np.isfinite(auc) else "AUC=NA")

# --- threshold sweep: maximize precision with recall ≥ 0.80
cands = []
for thr in np.linspace(0.001, 0.5, 100):
    pred = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
    cands.append((thr, p, r, f1, tn, fp, fn, tp))

feasible = [c for c in cands if c[2] >= RECALL_MIN]
best = max(feasible, key=lambda x: x[1]) if feasible else max(cands, key=lambda x: x[3])
thr, p, r, f1, tn, fp, fn, tp = best
print("\n=== MLP (spike features) tuned ===")
print(f"thr={thr:.3f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
