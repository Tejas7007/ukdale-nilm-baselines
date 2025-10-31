# mlp_kettle_7d_noshuffle_diag.py
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV_PATH = "b1_kettle_7d_1min.csv"
ON_THRESHOLD_W = 10.0
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
ts = xy.index.to_numpy()

# --- ensure both classes in test
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
ts_train_end, ts_test_start = ts[split-1], ts[split]
print(f"Train end: {ts_train_end} | Test start: {ts_test_start}")
print(f"Class counts — train: pos={y_train.sum()}/{len(y_train)}, test: pos={y_test.sum()}/{len(y_test)}")

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

# --- show probabilities for the POSITIVE timestamps (to diagnose)
pos_idx = np.where(y_test == 1)[0]
pos_rows = [(ts[split + i], float(proba[i])) for i in pos_idx]
pos_rows_sorted = sorted(pos_rows, key=lambda x: x[1], reverse=True)
print("\nTop positives by predicted probability:")
for t, p in pos_rows_sorted:
    print(f"  {t}  ->  {p:.4f}")

# --- ultra-low threshold sweep (down to 1e-4)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
best_f1 = None
best_recall = None
for thr in np.concatenate([np.linspace(1e-4, 0.05, 30), np.linspace(0.05, 0.5, 46)]):
    pred = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
    if best_f1 is None or f1 > best_f1[2]:
        best_f1 = (thr, p, r, f1, tn, fp, fn, tp)
    if (best_recall is None or (r > best_recall[2])) and r >= 0.80:
        # keep the highest precision among recall>=0.80
        if best_recall is None or p > best_recall[1]:
            best_recall = (thr, p, r, f1, tn, fp, fn, tp)

print("\nBest F1 (any threshold):")
thr,p,r,f1,tn,fp,fn,tp = best_f1
print(f"  thr={thr:.4f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}")

if best_recall:
    thr,p,r,f1,tn,fp,fn,tp = best_recall
    print("\nBest precision with recall≥0.80:")
    print(f"  thr={thr:.4f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}")
else:
    print("\nNo threshold achieved recall≥0.80 even at ultra-low thresholds.")
