# mlp_kettle.py
#You’re teaching a neural network to predict whether the kettle is ON or OFF by looking at the electricity power readings from your house (the “mains”)
import pandas as pd, numpy as np
from sklearn.neural_network import MLPClassifier #MLPClassifier: your neural network model (Multi-Layer Perceptron).
from sklearn.preprocessing import StandardScaler #StandardScaler: scales all numbers to a similar range (so large numbers don’t dominate)
from sklearn.pipeline import Pipeline #Pipeline: lets you combine multiple steps (like scaling + training) into one clean process.
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix #confusion_matrix: counts how many times it was right or wrong for each case.
)

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0 #If kettle power ≥ 10 watts → we say the kettle is ON.
ROLLS = [3, 5] #You’ll calculate averages over 3-minute and 5-minute windows. These “rolling” stats help the model see trends (like the kettle slowly heating).
RECALL_MIN = 0.80 #When tuning thresholds later, we’ll make sure recall ≥ 80% (so we don’t miss too many ON events).
RANDOM_STATE = 42 #This ensures the model behaves the same every time you run it; useful for reproducibility.

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0) #Converts the first column to datetime (your timestamps).

# target: ON/OFF
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

# features: mains + lags + rolling stats
X = pd.DataFrame(index=df.index) #Start an empty table (same time index) to hold your input features
X["mains_t"] = df["mains_w"] #This is your current mains power reading at time t.
for k in range(1, 6): #Adds the power readings from 1 to 5 minutes earlier; helps the model remember what was happening before
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
for w in ROLLS:
    X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean() #For every window (3 and 5 minutes): mean: average power (shows trend).
std (standard deviation): measures how “spiky” the power was.
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
pipe = Pipeline([ #Pipeline: combines multiple steps into one process.
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16), #hidden_layer_sizes=(32,16) → two hidden layers: one with 32 neurons, then one with 16.
        activation="relu", #activation="relu" → introduces nonlinearity, helping the network learn complex shapes.
        solver="adam", #solver="adam" → optimizer that adjusts weights using gradient descent.
        learning_rate_init=1e-3, #learning_rate_init=1e-3 → small step size for stable training.
        alpha=1e-4, #alpha=1e-4 → adds tiny regularization to prevent overfitting.
        batch_size=64, #processes 64 samples per training step.
        max_iter=200, #at most 200 passes through the data.
        early_stopping=True, #early_stopping=True → stops early if performance stops improving.
        n_iter_no_change=10, #n_iter_no_change=10 → if no improvement for 10 rounds, stop.
        random_state=RANDOM_STATE,
        verbose=False
    ))
])

pipe.fit(X_train, y_train)

# probabilities
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba) #AUC tells you how well the model distinguishes ON vs OFF.

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
for thr in np.linspace(0.1, 0.9, 33): #Checks thresholds from 0.1 → 0.9 in small steps. Each time, compute precision, recall, and confusion matrix.
    pred = (proba >= thr).astype(int) #Keeps only thresholds with recall ≥ 0.8,
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0) #Picks the one with best precision among them (or best F1 if none meet the recall target).
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
