# save_mlp_kettle_14d.py
import json, joblib, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

CSV_PATH = "b1_kettle_14d_1min.csv"
THRESHOLD = 0.06   # from your tuning
ON_THRESHOLD_W = 10.0
RANDOM_STATE = 42

# --- features function (same as training) ---
def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["mains_t"] = df["mains_w"]
    for k in range(1, 6):
        X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
    for w in [3, 5]:
        X[f"mains_rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
        X[f"mains_rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)
    return X

# --- load data & split exactly as before ---
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)
X = make_features(df)
xy = pd.concat([X, y.rename("y")], axis=1).dropna()
X, y = xy.drop(columns=["y"]).values, xy["y"].values

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

# --- fit final pipeline on *all* 14d data to deploy (optional but typical) ---
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
        random_state=RANDOM_STATE
    ))
])
pipe.fit(X, y)  # train on full 14d window

# --- save model and threshold ---
joblib.dump(pipe, "mlp_kettle_14d.joblib")
with open("mlp_kettle_14d_threshold.json", "w") as f:
    json.dump({"threshold": THRESHOLD, "csv": CSV_PATH}, f)

print("✅ Saved: mlp_kettle_14d.joblib and mlp_kettle_14d_threshold.json")
