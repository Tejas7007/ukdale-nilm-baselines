# logistic_kettle_rich_scaled.py  (robust split)
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

CSV_PATH =  "b1_kettle_14d_1min.csv"   # <-- 7-day file
ON_THRESHOLD_W = 10.0
LAGS = 15
ROLLS = [3, 5, 9, 15]
RECALL_MIN = 0.80
RNG = np.random.RandomState(42)

# ---------- load + features ----------
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
y = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)

X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, LAGS + 1):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)

X["dmains_t"] = X["mains_t"].diff()
for k in range(1, LAGS + 1):
    X[f"dmains_t_minus_{k}"] = X[f"mains_t_minus_{k}"].diff()

for w in ROLLS:
    X[f"rollmean_{w}"] = df["mains_w"].rolling(window=w, min_periods=1).mean()
    X[f"rollstd_{w}"]  = df["mains_w"].rolling(window=w, min_periods=1).std().fillna(0)

xy = pd.concat([X, y.rename("y")], axis=1).dropna()
X = xy.drop(columns=["y"]).values
y = xy["y"].values
ts = xy.index.to_numpy()

# ---------- adaptive time-ordered split to ensure both classes in test ----------
def find_good_split(y, base_frac=0.8, step=300):
    n = len(y)
    split = int(n * base_frac)
    # move split earlier until test has both classes, but don’t go below 60%
    while split > int(n*0.6):
        y_test = y[split:]
        pos = y_test.sum()
        neg = len(y_test) - pos
        if pos > 0 and neg > 0:
            return split
        split -= step
    # fallback: try later splits forward
    split = int(n * base_frac)
    while split < int(n*0.9):
        y_test = y[split:]
        pos = y_test.sum()
        neg = len(y_test) - pos
        if pos > 0 and neg > 0:
            return split
        split += step
    return int(n*0.8)

split = find_good_split(y, base_frac=0.8, step=300)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
ts_train_end = ts[split-1]; ts_test_start = ts[split]
print(f"Train end: {ts_train_end}  |  Test start: {ts_test_start}")
print(f"Class counts — train: pos={y_train.sum()}/{len(y_train)}, test: pos={y_test.sum()}/{len(y_test)}")

# ---------- fit/eval helper ----------
def fit_eval(penalty: str, C: float):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            solver="saga", penalty=penalty, C=C,
            max_iter=5000, class_weight="balanced",
            n_jobs=-1, tol=1e-4, random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    # AUC guard: defined only if both classes present
    auc = np.nan
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, proba)

    # threshold sweep; best precision s.t. recall >= RECALL_MIN
    cands = []
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (proba >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        # force 2x2 matrix shape even if a class is missing
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
        cands.append((thr, p, r, f1, tn, fp, fn, tp))

    feas = [c for c in cands if c[2] >= RECALL_MIN]
    best = max(feas, key=lambda x: x[1]) if feas else max(cands, key=lambda x: x[3])
    return auc, best

results = []
for penalty in ["l1", "l2"]:
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        auc, (thr,p,r,f1,tn,fp,fn,tp) = fit_eval(penalty, C)
        results.append((penalty, C, auc, thr, p, r, f1, tn, fp, fn, tp))

# pick best: highest precision with recall≥RECALL_MIN; break ties by AUC then F1
feas = [row for row in results if row[6] >= RECALL_MIN]
if feas:
    feas.sort(key=lambda x: (x[4], np.nan_to_num(x[2], nan=-1.0), x[6], x[7]), reverse=True)
    best = feas[0]
else:
    results.sort(key=lambda x: (x[6], x[4], np.nan_to_num(x[2], nan=-1.0)), reverse=True)
    best = results[0]

penalty, C, auc, thr, p, r, f1, tn, fp, fn, tp = best
print("\nBest (by precision @ recall≥0.80 if feasible):")
print(f" penalty={penalty}  C={C}  AUC={auc if not np.isnan(auc) else 'NA':}")
print(f" thr={thr:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
print(f" TN={tn}  FP={fp}  FN={fn}  TP={tp}")
