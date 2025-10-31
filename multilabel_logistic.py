# multilabel_logistic.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, roc_auc_score

CSV = "b1_MULTI_14d_1min.csv"
APPS = ["kettle", "microwave", "fridge freezer", "toaster"]
ON_W = {"default": 10.0, "fridge freezer": 30.0}  # tweak thresholds if you want

df = pd.read_csv(CSV, parse_dates=[0], index_col=0)

# features from mains
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1,6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)
for w in [3,5]:
    X[f"rollmean_{w}"] = df["mains_w"].rolling(w, min_periods=1).mean()
    X[f"rollstd_{w}"]  = df["mains_w"].rolling(w, min_periods=1).std().fillna(0)

# label matrix
Y = pd.DataFrame(index=df.index)
for a in APPS:
    th = ON_W.get(a, ON_W["default"])
    col = f"{a}_w"
    if col in df.columns:
        Y[a] = (df[col] >= th).astype(int)
    else:
        Y[a] = 0  # missing meter → all zeros

xy = pd.concat([X, Y], axis=1).dropna()
X = xy[X.columns].values
Y = xy[APPS].values
ts = xy.index

# time-ordered split (last 20% test)
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
apps_test = xy[APPS].iloc[split:]

pipe = Pipeline([
    ("sc", StandardScaler()),
    ("clf", OneVsRestClassifier(
        LogisticRegression(max_iter=500, class_weight="balanced", solver="saga", penalty="l2")
    ))
])

pipe.fit(X_train, Y_train)
Y_prob = pipe.decision_function(X_test)  # or predict_proba for probas
Y_pred = (pipe.predict_proba(X_test) >= 0.5).astype(int)

# per-label report
for i, a in enumerate(APPS):
    print(f"\n=== {a.upper()} ===")
    if apps_test[a].nunique() < 2:
        print("Not enough class variety in test.")
        continue
    auc = roc_auc_score(apps_test[a].values, Y_prob[:, i])
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    p,r,f1,_ = precision_recall_fscore_support(apps_test[a].values, Y_pred[:, i], average="binary", zero_division=0)
    tn,fp,fn,tp = confusion_matrix(apps_test[a].values, Y_pred[:, i], labels=[0,1]).ravel()
    print(f"AUC={auc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}")

# macro/micro report
from sklearn.metrics import classification_report
print("\n--- Macro/Micro report (threshold=0.5) ---")
print(classification_report(Y_test, Y_pred, target_names=APPS, digits=3))
