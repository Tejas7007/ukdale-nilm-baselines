import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np

GT_CSV   = "b1_MULTI_14d_1min.csv"            # ground truth file
PRED_CSV = "multilabel_predictions_pp.csv"     # or multilabel_predictions.csv
APPS = ["kettle", "microwave", "fridge freezer", "toaster"]
ON_W = {"default": 10.0, "kettle":10.0, "microwave":30.0, "fridge freezer":60.0, "toaster":25.0}

# load
gt = pd.read_csv(GT_CSV, parse_dates=[0], index_col=0)
pr = pd.read_csv(PRED_CSV, parse_dates=[0], index_col=0)

# only bring prediction columns to avoid name collisions like 'mains_w'
pred_cols = [c for c in pr.columns if c.endswith("_proba") or c.endswith("_on")]
df = gt.join(pr[pred_cols], how="inner")

rows = []
for a in APPS:
    gt_col = f"{a}_w"
    if gt_col not in df.columns:
        continue

    thr = ON_W.get(a, ON_W["default"])
    y_true = (df[gt_col] >= thr).astype(int)

    key = a.replace(" ", "_")
    proba_col = f"{key}_proba"
    on_col    = f"{key}_on"
    if proba_col not in df.columns or on_col not in df.columns:
        continue

    y_prob = df[proba_col].to_numpy()
    y_pred = df[on_col].to_numpy()

    # metrics (handle degenerate class cases safely)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = float("nan")
    if y_true.nunique() == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    rows.append({
        "appliance": a,
        "AUC": auc,
        "precision": p,
        "recall": r,
        "F1": f1,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "positives": int(y_true.sum())
    })

res = pd.DataFrame(rows).sort_values("appliance")
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(res.to_string(index=False))
res.to_csv("multilabel_eval_summary.csv", index=False)
print("\n✅ Saved: multilabel_eval_summary.csv")
