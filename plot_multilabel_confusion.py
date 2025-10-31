import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def norm_idx_utc_minute(idx):
    # Force to datetime, ensure tz-aware UTC, align to minute
    idx = pd.to_datetime(idx, errors="coerce", utc=True)
    return idx.floor("T")

# --- Load ground truth & predictions ---
gt = pd.read_csv("b1_MULTI_14d_1min.csv", parse_dates=[0])
pr = pd.read_csv("multilabel_predictions_pp.csv", parse_dates=[0])

gt.set_index(gt.columns[0], inplace=True)
pr.set_index(pr.columns[0], inplace=True)

# Align to UTC minute bins
gt.index = norm_idx_utc_minute(gt.index)
pr.index = norm_idx_utc_minute(pr.index)

# Drop mains if present on either side to avoid column overlap
gt.drop(columns=["mains_w"], inplace=True, errors="ignore")
pr.drop(columns=["mains_w"], inplace=True, errors="ignore")

# Map GT appliance power columns → normalized names
appliance_map = {
    "kettle": "kettle_w",
    "microwave": "microwave_w",
    "fridge_freezer": "fridge freezer_w",  # GT has a space here
    "toaster": "toaster_w",
}

# Ground truth (binary > 10 W)
gt_bin = pd.DataFrame(index=gt.index)
for ap_norm, gt_col in appliance_map.items():
    if gt_col in gt.columns:
        gt_bin[f"{ap_norm}_gt"] = (gt[gt_col].fillna(0) > 10).astype(int)

# Predictions: use *_on columns (already 0/1)
pred_bin = pd.DataFrame(index=pr.index)
for ap_norm in appliance_map.keys():
    # Will accept either underscore or space base names
    cand_cols = [
        f"{ap_norm}_on",
        f"{ap_norm.replace('_',' ')}_on",
    ]
    for c in cand_cols:
        if c in pr.columns:
            pred_bin[f"{ap_norm}_pred"] = pr[c].astype(int)
            break

# Inner-join on overlapping timestamps
df = gt_bin.join(pred_bin, how="inner")
if df.empty:
    raise RuntimeError("No overlapping rows after join; check time alignment.")

# Ensure we have both GT & pred for these appliances
appliances = [a for a in appliance_map if f"{a}_gt" in df.columns and f"{a}_pred" in df.columns]
if not appliances:
    raise RuntimeError("No overlapping appliance columns found (GT vs predictions).")

print("Using appliances:", appliances)

# --- Plot per-appliance confusion matrices ---
sns.set_context("talk")
rows, cols = 2, 2
fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
axes = axes.ravel()

for i, ap in enumerate(appliances):
    ax = axes[i]
    y_true = df[f"{ap}_gt"].values
    y_pred = df[f"{ap}_pred"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    sns.heatmap(np.array([[tn, fp], [fn, tp]]),
                annot=True, fmt="d", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["GT 0", "GT 1"])
    ax.set_title(f"{ap.replace('_',' ').title()} — P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

# Hide any unused subplots if fewer than 4
for j in range(i+1, rows*cols):
    fig.delaxes(axes[j])

plt.suptitle("UK-DALE B1 — Multi-Label Confusion Matrices (14d, 1-min)", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out = "plots/multilabel_confusion.png"
plt.savefig(out, dpi=160)
print(f"✅ Saved {out}")
