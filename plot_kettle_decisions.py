# plot_kettle_decisions.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- paths (edit if your filenames differ) ----
pred_csv  = Path("kettle_predictions.csv")          # has columns: proba, pred_on
truth_csv = Path("b1_kettle_14d_1min.csv")          # has columns incl. 'kettle_w' & 'mains_w'
thr_json  = Path("mlp_kettle_14d_threshold.json")   # optional; falls back to 0.5

# ---- load data ----
pr = pd.read_csv(pred_csv, index_col=0, parse_dates=True)
gt = pd.read_csv(truth_csv, index_col=0, parse_dates=True)

# make sure expected columns exist
if "kettle_w" not in gt.columns:
    raise ValueError("Expected 'kettle_w' in the ground-truth CSV.")
if "proba" not in pr.columns or "pred_on" not in pr.columns:
    raise ValueError("Expected 'proba' and 'pred_on' in kettle_predictions.csv.")

# derive binary ground truth from watts
GT_THRESH_W = 10.0  # tweak if you want more/less strict ON definition
gt["gt_on"] = (gt["kettle_w"] > GT_THRESH_W).astype(int)

# merge by time index
df = gt.join(pr[["proba", "pred_on"]], how="inner")

# load tuned threshold for viz (optional)
thr = 0.5
if thr_json.exists():
    try:
        thr = float(json.loads(thr_json.read_text()).get("threshold", thr))
    except Exception:
        pass

# ---- plot ----
plt.figure(figsize=(13,7))

# top: mains & kettle watts (scale mains so both are visible)
ax1 = plt.subplot(3,1,1)
scale = max(1.0, (df["mains_w"].quantile(0.99) / max(1.0, df["kettle_w"].quantile(0.99))) )
ax1.plot(df.index, df["mains_w"]/scale, label=f"Mains_w / {scale:.1f}")
ax1.plot(df.index, df["kettle_w"], label="Kettle_w")
ax1.set_ylabel("Watts (scaled)")
ax1.legend(loc="upper right")
ax1.set_title("UK-DALE B1 — Kettle: mains vs kettle (14 days)")

# middle: probability with threshold
ax2 = plt.subplot(3,1,2, sharex=ax1)
ax2.plot(df.index, df["proba"], label="Pred prob (ON)")
ax2.axhline(thr, ls="--", label=f"Threshold = {thr:.2f}")
ax2.set_ylabel("Probability")
ax2.legend(loc="upper right")
ax2.set_title("MLP probability and tuned threshold")

# bottom: GT vs prediction as 0/1
ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.step(df.index, df["gt_on"], where="post", label="GT ON")
ax3.step(df.index, df["pred_on"], where="post", label="Pred ON")
ax3.set_ylabel("ON/OFF")
ax3.set_yticks([0,1])
ax3.legend(loc="upper right")
ax3.set_title(f"Decisions (GT threshold {GT_THRESH_W:.0f}W)")

plt.tight_layout()
out = Path("plots/mlp_kettle_14d_decisions.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
print(f"✅ Saved: {out}")
