import pandas as pd, matplotlib.pyplot as plt, json
from pathlib import Path

pred = pd.read_csv("kettle_predictions.csv", index_col=0, parse_dates=True)
truth = pd.read_csv("b1_kettle_14d_1min.csv", index_col=0, parse_dates=True)
thr = 0.5
p = Path("mlp_kettle_14d_threshold.json")
if p.exists():
    thr = float(json.loads(p.read_text())["threshold"])
df = truth.join(pred, how="inner")[["kettle_w","proba"]]

plt.figure(figsize=(12,4))
plt.plot(df.index, df["proba"])
plt.axhline(thr, ls="--")
plt.title("MLP — Kettle (14d) — Probability w/ tuned threshold")
plt.tight_layout()
Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/mlp_kettle_14d_prob.png", dpi=150)
print("✅ Saved plots/mlp_kettle_14d_prob.png")
