<!-- Centered title -->
<h1 align="center">⚡ UK-DALE NILM — Linear / Logistic / MLP Baselines</h1>
<p align="center">
  <b>Building-1, 1-min aggregation, 14-day window</b><br/>
  Linear Regression · Logistic Regression · MLP · Multi-label
</p>

---

## 📌 TL;DR
We build strong baselines for **Non-Intrusive Load Monitoring (NILM)** on UK-DALE:
- Single-appliance: Kettle (regression + classification)
- Multi-appliance: Kettle, Microwave, Fridge-Freezer, Toaster (multi-label)
- Clear improvements from logistic→MLP and from 7-day→14-day context

---

## 🧠 Key Results

### Single-appliance (Kettle)
| Model | AUC | Precision | Recall | F1 | Note |
|---|---:|---:|---:|---:|---|
| Linear Regression | — | — | — | **MAE ≈ 76 W** | Watt-level baseline |
| Logistic (tuned) | **0.977** | 0.46 | 0.87 | 0.61 | Threshold selected for recall |
| MLP (14d tuned) | **0.999** | **0.83** | **0.85** | **0.84** | ✅ Deployed baseline |

### Multi-appliance (14d, MLP tuned)
| Appliance | AUC | Precision | Recall | F1 | Comment |
|---|---:|---:|---:|---:|---|
| Kettle | 0.995–0.998 | 0.47–0.83 | 0.85–0.98 | 0.61–0.84 | Distinct spikes → easy |
| Microwave | 0.986–0.996 | 0.71–0.97 | 0.29–0.39 | 0.44–0.50 | Sparse events |
| Fridge Freezer | 0.922–0.954 | **0.85–0.87** | **0.63–0.75** | **0.72–0.81** | Quasi-always-on |
| Toaster | 0.990–0.938 | 0.34–1.00 | 0.67–0.95 | 0.45–0.65 | Tiny bursts |

> Full metrics in `multilabel_eval_summary.csv` and per-label thresholds in `multilabel_thresholds.json`.

---

## 📈 Visual Results

### Alignment (ground truth vs mains)
<p align="center">
  <img src="plots/preview.png" width="780" alt="Mains vs Kettle (preview)"/>
  <br/><sub>Mains vs Kettle, 2-day slice used to sanity-check alignment/resampling.</sub>
</p>

### Regression & Logistic
<table>
  <tr>
    <td align="center">
      <img src="plots/linreg_kettle_test_overlay.png" width="250" alt="Linear regression overlay"/><br/>
      <sub>Linear regression: watt-level fit vs truth.</sub>
    </td>
    <td align="center">
      <img src="plots/logreg_kettle_threshold_tuning.png" width="250" alt="Logistic threshold tuning"/><br/>
      <sub>Logistic: precision/recall vs threshold.</sub>
    </td>
    <td align="center">
      <img src="plots/logreg_kettle_rolling_tuning.png" width="250" alt="Rolling logistic tuning"/><br/>
      <sub>Lagged features improve stability.</sub>
    </td>
  </tr>
</table>

### Neural Networks (MLP)
<table>
  <tr>
    <td align="center">
      <img src="plots/mlp_kettle_probs.png" width="250" alt="MLP kettle probabilities"/><br/>
      <sub>MLP probability trajectory (7–14d).</sub>
    </td>
    <!-- If you saved separate 14d/tuned images, keep them; else remove the next cells -->
    <!-- <td align="center">
      <img src="plots/mlp_kettle_14d_prob.png" width="250" alt="MLP 14d"/>
      <br/><sub>14-day context boosts recall.</sub>
    </td>
    <td align="center">
      <img src="plots/mlp_kettle_14d_tuned.png" width="250" alt="MLP tuned"/>
      <br/><sub>Tuned threshold for target recall.</sub>
    </td> -->
  </tr>
</table>

### Multi-label Overview
<p align="center">
  <img src="plots/multilabel_confusion.png" width="780" alt="Multilabel summary"/>
  <br/><sub>Aggregate confusion/summary across Kettle, Microwave, Fridge-Freezer, Toaster.</sub>
</p>

---

## 🔬 What made the difference?
- **Temporal context** (14d > 7d) improved **recall** for rare spikes.
- **Threshold tuning** by target recall (≥0.8/0.85) gave controllable trade-offs.
- For **fridge-freezer**, steady load benefits from **nonlinear** features (MLP).

---

## 🛣️ Next Steps (tracked as Issues)
- Class imbalance: **weighted BCE** / oversampling
- **Event-level metrics** (segment-F1, latency)
- **Sequence models** (LSTM / Transformer) for long-range context
- Cross-building generalization (B1→B2/3/4/5)
- Lightweight **Streamlit** dashboard for exploration

---
