<!-- Centered title -->
<h1 align="center">⚡ UK-DALE NILM — Linear / Logistic / MLP Baselines</h1>
<p align="center">
  <b>Building-1 · 1-min Aggregation · 14-Day Window</b><br/>
  Linear Regression · Logistic Regression · MLP · Multi-label Classification
</p>

---

## 📌 TL;DR
We build robust baselines for **Non-Intrusive Load Monitoring (NILM)** on the UK-DALE dataset:
- **Single-appliance:** Kettle (regression + classification)
- **Multi-appliance:** Kettle, Microwave, Fridge-Freezer, Toaster (multi-label)
- Significant improvement from **logistic → MLP**, and from **7-day → 14-day** context

---

## 🧠 Model Performance Summary

### Single-Appliance (Kettle)
| Model | AUC | Precision | Recall | F1 | Notes |
|:--|--:|--:|--:|--:|--|
| Linear Regression | — | — | — | — | **MAE ≈ 76 W**, stable watt-level baseline |
| Logistic (tuned) | **0.977** | 0.46 | 0.87 | 0.61 | Recall-optimized threshold |
| MLP (14d tuned) | **0.999** | **0.83** | **0.85** | **0.84** | ✅ Best single-appliance model |

### Multi-Appliance (14-Day MLP Tuned)
| Appliance | AUC | Precision | Recall | F1 | Comments |
|:--|--:|--:|--:|--:|--|
| **Kettle** | 0.995–0.998 | 0.47–0.83 | 0.85–0.98 | 0.61–0.84 | Distinct high spikes → easy |
| **Microwave** | 0.986–0.996 | 0.71–0.97 | 0.29–0.39 | 0.44–0.50 | Sparse short events |
| **Fridge-Freezer** | 0.922–0.954 | **0.85–0.87** | **0.63–0.75** | **0.72–0.81** | Quasi-steady appliance |
| **Toaster** | 0.938–0.990 | 0.34–1.00 | 0.67–0.95 | 0.45–0.65 | Low-duration peaks |

> 📄 Full metrics in `multilabel_eval_summary.csv`  
> 🎯 Thresholds stored in `multilabel_thresholds.json`

---

## 📈 Visual Results

### 🧩 Alignment (Ground Truth vs Mains)
<p align="center">
  <img src="plots/preview.png" width="780" alt="Mains vs Kettle preview"/>
  <br/>
  <sub>Mains vs Kettle: two-day slice used to verify alignment and sampling.</sub>
</p>

---

### ⚙️ Regression & Logistic Models
<table>
  <tr>
    <td align="center">
      <img src="plots/linreg_kettle_test_overlay.png" width="250" alt="Linear regression"/><br/>
      <sub>Linear regression: predicted wattage vs true usage.</sub>
    </td>
    <td align="center">
      <img src="plots/logreg_kettle_threshold_tuning.png" width="250" alt="Logistic tuning"/><br/>
      <sub>Logistic regression: precision–recall vs threshold.</sub>
    </td>
    <td align="center">
      <img src="plots/logreg_kettle_rolling_tuning.png" width="250" alt="Rolling logistic"/><br/>
      <sub>Rolling logistic: lagged features improve stability.</sub>
    </td>
  </tr>
</table>

---

### 🤖 Neural Networks (MLP)
<table>
  <tr>
    <td align="center">
      <img src="plots/mlp_kettle_probs.png" width="250" alt="MLP kettle probabilities"/><br/>
      <sub>MLP probability trajectory (7–14d context).</sub>
    </td>
    <td align="center">
      <img src="plots/mlp_kettle_14d.png" width="250" alt="MLP 14d"/><br/>
      <sub>14-day context boosts recall and smooths predictions.</sub>
    </td>
    <td align="center">
      <img src="plots/mlp_kettle_14d_tuned.png" width="250" alt="MLP tuned"/><br/>
      <sub>Tuned threshold for recall ≥ 0.85.</sub>
    </td>
  </tr>
</table>

---

### 🧮 Multi-Label Overview
<p align="center">
  <img src="plots/multilabel_confusion.png" width="780" alt="Multilabel confusion matrix"/>
  <br/>
  <sub>Aggregated confusion across Kettle, Microwave, Fridge-Freezer, and Toaster.</sub>
</p>

---

## 🔬 Key Insights
- **Temporal context** (14d > 7d) significantly improves recall for short-duration spikes.  
- **Threshold tuning** by target recall (≥ 0.85) balances precision–recall trade-offs.  
- **Fridge-Freezer** and **Toaster** benefit from nonlinear transformations (MLP).  
- **Multi-label setup** enhances rare appliance robustness.  
- **Rolling logistic** provides smoother prediction continuity.

---

## 🛣️ Next Steps
- 📈 Handle **class imbalance** via weighted BCE or SMOTE.  
- 🧠 Integrate **Bayesian optimization** for hyperparameter tuning.  
- 🔁 Add **sequence models** (LSTM, Transformer) for long-range context.  
- 🏠 Extend training to **Buildings 2–5** for cross-domain generalization.  
- 🌐 Build **Streamlit dashboard** for interactive NILM visualization.  

---

## ⚙️ Setup

```bash
conda create -n ukdale311 python=3.11 -y
conda activate ukdale311
pip install -r requirements.txt
pip install "git+https://github.com/nilmtk/nilmtk.git"

