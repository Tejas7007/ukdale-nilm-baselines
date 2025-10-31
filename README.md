<h1 align="center">⚡ UK-DALE NILM — Linear / Logistic / MLP Baselines</h1>
<p align="center">
  <b>Building-1, 1-min aggregation, 14-day window</b><br/>
  Linear Regression · Logistic Regression · MLP · Multi-Label Learning
</p>

---

## 📌 Overview
This project builds strong **Non-Intrusive Load Monitoring (NILM)** baselines on the UK-DALE dataset using:
- **Linear Regression** — Watt-level estimation  
- **Logistic Regression** — Binary ON/OFF classification  
- **Multi-Layer Perceptron (MLP)** — Context-aware deep learning  
- **Multi-label joint models** — Kettle, Microwave, Fridge-Freezer, Toaster  

Each model was trained on **Building 1**, aggregated at **1-minute resolution**, using **14-day windows** (March–April 2013).

---

## 📊 Model Performance Summary

<details>
<summary><b>Single-Appliance (Kettle)</b></summary>

| Model | AUC | Precision | Recall | F1 | Note |
|---|---:|---:|---:|---:|---|
| Linear Regression | — | — | — | **MAE ≈ 76 W** | Power-level baseline |
| Logistic (tuned) | **0.977** | 0.46 | 0.87 | 0.61 | Threshold tuned for recall |
| MLP (14d tuned) | **0.999** | **0.83** | **0.85** | **0.84** | ✅ Deployed baseline |
</details>

<details>
<summary><b>Multi-Appliance (14d MLP)</b></summary>

| Appliance | AUC | Precision | Recall | F1 | Comment |
|---|---:|---:|---:|---:|---|
| Kettle | 0.995–0.998 | 0.47–0.83 | 0.85–0.98 | 0.61–0.84 | Sharp spikes |
| Microwave | 0.986–0.996 | 0.71–0.97 | 0.29–0.39 | 0.44–0.50 | Sparse events |
| Fridge-Freezer | 0.922–0.954 | **0.85–0.87** | **0.63–0.75** | **0.72–0.81** | Quasi-always-on |
| Toaster | 0.990–0.938 | 0.34–1.00 | 0.67–0.95 | 0.45–0.65 | Tiny bursts |
</details>

---

## 📈 Visual Gallery

### 🔹 Alignment (Ground Truth vs Mains)
<p align="center">
  <img src="plots/preview.png" width="750" alt="Mains vs Kettle preview"/>
</p>

### 🔹 Regression & Logistic Models
<p align="center">
  <img src="plots/linreg_kettle_test_overlay.png" width="240" />
  <img src="plots/logreg_kettle_threshold_tuning.png" width="240" />
  <img src="plots/logreg_kettle_rolling_tuning.png" width="240" />
</p>

### 🔹 Neural Networks (MLP)
<table>
  <tr>
    <td align="center">
      <img src="plots/mlp_kettle_probs.png" width="320" />
      <br/><sub>MLP probability trajectory (7–14d context).</sub>
    </td>
    <td align="center">
      <img src="plots/mlp_kettle_14d_prob.png" width="320" />
      <br/><sub>14-day context boosts recall and smooths predictions.</sub>
    </td>
    <td align="center">
      <img src="plots/mlp_kettle_14d_decisions.png" width="320" />
      <br/><sub>Tuned decision overlay (thresholded ON/OFF).</sub>
    </td>
  </tr>
</table>

### 🔹 Multi-Label Overview
<p align="center">
  <img src="plots/multilabel_confusion.png" width="780" />
  <br/><sub>Aggregated confusion across Kettle, Microwave, Fridge-Freezer, Toaster.</sub>
</p>

---

## 🧠 Key Insights

- **Temporal context (14d > 7d)** significantly boosts recall for short-duration spikes.  
- **Threshold tuning (≥0.85 recall)** balances precision–recall trade-offs.  
- **Fridge-Freezer & Toaster** gain the most from nonlinear MLP transformations.  
- **Multi-label joint setup** enhances appliance robustness and generalization.  
- **Rolling logistic regression** improves prediction stability for noisy loads.

---

## 🚀 Next Steps
- 🧩 Handle class imbalance via weighted BCE or SMOTE.  
- 🧮 Integrate **Bayesian optimization** for hyperparameter tuning.  
- ⚙️ Extend to **sequence models (LSTM, Transformer)** for temporal reasoning.  
- 🏠 Test **cross-building transfer** (B1 → B2–B5).  
- 🧭 Add a lightweight **Streamlit dashboard** for visual NILM comparisons.

---

## 🧾 License
This repository is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 📚 Citation
If you use this work, please cite:

