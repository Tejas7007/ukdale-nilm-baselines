# ⚡ UK-DALE NILM — Linear / Logistic / MLP Baselines (Building 1)

This repository contains **baseline experiments** for Non-Intrusive Load Monitoring (NILM) on the **UK-DALE dataset**.

---

## 🧩 Overview
We explore three supervised learning paradigms for appliance-level energy disaggregation:

- **Linear Regression** — power estimation (Watts)
- **Logistic Regression** — ON/OFF classification
- **Neural Network (MLP)** — nonlinear temporal modeling
- **Multi-label pipeline** — joint classification for multiple appliances (kettle, microwave, fridge freezer, toaster)

---

## ⚙️ Dataset Details
**UK-DALE (Kelly & Knottenbelt, 2015)** records individual appliance power at 6-second resolution across several UK homes.

**In this baseline:**
- **Source file:** `ukdale.h5`
- **Building:** B1
- **Appliances:** Kettle, Microwave, Fridge Freezer, Toaster
- **Aggregation:** 1-minute sampling
- **Window:** 14 days (March–April 2013)

All datasets were exported using:
which produce aligned CSVs of mains + sub-metered loads.

---

## 🧠 Model Summary

| Model Type | Objective | Key Script | Output |
|-------------|------------|------------|---------|
| Linear Regression | Power estimation (Watts) | `linear_reg_kettle.py` | `linreg_kettle_test_overlay.png` |
| Logistic Regression | ON/OFF detection | `logistic_kettle_tune.py` | `logreg_kettle_threshold_tuning.png` |
| Neural Network (MLP) | Nonlinear binary classification | `mlp_kettle_14d_noshuffle.py` | `mlp_kettle_probs.png` |
| Multi-label Logistic | Joint appliance classification | `multilabel_logistic_tune.py` | `multilabel_mlp_summary.csv` |
| Multi-label MLP | End-to-end multi-appliance model | `multilabel_mlp_tune.py` | `multilabel_predictions_pp.csv` |

---

## 📊 Results Summary

### 🔹 Single-Appliance (Kettle)

| Model | AUC | Precision | Recall | F1 | Notes |
|--------|-----|-----------|--------|----|-------|
| Linear Regression | — | — | — | MAE ≈ **76 W** | Continuous power-level baseline |
| Logistic Regression | 0.977 | 0.46 | 0.87 | 0.61 | Tuned threshold = 0.61 |
| MLP (7 d) | 0.979 | 0.46 | 0.80 | 0.59 | Rolling window variant |
| MLP (14 d) | 0.9939 | 0.43 | 0.80 | 0.56 | Broader temporal context |
| ✅ Final MLP | 0.999 | 0.83 | 0.85 | 0.84 | **Deployed baseline** |

---

### 🔹 Multi-Appliance (14 d)

| Appliance | AUC | Precision | Recall | F1 | Comment |
|------------|-----|-----------|--------|----|----------|
| **Kettle** | 0.998 | 0.59 | 0.98 | 0.74 | Excellent separability |
| **Microwave** | 0.996 | 0.97 | 0.29 | 0.44 | Low activation frequency |
| **Fridge Freezer** | 0.922 | 0.85 | 0.63 | 0.72 | Quasi-continuous behavior |
| **Toaster** | 0.990 | 0.50 | 0.95 | 0.65 | Short-burst appliance |
| **Macro Avg.** | — | 0.73 | 0.71 | 0.64 | Balanced trade-off overall |

---

## 📈 Visual Results

### 🔹 Sample Alignment
**Mains vs Kettle (2-day preview)**
<br>
<img src="plots/preview.png" width="650"/>

---

### 🔹 Regression & Logistic Baselines

| Linear Regression | Logistic Regression | Rolling Logistic |
|-------------------|--------------------|------------------|
| <img src="plots/linreg_kettle_test_overlay.png" width="250"/> | <img src="plots/logreg_kettle_threshold_tuning.png" width="250"/> | <img src="plots/logreg_kettle_rolling_tuning.png" width="250"/> |

---

### 🔹 Neural Networks (MLP)

| 7-Day MLP | 14-Day MLP | Tuned 14-Day MLP |
|------------|-------------|----------------|
| <img src="plots/mlp_kettle_probs.png" width="250"/> | <img src="plots/mlp_kettle_14d_prob.png" width="250"/> | <img src="plots/mlp_kettle_14d_tuned.png" width="250"/> |

---

### 🔹 Multi-Label Learning

| Kettle | Microwave | Fridge Freezer | Toaster |
|:------:|:----------:|:---------------:|:--------:|
| <img src="plots/multilabel_kettle.png" width="200"/> | <img src="plots/multilabel_microwave.png" width="200"/> | <img src="plots/multilabel_fridge.png" width="200"/> | <img src="plots/multilabel_toaster.png" width="200"/> |

**Aggregate Evaluation**
<br>
<img src="plots/multilabel_confusion.png" width="600"/>

---

## 🚀 Optimization Roadmap

- [ ] Handle **class imbalance** (Weighted BCE / Oversampling)
- [ ] Add **temporal models** (LSTM, Transformer)
- [ ] Improve fridge/toaster labeling consistency
- [ ] Test **cross-building transfer** (B1 → B2)
- [ ] Add **event-level F1** metrics
- [ ] Launch **Streamlit dashboard** for visualization

---

## 🧾 Citation
> Tejas Dahiya (2025).  
> *UK-DALE NILM Baseline Experiments: Linear, Logistic, and MLP Models for Load Disaggregation.*
