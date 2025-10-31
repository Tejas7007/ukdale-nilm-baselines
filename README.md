Overview

This repository contains baseline experiments for Non-Intrusive Load Monitoring (NILM) using the UK-DALE dataset (Building 1 slice).
The goal is to predict appliance-level consumption or ON/OFF states from aggregated mains readings.

We explored three supervised learning paradigms:

Linear Regression — continuous power estimation

Logistic Regression — binary ON/OFF detection

Neural Networks (MLP) — nonlinear temporal modeling

Multi-label models — joint prediction for multiple appliances

⚙️ Dataset Details

UK-DALE (Kelly & Knottenbelt, 2015) records individual appliance power at 6-second resolution across several UK homes.

In this baseline:

Source file: ukdale.h5

Building: B1

Appliances: Kettle, Microwave, Fridge Freezer, Toaster

Aggregation: 1-minute sampling

Window: 14 days (March–April 2013)

All datasets were exported via view_ukdale.py and build_multi_appliance_csv.py into aligned CSVs (mains + sub-metered loads).

🧠 Model Summary
Model Type	Objective	Key Script	Output
Linear Regression	Power estimation (Watts)	linear_reg_kettle.py	linreg_kettle_test_overlay.png
Logistic Regression	ON/OFF classification	logistic_kettle_tune.py	logreg_kettle_threshold_tuning.png
Neural Network (MLP)	Nonlinear binary classification	mlp_kettle_14d_noshuffle.py	mlp_kettle_probs.png
Multi-label Logistic	Joint appliance classification	multilabel_logistic_tune.py	multilabel_mlp_summary.csv
Multi-label MLP	End-to-end nonlinear multi-appliance prediction	multilabel_mlp_tune.py	multilabel_predictions_pp.csv
📊 Results Summary
🔹 Single-Appliance (Kettle)
Model	AUC	Precision	Recall	F1	Notes
Linear Regression	—	—	—	MAE ≈ 76 W	Power-level baseline
Logistic Regression	0.977	0.46	0.87	0.61	Tuned threshold = 0.61
MLP (7 d)	0.979	0.46	0.80	0.59	Rolling window
MLP (14 d)	0.9939	0.43	0.80	0.56	Broader context improves recall
MLP (final tuned)	0.999	0.83	0.85	0.84	✅ Deployed baseline
🔹 Multi-Appliance (14 d)
Appliance	AUC	Precision	Recall	F1	Comment
Kettle	0.998	0.59	0.98	0.74	Excellent separability
Microwave	0.996	0.97	0.29	0.44	Low activation frequency
Fridge Freezer	0.922	0.85	0.63	0.72	Quasi-continuous behavior
Toaster	0.990	0.50	0.95	0.65	Short bursts detected
Macro Avg	—	0.73	0.71	0.64	Balanced tradeoff overall
```bash
conda create -n ukdale311 python=3.11 -y
conda activate ukdale311
pip install -r requirements.txt
pip install "git+https://github.com/nilmtk/nilmtk.git"


---

### 🧩 Step 5 — Initialize Git repo and make your first commit
```bash
git init
git add -A
git commit -m "Initial commit: UK-DALE NILM baseline experiments (linear, logistic, MLP, multi-label)"

git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/ukdale-nilm-baselines.git
git push -u origin main

