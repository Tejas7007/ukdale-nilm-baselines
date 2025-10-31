# UK-DALE NILM — Linear / Logistic / MLP Baselines (B1 slice)

This repo contains all baseline experiments run on the **UK-DALE** dataset:
- Linear Regression (Power estimation)
- Logistic Regression (On/Off detection)
- Neural Network (MLP) per-appliance
- Multi-label pipeline (kettle, microwave, fridge freezer, toaster)

## Environment
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

