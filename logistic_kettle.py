# logistic_kettle.py
#Predict kettle ON (1) vs OFF (0) each minute from the mains signal using logistic regression.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

CSV_PATH = "b1_kettle_2d_1min.csv"
ON_THRESHOLD_W = 10.0  # ON if kettle_w >= 10 W

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0) #Reads your 1-minute CSV (timestamps index, columns like mains_w, kettle_w)

# --- target: ON/OFF
y_on = (df["kettle_w"] >= ON_THRESHOLD_W).astype(int)  #Converts continuous watts to binary 0/1 target: 1 = ON, 0 = OFF.

# --- features: mains with lags (t, t-1..t-5)
X = pd.DataFrame(index=df.index) #Creates a new DataFrame X that will hold your features (inputs to the model).
X["mains_t"] = df["mains_w"] #mains_t = current minute.
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k) #This adds five extra columns: Each column contains the mains power reading from k minutes ago.

# align + drop NaNs from shifting
xy = pd.concat([X, y_on.rename("y")], axis=1).dropna() #shift() creates NaNs in the first few rows; dropna() removes them so X and y line up perfectly.
X = xy.drop(columns=["y"]).values
y = xy["y"].values

# --- time-ordered split: 80/20
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- model
clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced") # max_iter=1000 means the model is allowed up to 1000 steps to find the optimal parameters.This prevents the training from stopping too early before convergence
#The solver is the algorithm used to find the best parameters.
#class_weight="balanced": compensates class imbalance (kettle is OFF most minutes). Without this, the model might predict “OFF” too often.
#"lbfgs" (Limited-memory BFGS) is a numerical optimization method that:Works well for small to medium-sized feature sets.
clf.fit(X_train, y_train)

# --- eval
proba = clf.predict_proba(X_test)[:, 1] #predict_proba gives P(ON) per minute.
y_pred = (proba >= 0.5).astype(int) #Classify ON if probability ≥ 0.5.

acc = accuracy_score(y_test, y_pred) #Accuracy: share of correct ON/OFF decisions. Can be misleading with imbalance.
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0) #Precision (of ON): when we predict ON, how often is it truly ON? (Low precision = many false alarms.)
#Recall (of ON): of all true ON minutes, how many did we catch? (Low recall = missed kettle events.)
#F1: harmonic mean of precision & recall (balanced summary).
try:
    auc = roc_auc_score(y_test, proba) #AUC = 1.0 → Perfect model (always ranks real ONs above OFFs)AUC = 0.5 → Random guessing (the curve is a diagonal line)
except Exception:
    auc = float("nan") #But this function will fail in some cases: If your test set contains only one class (e.g. all OFFs or all ONs).In that case, AUC is undefined — because you need both positives and negatives to compute a meaningful curve.
cm = confusion_matrix(y_test, y_pred)
#|                    |    Predicted OFF (0) |     Predicted ON (1) |
#| ------------------ | -------------------: | -------------------: |
#| **Actual OFF (0)** |  True Negatives (TN) | False Positives (FP) |
#| **Actual ON (1)**  | False Negatives (FN) |  True Positives (TP) |

print("Accuracy :", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("F1       :", round(f1, 4))
print("ROC AUC  :", round(auc, 4))
print("Confusion matrix:\n", cm)

# optional: save a quick overlay plot of probabilities vs label
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(300, len(y_test)) #We’ll only plot up to 300 samples, to keep the graph readable.
    xs = np.arange(n) #Creates an array [0, 1, 2, ..., n-1] for the x-axis.ach value represents one time step (e.g., each minute).
    plt.figure(figsize=(12,4))
    plt.plot(xs, y_test[:n], label="Actual ON(1)/OFF(0)") #Plots the true labels (y_test) for the first n samples.
    plt.plot(xs, proba[:n], label="Pred prob(ON)") #Plots the model’s predicted probability of the kettle being ON.
    plt.legend() #Adds a small box explaining which color corresponds to actual vs predicted.
    plt.title("Logistic Regression — Kettle ON/OFF (test subset)")
    plt.tight_layout()
    plt.savefig("logreg_kettle_probs.png", dpi=150)
    print("✅ Saved: logreg_kettle_probs.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
