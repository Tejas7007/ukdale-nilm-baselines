# linear_reg_kettle.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

CSV_PATH = "b1_kettle_2d_1min.csv"  # created by your previous script

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)

# --- features/target
X = df[["mains_w"]].values  # feature: mains power (W)
y = df["kettle_w"].values   # target: kettle power (W)

# --- time-based split (no shuffling)
split = int(len(df) * 0.8)   # 80% train, 20% test in order
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- fit
model = LinearRegression()
model.fit(X_train, y_train)

# --- evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("coef (slope):", float(model.coef_[0]))
print("intercept   :", float(model.intercept_))
print("R^2         :", round(r2, 4))
print("MAE (watts) :", round(mae, 3))

# --- save a small overlay plot (first 300 test points)
try:
    import matplotlib
    matplotlib.use("Agg")  # no GUI
    import matplotlib.pyplot as plt

    n = min(300, len(y_test))
    xs = np.arange(n)
    plt.figure(figsize=(12,4))
    plt.plot(xs, y_test[:n], label="Actual (kettle W)")
    plt.plot(xs, y_pred[:n], label="Predicted (W)")
    plt.legend()
    plt.title("Baseline Linear Regression — Kettle Watts (test subset)")
    plt.tight_layout()
    plt.savefig("linreg_kettle_test_overlay.png", dpi=150)
    print("✅ Saved: linreg_kettle_test_overlay.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
