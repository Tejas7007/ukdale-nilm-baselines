# linear_reg_kettle.py
# We want to train a simple linear regression model that learns the relationship between the mains power and the kettle power for Building 1, based on your 1-minute data.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

CSV_PATH = "b1_kettle_2d_1min.csv"  # created by your previous script

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0) #parse_dates=[0] → treats the first column as timestamps.index_col=0 → makes the timestamps the DataFrame index.

# --- features/target
X = df[["mains_w"]].values  # feature: mains power (W)
y = df["kettle_w"].values   # target: kettle power (W)

# X = feature (mains power readings in watts).
# y = target (actual kettle power).

# --- time-based split (no shuffling)
split = int(len(df) * 0.8)   # 80% train, 20% test in order
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:] #y[:split] → training target values, y[split:] → testing target values

# --- fit
model = LinearRegression()
model.fit(X_train, y_train)

# --- evaluate
y_pred = model.predict(X_test) #Uses the model to predict kettle power for the test data.
r2 = r2_score(y_test, y_pred) #R² score measures goodness of fit: 1.0 = perfect, 0.0 = useless (same as mean).
mae = mean_absolute_error(y_test, y_pred) #MAE (Mean Absolute Error) measures average prediction error in watts.

print("coef (slope):", float(model.coef_[0]))
print("intercept   :", float(model.intercept_))
print("R^2         :", round(r2, 4))
print("MAE (watts) :", round(mae, 3))

# --- save a small overlay plot (first 300 test points)
try:
    import matplotlib
    matplotlib.use("Agg")  # no GUI, Don’t open a window, just render the plot straight to a file.
    import matplotlib.pyplot as plt

    n = min(300, len(y_test)) #ensures you only plot up to 300 points.
    xs = np.arange(n) #np.arange(n) makes an array [0, 1, 2, …, 299] for the x-axis (acts like time steps).
    plt.figure(figsize=(12,4)) #Makes a new plot canvas that’s 12 inches wide × 4 inches tall.
    plt.plot(xs, y_test[:n], label="Actual (kettle W)")
    plt.plot(xs, y_pred[:n], label="Predicted (W)")
    plt.legend()
    plt.title("Baseline Linear Regression — Kettle Watts (test subset)")
    plt.tight_layout()
    plt.savefig("linreg_kettle_test_overlay.png", dpi=150)
    print("✅ Saved: linreg_kettle_test_overlay.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
