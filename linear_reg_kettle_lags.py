# linear_reg_kettle_lags.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "b1_kettle_2d_1min.csv"

# --- load
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)

# --- feature engineering: lags of mains (1..5 minutes)
X = pd.DataFrame(index=df.index)
X["mains_t"] = df["mains_w"]
for k in range(1, 6):
    X[f"mains_t_minus_{k}"] = df["mains_w"].shift(k)

y = df["kettle_w"]

# drop rows with NaNs from shifting
xy = pd.concat([X, y], axis=1).dropna()
X = xy.drop(columns=["kettle_w"]).values
y = xy["kettle_w"].values

# time-ordered split (no shuffle)
split = int(len(xy) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- fit
model = LinearRegression()
model.fit(X_train, y_train)

# --- eval
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Features:", [f"mains_t_minus_{k}" if k>0 else "mains_t" for k in range(0,6)])
print("coef:", np.round(model.coef_, 4))
print("intercept:", round(float(model.intercept_), 4))
print("R^2:", round(r2, 4))
print("MAE (watts):", round(mae, 3))

# quick plot of first 300 test points
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(300, len(y_test))
    xs = np.arange(n)
    plt.figure(figsize=(12,4))
    plt.plot(xs, y_test[:n], label="Actual (kettle W)")
    plt.plot(xs, y_pred[:n], label="Predicted (W)")
    plt.legend()
    plt.title("Linear Regression with Lags — Kettle (test subset)")
    plt.tight_layout()
    plt.savefig("linreg_kettle_lags_overlay.png", dpi=150)
    print("✅ Saved: linreg_kettle_lags_overlay.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
