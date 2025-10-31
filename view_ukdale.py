# view_ukdale.py
from nilmtk import DataSet
import os
import warnings
from tables import exceptions as tbx
import pandas as pd

# ---- CONFIG ----
DATA_PATH = "/Users/tejas/Downloads/ukdale.h5"  # update if needed
BUILDING_ID = 1
PREFERRED_APPLIANCE = "kettle"
WINDOW_DAYS_OFFSET = 7      # start = dataset_start + this many days
WINDOW_DAYS_LENGTH = 14      # window size (days)
RESAMPLE_RULE = "1min"

# ---- silence HDF5 temp warnings ----
warnings.filterwarnings("ignore", category=tbx.UnclosedFileWarning)

# ---- helpers ----
def normalize_app_name(app_type) -> str:
    """Return lowercase string for appliance type."""
    if isinstance(app_type, dict):
        return str(app_type.get("type", "")).lower()
    return str(app_type).lower()

def clean_and_resample(s: pd.Series, rule: str) -> pd.Series:
    """Drop duplicate timestamps, coerce numeric, resample to rule (mean)."""
    s = s[~s.index.duplicated(keep="first")]
    s = pd.to_numeric(s, errors="coerce")
    return s.resample(rule).mean()

# ---- open dataset ----
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

ds = DataSet(DATA_PATH)
print("✅ Dataset loaded!")

try:
    b = ds.buildings[BUILDING_ID]

    # ---- list appliances ----
    print(f"\n🏠 Building {BUILDING_ID} appliances (type, instance):")
    apps = []
    for app in b.elec.appliances:
        inst = app.metadata.get("instance", "?")
        name = normalize_app_name(app.type)
        print(f" - {name}, instance={inst}")
        apps.append((name, inst))
    if not apps:
        raise RuntimeError("No appliance meters found in this building.")

    # pick target appliance
    if any(name == PREFERRED_APPLIANCE for name, _ in apps):
        target_name = PREFERRED_APPLIANCE
        target_inst = next(inst for name, inst in apps if name == target_name)
    else:
        target_name, target_inst = apps[0]
        print(f"\nℹ️ '{PREFERRED_APPLIANCE}' not found; using '{target_name}' instead.")

    elec = b.elec

    # ---- choose a small window and fix tz ----
    tf = elec.mains().get_timeframe()
    print(f"\n🕒 Available timeframe: {tf.start} → {tf.end}")

    start = tf.start + pd.Timedelta(days=WINDOW_DAYS_OFFSET)
    end   = start + pd.Timedelta(days=WINDOW_DAYS_LENGTH)
    print(f"🪟 Using window: {start} → {end}")

    # Convert tz-aware → naive UTC
    start_naive = start.tz_convert("UTC").tz_localize(None)
    end_naive   = end.tz_convert("UTC").tz_localize(None)

    # Apply window (no tz arg)
    ds.set_window(start=start_naive, end=end_naive)

    # ---- load raw series ----
    mains_raw = next(elec.mains().power_series())
    target_meters = elec.select_using_appliances(type=target_name).meters
    target_raw = next(target_meters[0].power_series())

    # ---- clean + resample to 1 min ----
    mains_1m  = clean_and_resample(mains_raw, RESAMPLE_RULE).rename("mains_w")
    target_1m = clean_and_resample(target_raw, RESAMPLE_RULE).rename(f"{target_name}_w")

    # ---- align, drop NaNs ----
    df = pd.concat([mains_1m, target_1m], axis=1).dropna()
    print("\n🔎 Aligned sample:")
    print(df.head())

    # ---- save outputs ----
    out_csv = f"b{BUILDING_ID}_{target_name}_{WINDOW_DAYS_LENGTH}d_{RESAMPLE_RULE}.csv"
    df.to_csv(out_csv, index=True)
    print(f"\n✅ Saved aligned CSV: {out_csv} (rows={len(df)})")

    # ---- save plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(500, len(df))
        plt.figure(figsize=(12, 4))
        plt.plot(df.index[:n], df["mains_w"].iloc[:n], label="Mains (W)")
        plt.plot(df.index[:n], df[f"{target_name}_w"].iloc[:n], label=f"{target_name.title()} (W)")
        plt.legend()
        plt.title(f"UK-DALE — Building {BUILDING_ID} (first {n} minutes)\nMains vs. {target_name.title()}")
        plt.tight_layout()
        out_png = "preview.png"
        plt.savefig(out_png, dpi=150)
        print(f"✅ Saved plot: {out_png}")
    except Exception as e:
        print(f"(Plot skipped: {e})")

finally:
    try:
        ds.set_window()  # reset
    except Exception:
        pass
    ds.store.close()
