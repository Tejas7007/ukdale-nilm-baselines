# build_multi_appliance_csv.py
from nilmtk import DataSet
import pandas as pd
from datetime import timedelta

H5 = "/Users/tejas/Downloads/ukdale.h5"
BUILDING = 1
APPLIANCES = ["kettle", "microwave", "fridge freezer", "toaster"]  # edit list if you like
SAMPLE_SEC = 60
DAYS = 14

def to_naive_utc(ts):
    # ts is tz-aware; convert to UTC then drop tz (naive)
    return ts.tz_convert("UTC").tz_localize(None)

ds = DataSet(H5)
b = ds.buildings[BUILDING]
elec = b.elec

# --- choose a 14d window from the dataset start ---
tf = elec.mains().get_timeframe()
start_naive = to_naive_utc(tf.start)
end_naive   = start_naive + timedelta(days=DAYS)

# IMPORTANT: set a NAIVE window (no timezone); NILMTK will localize internally
ds.set_window(start=start_naive, end=end_naive)

# --- mains (1-min) ---
mains = next(elec.mains().power_series(sample_period=SAMPLE_SEC)).rename("mains_w")

# --- per-appliance series ---
dfs = [mains.to_frame()]
for name in APPLIANCES:
    sel = elec.select_using_appliances(type=name)
    if not sel.meters:
        print(f"⚠️ No meter found for '{name}', skipping.")
        continue
    s = next(sel.meters[0].power_series(sample_period=SAMPLE_SEC))
    dfs.append(s.rename(f"{name}_w").to_frame())

# --- align & small gap fill ---
df = pd.concat(dfs, axis=1).sort_index()
df = df.ffill(limit=2)                 # fill up to 2-minute gaps
df = df.dropna(subset=["mains_w"])     # require mains

print("Columns:", list(df.columns))
print("Rows:", len(df), "| Time:", df.index.min(), "→", df.index.max())

OUT = "b1_MULTI_14d_1min.csv"
df.to_csv(OUT)
print(f"✅ Saved {OUT}")

ds.store.close()
