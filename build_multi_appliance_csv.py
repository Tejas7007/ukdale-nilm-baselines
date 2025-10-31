# build_multi_appliance_csv.py
#The script loads Building 1 from the UK-DALE dataset, selects the first 14 days of data, resamples everything to 1-minute intervals.
# it merges mains and selected appliance power readings into one aligned DataFrame, fills tiny gaps, and exports the result to a CSV for easy analysis or Excel viewing.
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
elec = b.elec #Gets that building’s ElectricalMeterGroup, i.e., all its mains and sub-meters.

# --- choose a 14d window from the dataset start --- (Just look at the first 14 days of data, in UTC, timezone-free)
tf = elec.mains().get_timeframe() #Fetches the full time range available for the building’s mains meter.
start_naive = to_naive_utc(tf.start) # to_naive_utc() converts the start time to UTC and removes its timezone tag.
end_naive   = start_naive + timedelta(days=DAYS)

# IMPORTANT: set a NAIVE window (no timezone); NILMTK will localize internally
ds.set_window(start=start_naive, end=end_naive) #Restricts the active dataset window to that 14-day range.

# --- mains (1-min) ---
mains = next(elec.mains().power_series(sample_period=SAMPLE_SEC)).rename("mains_w") #Reads the aggregate mains power readings, resampled to 60 seconds (1 minute).

# .rename("mains_w") names the column “mains_w” (watts).

# --- per-appliance series ---
dfs = [mains.to_frame()] #Starts a list of DataFrames with the mains as the first column.
for name in APPLIANCES:
    sel = elec.select_using_appliances(type=name) #looks inside the dataset metadata to find which meter corresponds to that appliance.
    if not sel.meters: # If that appliance doesn’t exist, it prints a warning and skips it.
        print(f"⚠️ No meter found for '{name}', skipping.") 
        continue
    s = next(sel.meters[0].power_series(sample_period=SAMPLE_SEC)) #Otherwise, it loads the meter’s 1-minute power data (power_series), converts it to a pandas DataFrame, renames the column ("kettle_w", "microwave_w", etc.), and adds it to the list dfs.
    dfs.append(s.rename(f"{name}_w").to_frame())

# --- align & small gap fill --- #Concatenates all individual DataFrames horizontally (column-wise) into one big table.
df = pd.concat(dfs, axis=1).sort_index()
df = df.ffill(limit=2)                 # fill up to 2-minute gaps , Fills small missing gaps (up to 2 minutes) using the previous known value.
df = df.dropna(subset=["mains_w"])     # require mains , #Drops any rows where the mains reading is missing — mains is required as reference.

print("Columns:", list(df.columns))
print("Rows:", len(df), "| Time:", df.index.min(), "→", df.index.max())

OUT = "b1_MULTI_14d_1min.csv"
df.to_csv(OUT)
print(f"✅ Saved {OUT}")

ds.store.close()
