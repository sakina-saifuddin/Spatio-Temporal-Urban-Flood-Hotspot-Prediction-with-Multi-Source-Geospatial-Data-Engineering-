
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/marvin")
OUT = DATA_DIR / "out"
DATA = DATA_DIR

train  = pd.read_csv(DATA / "input.csv")
elev   = pd.read_csv(OUT / "elevation_slope.csv")
land   = pd.read_csv(OUT / "landcover_imperviousness.csv")
paved  = pd.read_csv(OUT / "paved_pct.csv")
stream = pd.read_csv(OUT / "stream_distances.csv")
rain   = pd.read_csv(OUT / "rain.csv")

others = {"elevation_slope": elev, "landcover": land, "paved_pct": paved,
          "stream_distances": stream, "rain": rain}

# 1. Row counts must match
sizes = {"train": len(train), **{k: len(v) for k, v in others.items()}}
assert len(set(sizes.values())) == 1, f"Row counts differ: {sizes}"

# 2. lat/lon must line up row by row
for name, df in others.items():
    ok = (np.isclose(train["lat"], df["lat"]) &
          np.isclose(train["lon"], df["lon"])).all()
    assert ok, f"lat/lon mismatch between training set and {name}"

# 3. storm_date must line up too (rain.csv has it)
assert (train["storm_date"].astype(str) == rain["storm_date"].astype(str)).all(), \
    "storm_date mismatch between training set and rain.csv"

# 4. Fill the training columns line by line (by position)
train["elevation_m"]          = elev["elevation_m"].to_numpy()
train["slope_degrees"]        = elev["slope_degrees"].to_numpy()
train["distance_to_bayou_m"]  = stream["distance_to_bayou_m"].to_numpy()
train["paved_pct"]            = paved["paved_pct"].to_numpy()
train["land_cover_type"]      = land["land_cover_type"].to_numpy()
train["rainfall_inches"]      = rain["rainfall_inches"].to_numpy()

# Optional: keep imperviousness as an extra column
train["imperviousness"]       = land["imperviousness"].to_numpy()

train.to_csv(DATA / "out.csv", index=False)
print(f"Saved {len(train)} rows to {DATA / 'out.csv'}")
