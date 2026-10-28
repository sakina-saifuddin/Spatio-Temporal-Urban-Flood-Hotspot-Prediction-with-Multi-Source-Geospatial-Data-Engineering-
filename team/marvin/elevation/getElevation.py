import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import rowcol
from rasterio.warp import transform
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILE_DIR =  os.path.join(SCRIPT_DIR, "houston_dem_reprojected")

# TILE_DIR = "houston_dem_reprojected"   # folder with your DEM tiles (all in the same CRS)
INPUT_CSV = "data/input.csv"           # your input file
OUTPUT_CSV = "data/out/elevation_slope.csv"
PAD = 3                                # pixels of padding around each point

# ---- Build tile index once (reads only metadata) ----
paths = glob.glob(f"{TILE_DIR}/*.tif") + glob.glob(f"{TILE_DIR}/*.tiff")
if not paths:
    raise RuntimeError(f"No TIFF files found in {TILE_DIR}")

tile_index = []
for p in paths:
    with rasterio.open(p) as src:
        tile_index.append((p, src.bounds))
        DEM_CRS = src.crs
        RES_X, RES_Y = src.res
        NODATA = src.nodata if src.nodata is not None else -9999.0


def get_elevation_and_slope(lat, lon):
    xs, ys = transform("EPSG:4326", DEM_CRS, [lon], [lat])
    x, y = xs[0], ys[0]

    left, right = x - PAD * RES_X, x + PAD * RES_X
    bottom, top = y - PAD * RES_Y, y + PAD * RES_Y

    hits = [
        p for p, b in tile_index
        if not (b.right < left or b.left > right or b.top < bottom or b.bottom > top)
    ]
    if not hits:
        return None, None

    srcs = [rasterio.open(p) for p in hits]
    try:
        mosaic, tf = merge(srcs, bounds=(left, bottom, right, top), nodata=NODATA)
    finally:
        for s in srcs:
            s.close()

    arr = mosaic[0].astype(float)
    arr[arr == NODATA] = np.nan

    r, c = rowcol(tf, x, y)
    if r < 1 or c < 1 or r + 2 > arr.shape[0] or c + 2 > arr.shape[1]:
        return None, None

    z = arr[r - 1:r + 2, c - 1:c + 2]
    if np.isnan(z).any():
        return None, None

    # Horn's method
    dz_dx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * RES_X)
    dz_dy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * RES_Y)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    return float(z[1, 1]), float(slope)


# ---- Read CSV and compute for each unique location ----
df = pd.read_csv(INPUT_CSV)

points = df[["lat", "lon"]]

results = []
for i, row in points.iterrows():
    elev, slope = get_elevation_and_slope(row["lat"], row["lon"])
    results.append((elev, slope))
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(points)} points")

points["elevation_m"] = [r[0] for r in results]
points["slope_degrees"] = [r[1] for r in results]

points.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(points)} rows to {OUTPUT_CSV}")
print(points.head())