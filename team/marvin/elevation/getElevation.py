
import glob
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import rowcol
from rasterio.warp import transform


# ============================================================
# 1. File paths and configuration
# ============================================================

DATA_DIR = "data/marvin"

TILE_DIR = os.path.join(DATA_DIR, "houston_dem_reprojected")
INPUT_CSV = os.path.join(DATA_DIR, "input.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "out", "elevation_slope.csv")

PAD = 3  # Pixels of padding around each point


# ============================================================
# 2. Create required directories if they do not exist
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

if not os.path.isdir(TILE_DIR):
    raise FileNotFoundError(f"DEM tile directory not found: {TILE_DIR}")


# ============================================================
# 3. Build tile index (reads only metadata)
# ============================================================

paths = (
    glob.glob(os.path.join(TILE_DIR, "*.tif"))
    + glob.glob(os.path.join(TILE_DIR, "*.tiff"))
)

if not paths:
    raise RuntimeError(f"No TIFF files found in {TILE_DIR}")

tile_index = []

DEM_CRS = None
RES_X = None
RES_Y = None
NODATA = None

for path in paths:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"DEM tile has no CRS: {path}")

        if DEM_CRS is None:
            DEM_CRS = src.crs
            RES_X, RES_Y = src.res
            NODATA = (
                src.nodata
                if src.nodata is not None
                else -9999.0
            )
        else:
            if src.crs != DEM_CRS:
                raise ValueError(
                    f"DEM tiles have different CRSs: {path}"
                )

            if not np.allclose(src.res, (RES_X, RES_Y)):
                raise ValueError(
                    f"DEM tiles have different resolutions: {path}"
                )

        tile_index.append((path, src.bounds))

print(f"Found {len(tile_index)} DEM tiles.")
print(f"DEM CRS: {DEM_CRS}")
print(f"DEM resolution: {RES_X}, {RES_Y}")


# ============================================================
# 4. Get elevation and slope for one location
# ============================================================

def get_elevation_and_slope(lat, lon):
    # Check for missing coordinates
    if pd.isna(lat) or pd.isna(lon):
        return None, None

    # Check coordinate ranges
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None

    # Transform longitude/latitude into the DEM CRS
    xs, ys = transform(
        "EPSG:4326",
        DEM_CRS,
        [lon],
        [lat]
    )

    x, y = xs[0], ys[0]

    # Define a window around the point
    left = x - PAD * RES_X
    right = x + PAD * RES_X
    bottom = y - PAD * RES_Y
    top = y + PAD * RES_Y

    # Find tiles that intersect the window
    hits = [
        path
        for path, bounds in tile_index
        if not (
            bounds.right < left
            or bounds.left > right
            or bounds.top < bottom
            or bounds.bottom > top
        )
    ]

    if not hits:
        return None, None

    # Merge the relevant tiles
    srcs = [rasterio.open(path) for path in hits]

    try:
        mosaic, tf = merge(
            srcs,
            bounds=(left, bottom, right, top),
            nodata=NODATA
        )
    finally:
        for src in srcs:
            src.close()

    arr = mosaic[0].astype(float)

    # Mask NoData values
    if NODATA is not None:
        arr[arr == NODATA] = np.nan

    # Convert point coordinates to raster row and column
    r, c = rowcol(tf, x, y)

    # Ensure a complete 3x3 neighborhood is available
    if (
        r < 1
        or c < 1
        or r + 2 > arr.shape[0]
        or c + 2 > arr.shape[1]
    ):
        return None, None

    z = arr[r - 1:r + 2, c - 1:c + 2]

    if z.shape != (3, 3) or not np.isfinite(z).all():
        return None, None

    # Horn's method for calculating slope
    dz_dx = (
        (z[0, 2] + 2 * z[1, 2] + z[2, 2])
        - (z[0, 0] + 2 * z[1, 0] + z[2, 0])
    ) / (8 * RES_X)

    dz_dy = (
        (z[2, 0] + 2 * z[2, 1] + z[2, 2])
        - (z[0, 0] + 2 * z[0, 1] + z[0, 2])
    ) / (8 * RES_Y)

    slope = np.degrees(
        np.arctan(np.hypot(dz_dx, dz_dy))
    )

    # Return center-pixel elevation and slope in degrees
    return float(z[1, 1]), float(slope)


# ============================================================
# 5. Read input CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_columns = {"lat", "lon"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Input CSV is missing required columns: {missing_columns}"
    )

# Keep only the latitude and longitude columns
points = df[["lat", "lon"]].copy()

# Convert coordinates to numeric values
points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
points["lon"] = pd.to_numeric(points["lon"], errors="coerce")


# ============================================================
# 6. Compute elevation and slope for each point
# ============================================================

results = []

total_points = len(points)

for i, row in enumerate(points.itertuples(index=False), start=1):
    elev, slope = get_elevation_and_slope(
        row.lat,
        row.lon
    )

    results.append((elev, slope))

    if i % 100 == 0 or i == total_points:
        print(f"Processed {i}/{total_points} points")

points["elevation_m"] = [result[0] for result in results]
points["slope_degrees"] = [result[1] for result in results]


# ============================================================
# 7. Create output directory and save results
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

points.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(points)} rows to {OUTPUT_CSV}")
print(
    f"Rows with valid elevation: "
    f"{points['elevation_m'].notna().sum()}"
)
print(
    f"Rows with valid slope: "
    f"{points['slope_degrees'].notna().sum()}"
)

print("\nFirst five results:")
print(points.head())
