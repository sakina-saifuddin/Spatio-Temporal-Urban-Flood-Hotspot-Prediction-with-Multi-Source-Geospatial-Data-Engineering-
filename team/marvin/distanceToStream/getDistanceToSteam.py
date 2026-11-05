
import geopandas as gpd
import pandas as pd
import glob
import os
import zipfile
import urllib.request

# ============================================================
# 1. File paths and configuration
# ============================================================

DATA_DIR = "data/marvin"

input_csv = os.path.join(DATA_DIR, "input.csv")
output_csv = os.path.join(DATA_DIR, "out", "stream_distances.csv")

dataset_name = "Shape"
dataset_folder = os.path.join(DATA_DIR, dataset_name)
zip_path = os.path.join(DATA_DIR, f"{dataset_name}.zip")
folder = dataset_folder

download_url = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/"
    "Hydrography/NHD/State/Shape/"
    "NHD_H_Texas_State_Shape.zip"
)

target_crs = "EPSG:3081"

# ============================================================
# 2. Create required directories if they do not exist
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {os.path.dirname(output_csv)}")

# ============================================================
# 3. Check input CSV
# ============================================================

if not os.path.isfile(input_csv):
    raise FileNotFoundError(
        f"Input CSV not found: {input_csv}"
    )

# ============================================================
# 4. Download and extract NHD dataset if needed
# ============================================================

flowline_files = glob.glob(
    os.path.join(folder, "NHDFlowline_*.shp")
)

if not flowline_files:
    print("NHD Texas dataset not found or incomplete. Downloading...")

    try:
        urllib.request.urlretrieve(download_url, zip_path)
        print("Download complete. Extracting...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

        print("Extraction complete.")

    finally:
        # Remove the ZIP file if it exists
        if os.path.isfile(zip_path):
            os.remove(zip_path)

    # Find flowline shapefiles after extraction
    flowline_files = glob.glob(
        os.path.join(folder, "NHDFlowline_*.shp")
    )

if not flowline_files:
    raise FileNotFoundError(
        f"No NHDFlowline shapefiles found in: {folder}. "
        "Check the downloaded archive and extraction directory."
    )

print(f"Found {len(flowline_files)} flowline shapefile(s).")

# ============================================================
# 5. Load input CSV
# ============================================================

df = pd.read_csv(input_csv)

required_columns = {"lat", "lon"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Input CSV is missing required columns: {missing_columns}"
    )

# ============================================================
# 6. Load all NHD flowline shapefiles
# ============================================================

flowline_gdfs = [
    gpd.read_file(file)
    for file in flowline_files
]

# Ensure all flowlines use the same CRS
source_crs = flowline_gdfs[0].crs

if source_crs is None:
    raise ValueError(
        f"Missing CRS in shapefile: {flowline_files[0]}"
    )

for i, gdf in enumerate(flowline_gdfs):
    if gdf.crs is None:
        raise ValueError(
            f"Missing CRS in shapefile: {flowline_files[i]}"
        )

    if gdf.crs != source_crs:
        flowline_gdfs[i] = gdf.to_crs(source_crs)

flowlines = gpd.GeoDataFrame(
    pd.concat(flowline_gdfs, ignore_index=True),
    geometry="geometry",
    crs=source_crs
)

# ============================================================
# 7. Keep only valid stream geometries
# ============================================================

flowlines = flowlines[
    flowlines.geometry.notna()
    & ~flowlines.geometry.is_empty
].copy()

if flowlines.empty:
    raise ValueError("No valid flowline geometries were found.")

# Project flowlines to a CRS measured in meters
flowlines_proj = flowlines.to_crs(target_crs)

# Keep only geometry for the nearest-stream calculation
flowlines_proj = flowlines_proj[["geometry"]]

print(f"Loaded {len(flowlines_proj)} valid flowline geometries.")

# ============================================================
# 8. Create points from CSV latitude and longitude
# ============================================================

# Convert coordinates to numeric values
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

# Exclude missing, non-numeric, and out-of-range coordinates
valid_coords = (
    df["lat"].notna()
    & df["lon"].notna()
    & df["lat"].between(-90, 90)
    & df["lon"].between(-180, 180)
)

valid_df = df.loc[valid_coords].copy()

if valid_df.empty:
    raise ValueError(
        "No valid latitude/longitude coordinates found in the input CSV."
    )

points_gdf = gpd.GeoDataFrame(
    valid_df[["lat", "lon"]].copy(),
    geometry=gpd.points_from_xy(
        valid_df["lon"],
        valid_df["lat"]
    ),
    crs="EPSG:4326"
)

# ============================================================
# 9. Project points to the same CRS as flowlines
# ============================================================

points_proj = points_gdf.to_crs(target_crs)

# ============================================================
# 10. Find the nearest stream for every valid point
# ============================================================

nearest_stream = gpd.sjoin_nearest(
    points_proj,
    flowlines_proj,
    how="left",
    distance_col="distance_to_bayou_m"
)

# A point can match multiple flowlines if they are tied
# at the same minimum distance. Keep one result per input row.
nearest_stream = nearest_stream[
    ~nearest_stream.index.duplicated(keep="first")
]

# ============================================================
# 11. Create output with exactly three columns
# ============================================================

results = pd.DataFrame({
    "lat": nearest_stream["lat"].to_numpy(),
    "lon": nearest_stream["lon"].to_numpy(),
    "distance_to_bayou_m": nearest_stream[
        "distance_to_bayou_m"
    ].to_numpy()
})

# ============================================================
# 12. Save results
# ============================================================

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

results.to_csv(output_csv, index=False)

print(f"\nSaved {len(results)} records to {output_csv}")
print("\nFirst five results:")
print(results.head())

print(f"\nTotal input records: {len(df)}")
print(f"Records with valid coordinates: {len(valid_df)}")
print(f"Records saved: {len(results)}")
print(f"Records skipped due to invalid coordinates: {len(df) - len(valid_df)}")
