
import geopandas as gpd
import pandas as pd
import glob
import os
import zipfile
import urllib.request

# File paths
input_csv = "data/input.csv"
output_csv = "data/out/stream_distances.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# NHD dataset paths
dataset_name = "Shape"
dataset_folder = os.path.join(SCRIPT_DIR, dataset_name)
zip_path = os.path.join(SCRIPT_DIR, f"{dataset_name}.zip")
folder =  os.path.join(SCRIPT_DIR, "Shape")

# Download URL
download_url = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/"
    "Hydrography/NHD/State/Shape/"
    "NHD_H_Texas_State_Shape.zip"
)

# 1. Download and extract dataset if it does not exist
if not os.path.isdir(dataset_folder):
    print("NHD Texas dataset not found. Downloading...")

    urllib.request.urlretrieve(download_url, zip_path)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(SCRIPT_DIR)

    print("Extraction complete.")

    # Remove ZIP file after successful extraction
    os.remove(zip_path)

else:
    print("NHD Texas dataset already exists. Skipping download.")

# 2. Load input CSV
df = pd.read_csv(input_csv)

# 3. Load all NHD flowline shapefiles
flowline_files = glob.glob(
    os.path.join(folder, "NHDFlowline_*.shp")
)

if not flowline_files:
    raise FileNotFoundError(
        f"No NHDFlowline shapefiles found in: {folder}"
    )

flowlines = gpd.GeoDataFrame(
    pd.concat(
        [gpd.read_file(f) for f in flowline_files],
        ignore_index=True
    ),
    crs=gpd.read_file(flowline_files[0]).crs
)

# 4. Keep only valid stream geometries
flowlines = flowlines[
    flowlines.geometry.notna()
    & ~flowlines.geometry.is_empty
].copy()

# 5. Project flowlines to a CRS measured in meters
target_crs = "EPSG:3081"
flowlines_proj = flowlines.to_crs(target_crs)

# Keep only geometry for the spatial join
flowlines_proj = flowlines_proj[["geometry"]]

# 6. Create points from CSV latitude and longitude
valid_coords = df["lat"].notna() & df["lon"].notna()

points_gdf = gpd.GeoDataFrame(
    df.loc[valid_coords, ["lat", "lon"]].copy(),
    geometry=gpd.points_from_xy(
        df.loc[valid_coords, "lon"],
        df.loc[valid_coords, "lat"]
    ),
    crs="EPSG:4326"
)

# 7. Project points to the same CRS
points_proj = points_gdf.to_crs(target_crs)

# 8. Find the nearest stream for every point
nearest_stream = gpd.sjoin_nearest(
    points_proj,
    flowlines_proj,
    how="left",
    distance_col="distance_to_bayou_m"
)

nearest_stream = nearest_stream[
    ~nearest_stream.index.duplicated(keep="first")
]

# 9. Create output with exactly three columns
results = pd.DataFrame({
    "lat": nearest_stream["lat"].values,
    "lon": nearest_stream["lon"].values,
    "distance_to_bayou_m": nearest_stream[
        "distance_to_bayou_m"
    ].values
})

# 10. Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# 11. Save results
results.to_csv(output_csv, index=False)

print(f"Saved {len(results)} records to {output_csv}")
print(results.head())
