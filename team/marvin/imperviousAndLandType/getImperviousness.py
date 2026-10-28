import pandas as pd
import rasterio
from pyproj import Transformer
import os
import requests
import zipfile
from pathlib import Path


# Input and output files
input_csv = "data/input.csv"
output_csv = "data/out/landcover_imperviousness.csv"

# Script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# NLCD raster files
impervious_tif = SCRIPT_DIR / "Annual_NLCD_FctImp_2025_CU_C1V2.tif"
landcover_tif = SCRIPT_DIR / "Annual_NLCD_LndCov_2025_CU_C1V2.tif"

# MRLC download URLs
impervious_zip_url = (
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
    "Annual_NLCD_FctImp_2025_CU_C1V2.zip"
)

landcover_zip_url = (
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
    "Annual_NLCD_LndCov_2025_CU_C1V2.zip"
)


def download_and_extract(url, target_tif):
    """
    Download a ZIP file and extract the required TIFF file
    if it does not already exist.
    """
    target_tif = Path(target_tif)

    # Skip downloading if the TIFF already exists
    if target_tif.exists():
        print(f"File already exists: {target_tif.name}")
        return

    zip_path = SCRIPT_DIR / (target_tif.stem + ".zip")

    print(f"Downloading {target_tif.name}...")

    # Download the ZIP file
    response = requests.get(url, stream=True, timeout=(30, 300))
    response.raise_for_status()

    # Save the ZIP file
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {zip_path.name}")

    # Extract the required TIFF file
    print(f"Extracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Find the TIFF by its exact filename, including subfolders
            matching_files = [
                name for name in z.namelist()
                if Path(name).name == target_tif.name
            ]

            if not matching_files:
                raise FileNotFoundError(
                    f"{target_tif.name} was not found in {zip_path.name}"
                )

            # Extract the TIFF directly to SCRIPT_DIR
            with z.open(matching_files[0]) as source, open(
                target_tif, "wb"
            ) as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

    finally:
        # Remove the downloaded ZIP after extraction
        if zip_path.exists():
            zip_path.unlink()

    print(f"Extracted: {target_tif.name}")


# Download missing raster files
download_and_extract(impervious_zip_url, impervious_tif)
download_and_extract(landcover_zip_url, landcover_tif)


# NLCD land-cover class descriptions
land_cover_classes = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}


# Read input CSV
df = pd.read_csv(input_csv)

# Open both rasters
with rasterio.open(impervious_tif) as imp_src, \
     rasterio.open(landcover_tif) as lc_src:

    # Coordinate transformers (WGS84 -> raster CRS)
    imp_transformer = Transformer.from_crs(
        "EPSG:4326", imp_src.crs, always_xy=True
    )

    lc_transformer = Transformer.from_crs(
        "EPSG:4326", lc_src.crs, always_xy=True
    )

    results = []

    for _, row in df.iterrows():
        lat = row["lat"]
        lon = row["lon"]

        imperviousness = None
        land_cover = None

        if pd.notna(lat) and pd.notna(lon):
            # Extract impervious surface
            x, y = imp_transformer.transform(lon, lat)
            value = next(imp_src.sample([(x, y)]))[0]

            if imp_src.nodata is None or value != imp_src.nodata:
                imperviousness = float(value / 100)

            # Extract land-cover class
            x, y = lc_transformer.transform(lon, lat)
            value = next(lc_src.sample([(x, y)]))[0]

            if lc_src.nodata is None or value != lc_src.nodata:
                land_cover = land_cover_classes.get(
                    int(value), f"Unknown class ({int(value)})"
                )

        results.append({
            "lat": lat,
            "lon": lon,
            "imperviousness": imperviousness,
            "land_cover_type": land_cover
        })


# Create output directory if necessary
output_path = Path(output_csv)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save output CSV with exactly four columns
output_df = pd.DataFrame(
    results,
    columns=[
        "lat",
        "lon",
        "imperviousness",
        "land_cover_type"
    ]
)

output_df.to_csv(output_path, index=False)

print(f"Saved {len(output_df)} records to {output_path}")
print(output_df.head())