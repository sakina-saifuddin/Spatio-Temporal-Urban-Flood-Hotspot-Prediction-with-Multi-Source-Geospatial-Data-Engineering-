
import os
import zipfile
from pathlib import Path

import pandas as pd
import rasterio
import requests
from pyproj import Transformer


# ============================================================
# 1. Paths and configuration
# ============================================================

DATA_DIR = Path("data/marvin")

INPUT_CSV = DATA_DIR / "input.csv"
OUTPUT_CSV = DATA_DIR / "out" / "landcover_imperviousness.csv"

# NLCD raster files
impervious_tif = DATA_DIR / "Annual_NLCD_FctImp_2025_CU_C1V2.tif"
landcover_tif = DATA_DIR / "Annual_NLCD_LndCov_2025_CU_C1V2.tif"

# MRLC download URLs
impervious_zip_url = (
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
    "Annual_NLCD_FctImp_2025_CU_C1V2.zip"
)

landcover_zip_url = (
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
    "Annual_NLCD_LndCov_2025_CU_C1V2.zip"
)


# ============================================================
# 2. Create required directories
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")


# ============================================================
# 3. Download and extract missing raster files
# ============================================================

def download_and_extract(url, target_tif):
    """
    Download a ZIP file and extract the required TIFF file
    if it does not already exist.
    """
    target_tif = Path(target_tif)

    if target_tif.exists():
        print(f"File already exists: {target_tif}")
        return

    zip_path = DATA_DIR / (target_tif.stem + ".zip")
    temp_tif = target_tif.with_suffix(".tmp")

    print(f"Downloading {target_tif.name}...")

    try:
        # Download the ZIP file
        with requests.get(
            url,
            stream=True,
            timeout=(30, 300)
        ) as response:
            response.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded: {zip_path.name}")
        print(f"Extracting {zip_path.name}...")

        # Locate the required TIFF in the archive
        with zipfile.ZipFile(zip_path, "r") as z:
            matching_files = [
                name
                for name in z.namelist()
                if Path(name).name == target_tif.name
            ]

            if not matching_files:
                raise FileNotFoundError(
                    f"{target_tif.name} was not found "
                    f"in {zip_path.name}"
                )

            # Extract to a temporary file first
            with z.open(matching_files[0]) as source, open(
                temp_tif, "wb"
            ) as destination:
                while True:
                    chunk = source.read(1024 * 1024)

                    if not chunk:
                        break

                    destination.write(chunk)

        # Rename only after successful extraction
        temp_tif.replace(target_tif)

        print(f"Extracted: {target_tif}")

    finally:
        # Clean up temporary files
        if zip_path.exists():
            zip_path.unlink()

        if temp_tif.exists():
            temp_tif.unlink()


# Download missing raster files
download_and_extract(
    impervious_zip_url,
    impervious_tif
)

download_and_extract(
    landcover_zip_url,
    landcover_tif
)


# ============================================================
# 4. NLCD land-cover class descriptions
# ============================================================

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
    95: "Emergent Herbaceous Wetlands",
}


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

# Ensure coordinates are numeric
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")


# ============================================================
# 6. Extract imperviousness and land-cover values
# ============================================================

results = []

with rasterio.open(impervious_tif) as imp_src, \
     rasterio.open(landcover_tif) as lc_src:

    if imp_src.crs is None or lc_src.crs is None:
        raise ValueError("One or both NLCD rasters have no CRS.")

    # Coordinate transformers: WGS84 -> raster CRS
    imp_transformer = Transformer.from_crs(
        "EPSG:4326",
        imp_src.crs,
        always_xy=True
    )

    lc_transformer = Transformer.from_crs(
        "EPSG:4326",
        lc_src.crs,
        always_xy=True
    )

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):
        lat = row.lat
        lon = row.lon

        imperviousness = None
        land_cover = None

        # Skip missing or out-of-range coordinates
        valid_coords = (
            pd.notna(lat)
            and pd.notna(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        )

        if valid_coords:
            # --------------------------------------------
            # Extract impervious surface
            # --------------------------------------------

            x, y = imp_transformer.transform(lon, lat)

            if (
                imp_src.bounds.left <= x <= imp_src.bounds.right
                and imp_src.bounds.bottom <= y <= imp_src.bounds.top
            ):
                value = next(
                    imp_src.sample([(x, y)], masked=True)
                )[0]

                if not pd.isna(value):
                    imperviousness = float(value) / 100.0

            # --------------------------------------------
            # Extract land-cover class
            # --------------------------------------------

            x, y = lc_transformer.transform(lon, lat)

            if (
                lc_src.bounds.left <= x <= lc_src.bounds.right
                and lc_src.bounds.bottom <= y <= lc_src.bounds.top
            ):
                value = next(
                    lc_src.sample([(x, y)], masked=True)
                )[0]

                if not pd.isna(value):
                    class_value = int(value)

                    land_cover = land_cover_classes.get(
                        class_value,
                        f"Unknown class ({class_value})"
                    )

        results.append({
            "lat": lat,
            "lon": lon,
            "imperviousness": imperviousness,
            "land_cover_type": land_cover,
        })

        if i % 100 == 0 or i == len(df):
            print(f"Processed {i}/{len(df)} points")


# ============================================================
# 7. Save output CSV with exactly four columns
# ============================================================

output_df = pd.DataFrame(
    results,
    columns=[
        "lat",
        "lon",
        "imperviousness",
        "land_cover_type",
    ]
)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(output_df)} records to {OUTPUT_CSV}")
print("\nFirst five results:")
print(output_df.head())

print(
    "\nRecords with imperviousness values: "
    f"{output_df['imperviousness'].notna().sum()}"
)

print(
    "Records with land-cover values: "
    f"{output_df['land_cover_type'].notna().sum()}"
)
