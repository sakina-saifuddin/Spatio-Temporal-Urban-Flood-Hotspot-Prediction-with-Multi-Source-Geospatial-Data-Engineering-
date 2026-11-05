
import os
import glob
import rasterio

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)


# ============================================================
# 1. Paths and configuration
# ============================================================

DATA_DIR = "data/marvin"

tile_dir = os.path.join(DATA_DIR, "houston_dem_tiles")
reprojected_dir = os.path.join(DATA_DIR, "houston_dem_reprojected")

# Houston: projected coordinate system with meter units
TARGET_CRS = "EPSG:32615"
TARGET_RESOLUTION = 1.0  # 1 meter per pixel

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(reprojected_dir, exist_ok=True)


# ============================================================
# 2. Find input DEM tiles
# ============================================================

if not os.path.isdir(tile_dir):
    raise FileNotFoundError(
        f"DEM tile directory not found: {tile_dir}"
    )

tif_files = (
    glob.glob(os.path.join(tile_dir, "*.tif"))
    + glob.glob(os.path.join(tile_dir, "*.tiff"))
)

if not tif_files:
    raise RuntimeError(
        f"No TIFF files found in {tile_dir}"
    )

print(f"Found {len(tif_files)} input DEM tiles.")


# ============================================================
# 3. Reproject every tile
# ============================================================

reprojected_files = []

for i, path in enumerate(tif_files):
    output_path = os.path.join(
        reprojected_dir,
        f"tile_{i:04d}.tif"
    )

    print(f"Reprojecting {path}")

    with rasterio.open(path) as src:
        if src.crs is None:
            print(f"Skipping file without CRS: {path}")
            continue

        # Preserve source NoData where available
        nodata = (
            src.nodata
            if src.nodata is not None
            else -9999.0
        )

        # Calculate target dimensions and transform
        dst_transform, dst_width, dst_height = (
            calculate_default_transform(
                src.crs,
                TARGET_CRS,
                src.width,
                src.height,
                *src.bounds,
                resolution=TARGET_RESOLUTION,
            )
        )

        # Prepare output raster metadata
        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            crs=TARGET_CRS,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            nodata=nodata,
            dtype="float32",
            count=1,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        # Write reprojected raster
        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )

    reprojected_files.append(output_path)
    print(f"Saved: {output_path}")


# ============================================================
# 4. Report results
# ============================================================

if not reprojected_files:
    raise RuntimeError(
        "No usable TIFF files were reprojected."
    )

print("\nReprojection complete.")
print(f"Reprojected tiles: {len(reprojected_files)}")
print(f"Output directory: {reprojected_dir}")

print("\nOutput files:")
for path in reprojected_files:
    print(path)