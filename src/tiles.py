"""Helpers for reprojecting elevation rasters and extracting terrain tiles."""

from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds


def reproject_raster(
    source_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:32615",
    resolution: float = 1.0,
) -> Path:
    """Reproject one DEM to a meter-based CRS and pixel size."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_path) as source:
        nodata = source.nodata if source.nodata is not None else -9999.0
        transform, width, height = calculate_default_transform(
            source.crs,
            target_crs,
            source.width,
            source.height,
            *source.bounds,
            resolution=resolution,
        )
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=nodata,
            dtype="float32",
            count=1,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path, "w", **profile) as destination:
            reproject(
                source=rasterio.band(source, 1),
                destination=rasterio.band(destination, 1),
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )
    return output_path


def make_tiles(
    df: pd.DataFrame,
    raster_paths: Iterable[str | Path],
    out_path: str | Path,
    tile_size: int = 32,
    window_size_m: float = 320.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract relative elevation tiles and save them with their row indices."""
    raster_paths = [Path(path) for path in raster_paths]
    if not raster_paths:
        raise ValueError("At least one elevation raster is required.")
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("DataFrame must contain lat and lon columns.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_indices = (
        df["row_index"].to_numpy()
        if "row_index" in df.columns
        else df.index.to_numpy()
    )
    tiles = np.zeros((len(df), tile_size, tile_size), dtype=np.float16)
    missing_count = 0

    with ExitStack() as stack:
        rasters = [stack.enter_context(rasterio.open(path)) for path in raster_paths]
        raster_crs = rasters[0].crs
        if raster_crs is None or any(dataset.crs != raster_crs for dataset in rasters):
            raise ValueError("All rasters must have the same defined CRS.")
        to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

        for tile_index, point in enumerate(df[["lat", "lon"]].itertuples(index=False)):
            x, y = to_raster.transform(float(point.lon), float(point.lat))
            dataset = next(
                (
                    candidate
                    for candidate in rasters
                    if candidate.bounds.left <= x <= candidate.bounds.right
                    and candidate.bounds.bottom <= y <= candidate.bounds.top
                ),
                None,
            )
            if dataset is None:
                missing_count += 1
                continue

            center_row, center_col = rowcol(dataset.transform, x, y)
            center = dataset.read(
                1,
                window=Window(center_col, center_row, 1, 1),
                masked=True,
            )
            if np.ma.is_masked(center[0, 0]):
                missing_count += 1
                continue

            half_width = window_size_m / 2
            window = from_bounds(
                x - half_width,
                y - half_width,
                x + half_width,
                y + half_width,
                transform=dataset.transform,
            )
            elevation = dataset.read(
                1,
                window=window,
                out_shape=(tile_size, tile_size),
                resampling=Resampling.average,
                masked=True,
                boundless=True,
            )
            if np.ma.getmaskarray(elevation).any():
                missing_count += 1
            relative = (elevation.astype(np.float32) - float(center[0, 0])) / 10.0
            tiles[tile_index] = np.ma.filled(relative, 0).astype(np.float16)

    np.save(out_path, tiles)
    np.save(out_path.with_name(f"{out_path.stem}_row_index.npy"), row_indices)
    return tiles, row_indices, missing_count
