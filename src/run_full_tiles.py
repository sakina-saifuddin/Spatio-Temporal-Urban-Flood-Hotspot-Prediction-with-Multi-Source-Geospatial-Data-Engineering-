"""Download elevation tiles and build resumable training terrain tiles."""

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

from src.tiles import reproject_raster


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CSV = ROOT / "data" / "processed" / "training_clean.csv"
PILOT_SOURCE = ROOT / "data" / "marvin" / "pilot" / "source"
PILOT_REPROJECTED = ROOT / "data" / "marvin" / "pilot" / "reprojected"
FULL_SOURCE = ROOT / "data" / "marvin" / "full" / "source"
FULL_REPROJECTED = ROOT / "data" / "marvin" / "full" / "reprojected"
TILE_DIR = ROOT / "data" / "processed" / "tiles"
TILE_PATH = TILE_DIR / "train_tiles.npy"
INDEX_PATH = TILE_DIR / "train_tiles_row_index.npy"
HAS_TILE_PATH = TILE_DIR / "train_tiles_has_tile.npy"
MISSING_PATH = TILE_DIR / "train_tiles_has_missing_pixels.npy"
STATE_PATH = TILE_DIR / "train_tiles_progress.json"
FIGURE_PATH = ROOT / "reports" / "figures" / "example_tiles_full.png"
LOG_PATH = ROOT / "logs" / "tiles_run.log"
TARGET_CRS = "EPSG:32615"
TARGET_RESOLUTION = 1.0
WINDOW_METERS = 320.0
TILE_SIZE = 32
SEED = 42


class JobLog:
    """Write progress to both the terminal and a persistent log."""

    def __init__(self) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.file = LOG_PATH.open("a", encoding="utf-8", buffering=1)
        self.started = time.perf_counter()
        self.write(f"\n--- Run started {datetime.now(timezone.utc).isoformat()} ---")

    def write(self, message: str) -> None:
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self.file.write(line + "\n")

    def close(self) -> None:
        self.file.close()


def catalog_tiles(points: pd.DataFrame) -> list[dict[str, object]]:
    """Select B24 DEM files whose footprints cover training-point windows."""
    bbox = ",".join(
        str(value)
        for value in (
            points["lon"].min(),
            points["lat"].min(),
            points["lon"].max(),
            points["lat"].max(),
        )
    )
    params = {
        "datasets": "Digital Elevation Model (DEM) 1 meter",
        "bbox": bbox,
        "prodFormats": "GeoTIFF",
        "max": 1000,
        "outputFormat": "JSON",
    }
    url = "https://tnmaccess.nationalmap.gov/api/v1/products?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Flood terrain pipeline/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)

    lat_pad = 160.0 / 111320.0
    selected = []
    for product in payload.get("items", []):
        if "TX_Houston_B24" not in product.get("title", ""):
            continue
        if not product.get("downloadURL", "").lower().split("?")[0].endswith((".tif", ".tiff")):
            continue
        bounds = product.get("boundingBox", {})
        lon_pad = 160.0 / (111320.0 * np.cos(np.radians(points["lat"].to_numpy())))
        intersects = (
            (bounds["maxX"] >= points["lon"].to_numpy() - lon_pad)
            & (bounds["minX"] <= points["lon"].to_numpy() + lon_pad)
            & (bounds["maxY"] >= points["lat"].to_numpy() - lat_pad)
            & (bounds["minY"] <= points["lat"].to_numpy() + lat_pad)
        )
        if intersects.any():
            selected.append(product)
    selected.sort(key=lambda product: product["title"])
    return selected


def find_valid_file(
    filename: str,
    candidate_dirs: tuple[Path, ...],
    expected_size: int | None = None,
) -> Path | None:
    """Find a cached file and reject incomplete or unexpected files."""
    for directory in candidate_dirs:
        path = directory / filename
        if not path.exists():
            continue
        if expected_size is not None and path.stat().st_size != expected_size:
            raise IOError(
                f"Existing file has unexpected size; preserving it: {path} "
                f"({path.stat().st_size} != {expected_size})"
            )
        return path
    return None


def download_file(
    url: str,
    path: Path,
    expected_size: int,
) -> tuple[Path, float]:
    """Download a GeoTIFF, resuming an existing partial file when supported."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != expected_size:
            raise IOError(f"Existing download has unexpected size; preserving it: {path}")
        return path, 0.0

    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise IOError(f"Partial download is larger than expected; preserving it: {partial}")

    headers = {"User-Agent": "AI-Flood terrain pipeline/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        if offset and getattr(response, "status", None) != 206:
            raise IOError(f"Server did not honor resume range for {partial}; file was preserved.")
        mode = "ab" if partial.exists() else "xb"
        with partial.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    elapsed = time.perf_counter() - started
    if partial.stat().st_size != expected_size:
        raise IOError(
            f"Download size mismatch for {partial}: "
            f"{partial.stat().st_size} != {expected_size}; partial file was preserved."
        )
    os.replace(partial, path)
    return path, elapsed


def ensure_reprojected(source: Path, destination: Path) -> tuple[Path, float, bool]:
    """Reuse a valid projected raster or create it atomically."""
    for candidate in (PILOT_REPROJECTED / source.name, destination):
        if not candidate.exists():
            continue
        with rasterio.open(candidate) as dataset:
            if dataset.crs != TARGET_CRS or not np.allclose(dataset.res, (1.0, 1.0)):
                raise ValueError(f"Existing projected raster has unexpected metadata: {candidate}")
        return candidate, 0.0, True

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(
        f"{destination.stem}.{datetime.now().strftime('%Y%m%d%H%M%S%f')}.tmp.tif"
    )
    started = time.perf_counter()
    reproject_raster(source, temp_path, TARGET_CRS, TARGET_RESOLUTION)
    elapsed = time.perf_counter() - started
    with rasterio.open(temp_path) as dataset:
        if dataset.crs != TARGET_CRS or not np.allclose(dataset.res, (1.0, 1.0)):
            raise ValueError(f"Reprojection output has unexpected metadata: {temp_path}")
    os.replace(temp_path, destination)
    return destination, elapsed, False


def save_state(completed_rows: int) -> None:
    """Atomically save terrain-extraction progress."""
    temp_path = STATE_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps({"completed_rows": completed_rows}),
        encoding="utf-8",
    )
    os.replace(temp_path, STATE_PATH)


def read_valid_center(
    datasets: list[rasterio.io.DatasetReader],
    x: float,
    y: float,
) -> tuple[rasterio.io.DatasetReader | None, np.ma.MaskedArray | None, int, int]:
    """Find a covering raster with a valid elevation at the point."""
    for dataset in datasets:
        if not (
            dataset.bounds.left <= x < dataset.bounds.right
            and dataset.bounds.bottom < y <= dataset.bounds.top
        ):
            continue
        row, col = rasterio.transform.rowcol(dataset.transform, x, y)
        center = dataset.read(
            1,
            window=Window(col, row, 1, 1),
            masked=True,
        )
        if not np.ma.is_masked(center[0, 0]) and np.isfinite(center[0, 0]):
            return dataset, center, row, col
    return None, None, -1, -1


def build_training_tiles(
    points: pd.DataFrame,
    rasters: list[Path],
    log: JobLog,
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-aligned terrain tiles, resuming from the last checkpoint."""
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    expected_shape = (len(points), TILE_SIZE, TILE_SIZE)
    if TILE_PATH.exists():
        tiles = np.lib.format.open_memmap(TILE_PATH, mode="r+")
        if tiles.shape != expected_shape or tiles.dtype != np.float16:
            raise ValueError(f"Existing tile output has unexpected shape or dtype: {TILE_PATH}")
    else:
        tiles = np.lib.format.open_memmap(
            TILE_PATH,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )

    if INDEX_PATH.exists():
        row_indices = np.load(INDEX_PATH)
        if not np.array_equal(row_indices, np.arange(len(points), dtype=np.int64)):
            raise ValueError(f"Existing row index does not match the cleaned training order: {INDEX_PATH}")
    else:
        row_indices = np.arange(len(points), dtype=np.int64)
        np.save(INDEX_PATH, row_indices)

    def open_mask(path: Path) -> np.memmap:
        if path.exists():
            mask = np.lib.format.open_memmap(path, mode="r+")
            if mask.shape != (len(points),) or mask.dtype != np.uint8:
                raise ValueError(f"Existing mask has unexpected shape or dtype: {path}")
            return mask
        return np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(points),),
        )

    has_tile = open_mask(HAS_TILE_PATH)
    has_missing = open_mask(MISSING_PATH)
    completed = 0
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        completed = int(state["completed_rows"])
        if not 0 <= completed <= len(points):
            raise ValueError(f"Invalid extraction checkpoint: {completed}")
    log.write(f"Terrain extraction resumes at row {completed:,}/{len(points):,}.")

    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    xs, ys = transformer.transform(points["lon"].to_numpy(), points["lat"].to_numpy())
    datasets = [rasterio.open(path) for path in rasters]
    try:
        for row_index in range(completed, len(points)):
            x, y = xs[row_index], ys[row_index]
            dataset, center, _, _ = read_valid_center(datasets, x, y)
            has_tile[row_index] = 0
            has_missing[row_index] = 1
            if dataset is not None and center is not None:
                has_tile[row_index] = 1
                half = WINDOW_METERS / 2
                window = from_bounds(
                    x - half,
                    y - half,
                    x + half,
                    y + half,
                    transform=dataset.transform,
                )
                elevation = dataset.read(
                    1,
                    window=window,
                    out_shape=(TILE_SIZE, TILE_SIZE),
                    resampling=Resampling.average,
                    masked=True,
                    boundless=True,
                )
                missing_mask = np.ma.getmaskarray(elevation) | ~np.isfinite(
                    np.ma.filled(elevation, np.nan)
                )
                has_missing[row_index] = int(missing_mask.any())
                relative = (
                    elevation.astype(np.float32) - float(center[0, 0])
                ) / 10.0
                tiles[row_index] = np.ma.filled(relative, 0).astype(np.float16)
            else:
                tiles[row_index] = 0

            next_row = row_index + 1
            if next_row % 500 == 0 or next_row == len(points):
                tiles.flush()
                has_tile.flush()
                has_missing.flush()
                save_state(next_row)
                log.write(f"Terrain rows {next_row:,}/{len(points):,}.")
    finally:
        for dataset in datasets:
            dataset.close()
    tiles.flush()
    has_tile.flush()
    has_missing.flush()
    return row_indices, np.asarray(has_tile)


def save_examples(points: pd.DataFrame, tiles: np.ndarray, has_tile: np.ndarray) -> None:
    """Save four flooded and four non-flooded terrain examples."""
    if FIGURE_PATH.exists():
        return
    rng = np.random.default_rng(SEED)
    labels = points["flooded"].to_numpy()
    selected: list[tuple[int, int]] = []
    for label in (1, 0):
        candidates = np.flatnonzero((labels == label) & (has_tile == 1))
        if len(candidates) < 4:
            raise ValueError(f"Need four valid terrain tiles for flooded={label}, found {len(candidates)}")
        selected.extend((int(index), label) for index in rng.choice(candidates, 4, replace=False))

    values = np.asarray(tiles[[index for index, _ in selected]], dtype=np.float32) * 10.0
    bound = max(float(np.nanpercentile(np.abs(values), 98)), 0.1)
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    image = None
    for axis, (index, label), array in zip(axes.flat, selected, values):
        image = axis.imshow(array, cmap="terrain", vmin=-bound, vmax=bound)
        axis.set_title("Flooded" if label == 1 else "Not flooded")
        axis.set_xticks([])
        axis.set_yticks([])
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Relative elevation (m)", shrink=0.8)
    figure.suptitle("Training terrain tiles (320 m windows, 32x32)")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    log = JobLog()
    try:
        points = pd.read_csv(
            TRAINING_CSV,
            usecols=["lat", "lon", "elevation_m", "flooded"],
        )
        if len(points) != 66514:
            raise ValueError(f"Expected 66,514 cleaned training rows, got {len(points)}")
        products = catalog_tiles(points)
        if len(products) != 76:
            raise ValueError(f"Expected 76 selected Houston B24 tiles, found {len(products)}")
        FULL_SOURCE.mkdir(parents=True, exist_ok=True)
        FULL_REPROJECTED.mkdir(parents=True, exist_ok=True)
        log.write(f"Selected {len(products)} catalog tiles for {len(points):,} training rows.")

        rasters: list[Path] = []
        for number, product in enumerate(products, start=1):
            started = time.perf_counter()
            name = Path(urllib.parse.urlparse(product["downloadURL"]).path).name
            expected_size = int(product["sizeInBytes"])
            source = find_valid_file(name, (PILOT_SOURCE, FULL_SOURCE), expected_size)
            download_seconds = 0.0
            downloaded = source is None
            if source is None:
                source, download_seconds = download_file(
                    product["downloadURL"],
                    FULL_SOURCE / name,
                    expected_size,
                )

            projected, conversion_seconds, reused_projection = ensure_reprojected(
                source,
                FULL_REPROJECTED / name,
            )
            rasters.append(projected)
            status = (
                f"Tile {number:02d}/76 | "
                f"download={'done' if downloaded else 'skip'} {download_seconds:.1f}s | "
                f"reproject={'skip' if reused_projection else 'done'} {conversion_seconds:.1f}s | "
                f"elapsed={time.perf_counter() - log.started:.1f}s | {name}"
            )
            log.write(status)

        log.write("All 76 source rasters are available and reprojected.")
        row_indices, has_tile = build_training_tiles(points, rasters, log)
        tiles = np.load(TILE_PATH, mmap_mode="r")
        has_missing = np.load(MISSING_PATH, mmap_mode="r")
        valid_rows = np.flatnonzero(has_tile)
        if len(valid_rows) < 2000:
            raise ValueError(f"Need 2,000 rows with valid DEM centers, found {len(valid_rows)}")
        sample = np.sort(
            np.random.default_rng(SEED).choice(valid_rows, size=2000, replace=False)
        )
        transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
        xs, ys = transformer.transform(
            points["lon"].to_numpy()[sample],
            points["lat"].to_numpy()[sample],
        )
        predictions = np.full(2000, np.nan, dtype=np.float64)
        open_rasters = [rasterio.open(path) for path in rasters]
        try:
            for position, (x, y) in enumerate(zip(xs, ys)):
                _, center, _, _ = read_valid_center(open_rasters, x, y)
                if center is not None:
                    predictions[position] = float(center[0, 0])
        finally:
            for dataset in open_rasters:
                dataset.close()

        valid_sample = np.isfinite(predictions)
        if valid_sample.sum() != 2000:
            raise ValueError(f"Elevation check found only {valid_sample.sum()} valid samples.")
        truth = points["elevation_m"].to_numpy()[sample].astype(np.float64)
        correlation = float(np.corrcoef(truth, predictions)[0, 1])
        mae = float(np.mean(np.abs(truth - predictions)))
        log.write(
            f"Elevation check (2,000 seed-42 rows): correlation={correlation:.6f}, "
            f"MAE={mae:.6f} m."
        )

        flooded = points["flooded"].to_numpy() == 1
        missing = np.asarray(has_missing, dtype=bool)
        valid_tile = np.asarray(has_tile, dtype=bool)
        for label, selection in (
            ("flooded", flooded),
            ("not_flooded", ~flooded),
        ):
            eligible = selection & valid_tile
            log.write(
                f"Missing-pixel share for {label}: "
                f"{int((eligible & missing).sum())}/{int(eligible.sum())}="
                f"{(eligible & missing).sum() / max(eligible.sum(), 1):.6%}; "
                f"valid center tile rows={int(eligible.sum())}."
            )
        missing_rows = missing & valid_tile
        complete_rows = ~missing & valid_tile
        log.write(
            f"Flooded share among tiles with missing pixels: "
            f"{int((flooded & missing_rows).sum())}/{int(missing_rows.sum())}="
            f"{(flooded & missing_rows).sum() / max(missing_rows.sum(), 1):.6%}; "
            f"among complete tiles: "
            f"{int((flooded & complete_rows).sum())}/{int(complete_rows.sum())}="
            f"{(flooded & complete_rows).sum() / max(complete_rows.sum(), 1):.6%}."
        )
        log.write(
            f"Rows with valid center tile={int(valid_tile.sum()):,}/{len(points):,}; "
            f"rows without valid center tile={int((~valid_tile).sum()):,}; "
            f"tiles with any missing/nodata pixels={int((missing & valid_tile).sum()):,}."
        )
        save_examples(points, tiles, valid_tile.astype(np.uint8))
        log.write(
            f"{'Saved' if not FIGURE_PATH.exists() else 'Available'} example figure: "
            f"{FIGURE_PATH}"
        )
        log.write(
            f"Saved terrain array ({TILE_PATH.stat().st_size:,} bytes) and row index "
            f"({INDEX_PATH.stat().st_size:,} bytes); valid row indices={len(row_indices):,}."
        )
        log.write(f"Run completed in {time.perf_counter() - log.started:.1f} seconds.")
    finally:
        log.close()


if __name__ == "__main__":
    main()
