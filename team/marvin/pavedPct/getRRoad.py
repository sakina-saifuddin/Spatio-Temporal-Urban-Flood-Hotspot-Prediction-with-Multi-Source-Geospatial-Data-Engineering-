
import os
import requests
import osmium
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely

from shapely.geometry import LineString


# ============================================================
# 1. Settings and file paths
# ============================================================

DATA_DIR = "data/marvin"

INPUT_CSV = os.path.join(DATA_DIR, "input.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "out", "paved_pct.csv")

OSM_PBF = os.path.join(DATA_DIR, "texas-261023.osm.pbf")
OSM_PBF_URL = (
    "https://download.geofabrik.de/north-america/us/"
    "texas-261023.osm.pbf"
)

ROADS_CACHE = os.path.join(
    DATA_DIR,
    "texas_highways_5070.gpkg"
)

CENSUS_YEAR = 2022

BUFFER_DISTANCE = 1609.344  # 1 mile in meters
CRS_AREA = "EPSG:5070"

HIGH_VOLUME_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
}


# ============================================================
# 2. Create required directories
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_CSV}"
    )


# ============================================================
# 3. Download Texas OSM PBF if missing
# ============================================================

def download_osm_pbf():
    """Download the Texas OSM PBF file if it is missing."""

    if os.path.exists(OSM_PBF):
        print(f"OSM PBF already exists: {OSM_PBF}")
        return

    print(f"Downloading OSM PBF from {OSM_PBF_URL}")

    temp_path = OSM_PBF + ".part"

    try:
        with requests.get(
            OSM_PBF_URL,
            stream=True,
            timeout=(30, 300)
        ) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        os.replace(temp_path, OSM_PBF)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"Download complete: {OSM_PBF}")


# ============================================================
# 4. Download Texas Census tracts
# ============================================================

def download_texas_tracts(year):
    """Download and load Texas Census tract boundaries."""

    url = (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
        f"tl_{year}_48_tract.zip"
    )

    filename = os.path.join(
        DATA_DIR,
        f"tl_{year}_48_tract.zip"
    )

    if not os.path.exists(filename):
        print(f"Downloading Census tracts for {year}...")

        response = requests.get(
            url,
            timeout=120
        )
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {filename}")
    else:
        print(f"Census tract ZIP already exists: {filename}")

    tracts = gpd.read_file(filename)[
        ["GEOID", "geometry"]
    ]

    return tracts


# ============================================================
# 5. Extract high-volume highways from OSM
# ============================================================

class HighwayHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def way(self, w):
        if w.tags.get("highway") not in HIGH_VOLUME_HIGHWAYS:
            return

        try:
            coords = [
                (node.lon, node.lat)
                for node in w.nodes
            ]
        except osmium.InvalidLocationError:
            return

        if len(coords) >= 2:
            self.lines.append(LineString(coords))


# ============================================================
# 6. Load highways and cache in projected CRS
# ============================================================

def load_highways():
    """Extract highways once, reproject once, and cache to disk."""

    if os.path.exists(ROADS_CACHE):
        print(f"Loading highway cache: {ROADS_CACHE}")
        return gpd.read_file(ROADS_CACHE)

    if not os.path.isfile(OSM_PBF):
        raise FileNotFoundError(
            f"OSM PBF file not found: {OSM_PBF}"
        )

    print("Extracting high-volume highways from OSM...")

    handler = HighwayHandler()

    handler.apply_file(
        OSM_PBF,
        locations=True,
        idx="sparse_mem_array"
    )

    if not handler.lines:
        raise ValueError("No matching highways found.")

    print(f"Extracted {len(handler.lines)} highway segments.")

    roads = gpd.GeoDataFrame(
        geometry=handler.lines,
        crs="EPSG:4326"
    ).to_crs(CRS_AREA)

    roads.to_file(
        ROADS_CACHE,
        driver="GPKG"
    )

    print(f"Highway cache saved: {ROADS_CACHE}")

    return roads


# ============================================================
# 7. Compute paved percentage for one Census tract
# ============================================================

def compute_paved_pct(tract_geom, roads, road_geoms):
    """
    Calculate the fraction of tract area covered by buffers
    around nearby high-volume highways.
    """

    if tract_geom is None or tract_geom.is_empty:
        return None

    if tract_geom.area == 0:
        return None

    # Find roads intersecting the tract's expanded area
    search_area = tract_geom.buffer(BUFFER_DISTANCE)

    idx = roads.sindex.query(
        search_area,
        predicate="intersects"
    )

    if len(idx) == 0:
        return 0.0

    nearby = road_geoms[idx]

    # Buffer roads by one mile and merge overlapping buffers
    road_buffer = shapely.union_all(
        shapely.buffer(
            nearby,
            BUFFER_DISTANCE,
            quad_segs=8
        )
    )

    # Fraction of tract area inside the road-buffer region
    covered_area = tract_geom.intersection(
        road_buffer
    ).area

    return float(
        min(covered_area / tract_geom.area, 1.0)
    )


# ============================================================
# 8. Main processing
# ============================================================

if __name__ == "__main__":

    # Download OSM data if missing
    download_osm_pbf()

    # Read input CSV
    df = pd.read_csv(INPUT_CSV)

    required_columns = {"lat", "lon"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}"
        )

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Download Census tract boundaries
    tracts = download_texas_tracts(CENSUS_YEAR)

    # Load or extract highways
    roads = load_highways()

    # --------------------------------------------------------
    # 8a. Assign each input point to a Census tract
    # --------------------------------------------------------

    valid_coords = (
        df["lat"].notna()
        & df["lon"].notna()
        & df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
    )

    points = gpd.GeoDataFrame(
        df.loc[valid_coords, ["lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(
            df.loc[valid_coords, "lon"],
            df.loc[valid_coords, "lat"]
        ),
        crs="EPSG:4326"
    )

    points = points.to_crs(tracts.crs)

    joined = gpd.sjoin(
        points,
        tracts,
        how="left",
        predicate="within"
    )

    # Handle points that match more than one tract
    joined = joined[
        ~joined.index.duplicated(keep="first")
    ]

    df["GEOID"] = joined["GEOID"].reindex(df.index)

    missing = df["GEOID"].isna().sum()

    if missing:
        print(
            f"Warning: {missing} rows had no Census tract "
            "(including invalid or out-of-area coordinates)."
        )

    # --------------------------------------------------------
    # 8b. Reproject tracts and calculate each unique tract once
    # --------------------------------------------------------

    tracts_proj = tracts.to_crs(CRS_AREA).set_index("GEOID")

    road_geoms = roads.geometry.values

    unique_geoids = df["GEOID"].dropna().unique()

    paved_pct_by_geoid = {}

    for n, geoid in enumerate(unique_geoids, start=1):

        tract_geom = tracts_proj.geometry.loc[geoid]

        paved_pct_by_geoid[geoid] = compute_paved_pct(
            tract_geom,
            roads,
            road_geoms
        )

        if n % 50 == 0 or n == len(unique_geoids):
            print(
                f"Tracts processed: {n}/{len(unique_geoids)}"
            )

    # --------------------------------------------------------
    # 8c. Assign results to input points
    # --------------------------------------------------------

    df["paved_pct"] = df["GEOID"].map(
        paved_pct_by_geoid
    )

    # --------------------------------------------------------
    # 8d. Save output
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    df[["lat", "lon", "paved_pct"]].to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(f"\nDone! Results saved to {OUTPUT_CSV}")
    print(f"Total input records: {len(df)}")
    print(f"Records with paved_pct: {df['paved_pct'].notna().sum()}")
    print(df[["lat", "lon", "paved_pct"]].head())
