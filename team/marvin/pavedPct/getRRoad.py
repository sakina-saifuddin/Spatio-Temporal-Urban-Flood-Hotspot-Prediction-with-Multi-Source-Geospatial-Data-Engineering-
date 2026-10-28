import os
import requests
import osmium
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely

from shapely.geometry import LineString

# Settings
INPUT_CSV = "data/input.csv"
OUTPUT_CSV = "data/out/paved_pct.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OSM_PBF = os.path.join(SCRIPT_DIR, "texas-261023.osm.pbf")
OSM_PBF_URL = "https://download.geofabrik.de/north-america/us/texas-261023.osm.pbf"

ROADS_CACHE = os.path.join(SCRIPT_DIR, "texas_highways_5070.gpkg")

CENSUS_YEAR = 2022
BUFFER_DISTANCE = 1609.344
CRS_AREA = "EPSG:5070"

HIGH_VOLUME_HIGHWAYS = {"motorway", "trunk", "primary"}

def download_osm_pbf():
    """Download the Texas OSM PBF file next to this script if missing."""
    if os.path.exists(OSM_PBF):
        print(f"OSM PBF already exists: {OSM_PBF}")
        return

    print(f"Downloading OSM PBF from {OSM_PBF_URL}")

    with requests.get(OSM_PBF_URL, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()

        # Write to a temporary file to avoid leaving a partial download.
        temp_path = OSM_PBF + ".part"
        try:
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            os.replace(temp_path, OSM_PBF)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print(f"Download complete: {OSM_PBF}")

def download_texas_tracts(year):
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_48_tract.zip"
    filename = f"tl_{year}_48_tract.zip"

    if not os.path.exists(filename):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)

    return gpd.read_file(filename)[["GEOID", "geometry"]]


class HighwayHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def way(self, w):
        if w.tags.get("highway") not in HIGH_VOLUME_HIGHWAYS:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        if len(coords) >= 2:
            self.lines.append(LineString(coords))


def load_highways():
    """Extract once, reproject once, cache to disk."""
    if os.path.exists(ROADS_CACHE):
        return gpd.read_file(ROADS_CACHE)

    handler = HighwayHandler()
    # sparse_mem_array is much lighter than the default for a state-sized file
    handler.apply_file(OSM_PBF, locations=True, idx="sparse_mem_array")

    if not handler.lines:
        raise ValueError("No matching highways found")

    roads = gpd.GeoDataFrame(geometry=handler.lines, crs="EPSG:4326").to_crs(CRS_AREA)
    roads.to_file(ROADS_CACHE, driver="GPKG")
    return roads


def compute_paved_pct(tract_geom, roads, road_geoms):
    # Only roads that could possibly reach this tract after buffering
    idx = roads.sindex.query(tract_geom.buffer(BUFFER_DISTANCE), predicate="intersects")
    if len(idx) == 0:
        return 0.0

    nearby = road_geoms[idx]
    road_buffer = shapely.union_all(shapely.buffer(nearby, BUFFER_DISTANCE, quad_segs=8))

    return tract_geom.intersection(road_buffer).area / tract_geom.area


if __name__ == "__main__":
    download_osm_pbf()
    df = pd.read_csv(INPUT_CSV)
    tracts = download_texas_tracts(CENSUS_YEAR)
    roads = load_highways()

    # Vectorized point-in-tract lookup for ALL rows at once
    points = gpd.GeoDataFrame(
        df[["lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(tracts.crs)

    joined = gpd.sjoin(points, tracts, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # guard against boundary dupes
    df["GEOID"] = joined["GEOID"].reindex(df.index)

    missing = df["GEOID"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows had no Census tract")

    # Reproject tracts once, compute once per unique tract
    tracts_proj = tracts.to_crs(CRS_AREA).set_index("GEOID")
    road_geoms = roads.geometry.values
    unique_geoids = df["GEOID"].dropna().unique()

    paved_pct_by_geoid = {}
    for n, geoid in enumerate(unique_geoids, 1):
        paved_pct_by_geoid[geoid] = compute_paved_pct(
            tracts_proj.geometry.loc[geoid], roads, road_geoms
        )
        if n % 50 == 0 or n == len(unique_geoids):
            print(f"Tracts processed: {n}/{len(unique_geoids)}")

    df["paved_pct"] = df["GEOID"].map(paved_pct_by_geoid)
    df[["lat", "lon", "paved_pct"]].to_csv(OUTPUT_CSV, index=False)

    print(f"Done! Results saved to {OUTPUT_CSV}")