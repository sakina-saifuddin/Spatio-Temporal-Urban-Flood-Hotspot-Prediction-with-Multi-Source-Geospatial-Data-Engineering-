
import io
import os

import numpy as np
import pandas as pd
import requests


# ============================================================
# 1. Paths and configuration
# ============================================================

DATA_DIR = "data/marvin"

INPUT_CSV = os.path.join(DATA_DIR, "input.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "out", "rain.csv")

NCEI_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

STATIONS_URL = (
    "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
    "ghcnd-stations.txt"
)

STATIONS_CACHE = os.path.join(DATA_DIR, "ghcnd-stations.txt")

RAW_DIR = os.path.join(DATA_DIR, "noaa_raw")
GAUGE_DIR = os.path.join(DATA_DIR, "gauges")

K_NEAREST = 3
PAD_DEG = 0.3

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(GAUGE_DIR, exist_ok=True)


# ============================================================
# 2. Load NOAA GHCN-Daily stations
# ============================================================

def load_stations(bbox):
    """
    Return GHCN-Daily station IDs inside bbox.

    bbox = (south, west, north, east)
    """

    if not os.path.exists(STATIONS_CACHE):
        print("Downloading NOAA station list (one time, ~10 MB)...")

        response = requests.get(
            STATIONS_URL,
            timeout=300
        )
        response.raise_for_status()

        with open(
            STATIONS_CACHE,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(response.text)

        print(f"Station list saved to {STATIONS_CACHE}")

    stations = pd.read_fwf(
        STATIONS_CACHE,
        colspecs=[
            (0, 11),
            (12, 20),
            (21, 30),
            (31, 37),
            (38, 40),
            (41, 71),
        ],
        names=[
            "id",
            "lat",
            "lon",
            "elev",
            "state",
            "name",
        ],
        header=None,
    )

    south, west, north, east = bbox

    stations = stations[
        stations["lat"].between(south, north)
        & stations["lon"].between(west, east)
    ]

    return stations["id"].tolist()


# ============================================================
# 3. Fetch rainfall observations from NOAA
# ============================================================

def fetch_gauges(
    date,
    station_ids,
    days_before=0,
    days_after=0,
    chunk=40,
):
    """
    Fetch station locations and rainfall in inches for a date
    or date window.
    """

    date = pd.to_datetime(date)

    first = date - pd.Timedelta(days=days_before)
    last = date + pd.Timedelta(days=days_after)

    frames = []

    for i in range(0, len(station_ids), chunk):
        batch = station_ids[i:i + chunk]

        params = {
            "dataset": "daily-summaries",
            "dataTypes": "PRCP",
            "stations": ",".join(batch),
            "startDate": first.strftime("%Y-%m-%d"),
            "endDate": (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "units": "standard",
            "includeStationLocation": "1",
            "format": "csv",
        }

        raw_path = os.path.join(
            RAW_DIR,
            f"prcp_{first:%Y%m%d}_{last:%Y%m%d}_part{i // chunk:03d}.csv",
        )

        if os.path.exists(raw_path):
            print(f"Using cached rainfall data: {os.path.basename(raw_path)}")

            with open(raw_path, "r", encoding="utf-8") as f:
                text = f.read()

        else:
            response = requests.get(
                NCEI_URL,
                params=params,
                timeout=120
            )

            if not response.ok:
                print("HTTP error:", response.status_code)
                print("Request URL:", response.url)
                print("Server response:", response.text[:500])
                response.raise_for_status()

            text = response.text

            with open(
                raw_path,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(text)

        if text.strip():
            try:
                frame = pd.read_csv(io.StringIO(text))

                if not frame.empty:
                    frames.append(frame)

            except pd.errors.EmptyDataError:
                continue

    if not frames:
        return pd.DataFrame(
            columns=["glat", "glon", "rain"]
        )

    df = pd.concat(frames, ignore_index=True)

    required_columns = {
        "DATE",
        "STATION",
        "PRCP",
        "LATITUDE",
        "LONGITUDE",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "NOAA response is missing required columns. "
            f"Received: {list(df.columns)}"
        )

    df["DATE"] = pd.to_datetime(
        df["DATE"],
        errors="coerce"
    )

    df = df[
        (df["DATE"] >= first)
        & (df["DATE"] <= last)
    ].copy()

    df["PRCP"] = pd.to_numeric(
        df["PRCP"],
        errors="coerce"
    )

    df["LATITUDE"] = pd.to_numeric(
        df["LATITUDE"],
        errors="coerce"
    )

    df["LONGITUDE"] = pd.to_numeric(
        df["LONGITUDE"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["PRCP", "LATITUDE", "LONGITUDE"]
    )

    # Sum daily precipitation across the requested date window
    gauges = df.groupby(
        ["STATION", "LATITUDE", "LONGITUDE"],
        as_index=False
    )["PRCP"].sum()

    gauges = gauges.rename(
        columns={
            "LATITUDE": "glat",
            "LONGITUDE": "glon",
            "PRCP": "rain",
        }
    )

    # Save cleaned gauge data for reuse/reference
    gauge_path = os.path.join(
        GAUGE_DIR,
        f"gauges_{first:%Y%m%d}_{last:%Y%m%d}.csv"
    )

    gauges.to_csv(gauge_path, index=False)

    return gauges


# ============================================================
# 4. Calculate distance between coordinates
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distances in kilometers."""

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        (lat1, lon1, lat2, lon2)
    )

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ============================================================
# 5. Estimate rainfall using inverse-distance weighting
# ============================================================

def idw_rain(lat, lon, gauges, k=K_NEAREST):
    """
    Estimate rainfall from the k nearest gauges using
    inverse-distance-squared weighting.
    """

    if gauges.empty:
        return np.nan

    distances = haversine_km(
        lat,
        lon,
        gauges["glat"].to_numpy(),
        gauges["glon"].to_numpy(),
    )

    k = min(k, len(gauges))

    indices = np.argsort(distances)[:k]

    nearest_distances = distances[indices]
    nearest_rain = gauges["rain"].to_numpy()[indices]

    # If a gauge is effectively at the input point, use its reading
    exact = nearest_distances < 1e-9

    if exact.any():
        return float(nearest_rain[exact].mean())

    weights = 1.0 / np.maximum(nearest_distances, 0.1) ** 2

    return float(
        np.sum(weights * nearest_rain) / np.sum(weights)
    )


# ============================================================
# 6. Main processing
# ============================================================

def main(
    in_csv,
    out_csv,
    days_before=0,
    days_after=0,
):
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(
            f"Input CSV not found: {in_csv}"
        )

    pts = pd.read_csv(in_csv)

    required_columns = {
        "lat",
        "lon",
        "storm_date",
    }

    missing_columns = required_columns - set(pts.columns)

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}"
        )

    pts["lat"] = pd.to_numeric(
        pts["lat"],
        errors="coerce"
    )

    pts["lon"] = pd.to_numeric(
        pts["lon"],
        errors="coerce"
    )

    pts["storm_date"] = pd.to_datetime(
        pts["storm_date"],
        errors="coerce"
    )

    # Find valid coordinate rows
    valid_coords = (
        pts["lat"].notna()
        & pts["lon"].notna()
        & pts["lat"].between(-90, 90)
        & pts["lon"].between(-180, 180)
    )

    valid_dates = pts["storm_date"].notna()

    usable = valid_coords & valid_dates

    pts["rainfall_inches"] = np.nan

    if not usable.any():
        raise ValueError(
            "No rows have both valid coordinates and storm_date values."
        )

    # Determine the geographic search area
    valid_points = pts.loc[usable]

    bbox = (
        valid_points["lat"].min() - PAD_DEG,
        valid_points["lon"].min() - PAD_DEG,
        valid_points["lat"].max() + PAD_DEG,
        valid_points["lon"].max() + PAD_DEG,
    )

    station_ids = load_stations(bbox)

    print(f"Found {len(station_ids)} NOAA stations in the search area.")

    if not station_ids:
        print("Warning: No NOAA stations found within the search area.")

    # Process each unique storm date once
    for date, indices in pts.loc[usable].groupby("storm_date").groups.items():
        print(
            f"Fetching gauges for {date:%Y-%m-%d} "
            f"({len(indices)} points)..."
        )

        gauges = fetch_gauges(
            date,
            station_ids,
            days_before=days_before,
            days_after=days_after,
        )

        print(f"  {len(gauges)} gauges with rainfall data.")

        pts.loc[indices, "rainfall_inches"] = [
            idw_rain(lat, lon, gauges)
            for lat, lon in zip(
                pts.loc[indices, "lat"],
                pts.loc[indices, "lon"],
            )
        ]

    # Save the output
    os.makedirs(
        os.path.dirname(out_csv) or ".",
        exist_ok=True
    )

    output = pts[
        [
            "lat",
            "lon",
            "storm_date",
            "rainfall_inches",
        ]
    ].copy()

    output.to_csv(out_csv, index=False)

    print(f"\nSaved {len(output)} rows to {out_csv}")
    print(
        "Rows with rainfall estimates: "
        f"{output['rainfall_inches'].notna().sum()}"
    )
    print(output.head())


# ============================================================
# 7. Run
# ============================================================

if __name__ == "__main__":
    main(
        INPUT_CSV,
        OUTPUT_CSV,
        days_before=0,
        days_after=0,
    )
