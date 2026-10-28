import io
import os
import sys
import numpy as np
import pandas as pd
import requests

NCEI_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
STATIONS_CACHE = "ghcnd-stations.txt"
RAW_DIR = "noaa_raw"        # raw downloads, one file per request
GAUGE_DIR = "gauges"        # cleaned gauge table per storm date
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(GAUGE_DIR, exist_ok=True)
K_NEAREST = 3      # number of gauges averaged per point
PAD_DEG = 0.3      # extra margin around your points so nearby gauges are included


def load_stations(bbox):
    """GHCN-Daily station IDs inside bbox = (south, west, north, east)."""
    if not os.path.exists(STATIONS_CACHE):
        print("Downloading station list (one time, ~10 MB)...")
        r = requests.get(STATIONS_URL, timeout=300)
        r.raise_for_status()
        with open(STATIONS_CACHE, "w", encoding="utf-8") as f:
            f.write(r.text)
    st = pd.read_fwf(
        STATIONS_CACHE,
        colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)],
        names=["id", "lat", "lon", "elev", "state", "name"],
        header=None,
    )
    s, w, n, e = bbox
    st = st[st.lat.between(s, n) & st.lon.between(w, e)]
    return st["id"].tolist()


def fetch_gauges(date, station_ids, days_before=0, days_after=0, chunk=40):
    """Gauge lat/lon + rainfall (inches) for one storm date (or date window)."""
    d = pd.to_datetime(date)
    first = d - pd.Timedelta(days=days_before)
    last = d + pd.Timedelta(days=days_after)

    frames = []
    for i in range(0, len(station_ids), chunk):
        params = {
            "dataset": "daily-summaries",
            "dataTypes": "PRCP",
            "stations": ",".join(station_ids[i:i + chunk]),
            "startDate": first.strftime("%Y-%m-%d"),
            "endDate": (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # must be > start
            "units": "standard",
            "includeStationLocation": "1",
            "format": "csv",
        }

        raw_path = os.path.join(
            RAW_DIR,
            f"prcp_{first:%Y%m%d}_{last:%Y%m%d}_part{i // chunk:03d}.csv",
        )

        if os.path.exists(raw_path):                 # already downloaded earlier
            text = open(raw_path, encoding="utf-8").read()
        else:
            r = requests.get(NCEI_URL, params=params, timeout=120)
            if not r.ok:
                print("HTTP", r.status_code, r.url)
                print("Server said:", r.text[:500])
                r.raise_for_status()
            text = r.text
            with open(raw_path, "w", encoding="utf-8") as f:   # save raw response
                f.write(text)

        if text.strip():
            frames.append(pd.read_csv(io.StringIO(text)))

    if not frames:
        return pd.DataFrame(columns=["glat", "glon", "rain"])

    df = pd.concat(frames, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[(df["DATE"] >= first) & (df["DATE"] <= last)]
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")
    df = df.dropna(subset=["PRCP", "LATITUDE", "LONGITUDE"])
    g = df.groupby(["STATION", "LATITUDE", "LONGITUDE"], as_index=False)["PRCP"].sum()
    return g.rename(columns={"LATITUDE": "glat", "LONGITUDE": "glon", "PRCP": "rain"})


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def idw_rain(lat, lon, gauges, k=K_NEAREST):
    """Inverse-distance-weighted rainfall from the k nearest gauges."""
    if gauges.empty:
        return np.nan
    d = haversine_km(lat, lon, gauges["glat"].values, gauges["glon"].values)
    idx = np.argsort(d)[:k]
    d_k = np.maximum(d[idx], 0.1)          # avoid divide-by-zero
    w = 1.0 / d_k ** 2
    return float(np.sum(w * gauges["rain"].values[idx]) / np.sum(w))


def main(in_csv, out_csv, days_before=0, days_after=0):
    pts = pd.read_csv(in_csv)

    bbox = (pts.lat.min() - PAD_DEG, pts.lon.min() - PAD_DEG,
            pts.lat.max() + PAD_DEG, pts.lon.max() + PAD_DEG)
    station_ids = load_stations(bbox)
    print(f"{len(station_ids)} stations in area")

    pts["rainfall_inches"] = np.nan
    for date, idx in pts.groupby("storm_date").groups.items():   # one download per date
        print(f"Fetching gauges for {date} ({len(idx)} points) ...")
        gauges = fetch_gauges(date, station_ids, days_before, days_after)
        print(f"  {len(gauges)} gauges with data")
        pts.loc[idx, "rainfall_inches"] = [
            idw_rain(la, lo, gauges)
            for la, lo in zip(pts.loc[idx, "lat"], pts.loc[idx, "lon"])
        ]

    out = pts[["lat", "lon", "storm_date", "rainfall_inches"]]
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")


main("data/input.csv", "data/out/rain.csv")
