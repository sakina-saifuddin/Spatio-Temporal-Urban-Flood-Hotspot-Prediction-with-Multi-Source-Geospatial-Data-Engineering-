
import os
import requests

from concurrent.futures import ThreadPoolExecutor


# ============================================================
# 1. Paths and configuration
# ============================================================

DATA_DIR = "data/marvin"
OUT_DIR = os.path.join(DATA_DIR, "houston_dem_tiles")

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Approximate Houston-area bounding box:
# west longitude, south latitude, east longitude, north latitude
BBOX = "-95.9,29.4,-94.9,30.2"

API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

params = {
    "datasets": "Digital Elevation Model (DEM) 1 meter",
    "bbox": BBOX,
    "prodFormats": "GeoTIFF",
    "max": 1000,
    "outputFormat": "JSON",
}


# ============================================================
# 2. Query The National Map API
# ============================================================

print("Searching for Houston-area DEM products...")

response = requests.get(
    API_URL,
    params=params,
    timeout=60
)
response.raise_for_status()

data = response.json()
products = data.get("items", [])

print(f"Products returned: {len(products)}")


# ============================================================
# 3. Collect downloadable GeoTIFF and ZIP URLs
# ============================================================

downloads = []

for item in products:
    url = item.get("downloadURL")
    title = item.get("title", "dem_tile")

    if url and url.lower().split("?")[0].endswith(
        (".tif", ".tiff", ".zip")
    ):
        downloads.append((title, url))

print(f"Downloadable files found: {len(downloads)}")


# ============================================================
# 4. Download each file
# ============================================================

def download_file(product):
    title, url = product

    filename = os.path.basename(url.split("?")[0])

    if not filename:
        filename = title.replace(" ", "_") + ".tif"

    path = os.path.join(OUT_DIR, filename)

    if os.path.exists(path):
        return f"Already downloaded: {filename}"

    try:
        with requests.get(
            url,
            stream=True,
            timeout=180
        ) as r:
            r.raise_for_status()

            with open(path, "wb") as f:
                for chunk in r.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        return f"Downloaded: {filename}"

    except Exception as e:
        if os.path.exists(path):
            os.remove(path)

        return f"Failed: {url} ({e})"


# ============================================================
# 5. Download files concurrently
# ============================================================

with ThreadPoolExecutor(max_workers=4) as executor:
    for result in executor.map(download_file, downloads):
        print(result)


# ============================================================
# 6. Summary
# ============================================================

print("\nDownload step complete.")
print(f"Output directory: {OUT_DIR}")
print(f"Files currently in output directory: {len(os.listdir(OUT_DIR))}")
