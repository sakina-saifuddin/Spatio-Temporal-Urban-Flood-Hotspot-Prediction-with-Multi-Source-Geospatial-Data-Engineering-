
import os
import requests

from concurrent.futures import ThreadPoolExecutor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "houston_dem_tiles")
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

response = requests.get(API_URL, params=params, timeout=60)
response.raise_for_status()
data = response.json()

products = data.get("items", [])
print(f"Products returned: {len(products)}")

# Print product names and URLs so you can verify the results
downloads = []

for item in products:
    url = item.get("downloadURL")
    title = item.get("title", "dem_tile")

    if url and url.lower().split("?")[0].endswith(
        (".tif", ".tiff", ".zip")
    ):
        downloads.append((title, url))

def download_file(product):
    title, url = product
    filename = os.path.basename(url.split("?")[0])

    if not filename:
        filename = title.replace(" ", "_") + ".tif"

    path = os.path.join(OUT_DIR, filename)

    if os.path.exists(path):
        return f"Already downloaded: {filename}"

    try:
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return f"Downloaded: {filename}"

    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return f"Failed: {url} ({e})"

with ThreadPoolExecutor(max_workers=4) as executor:
    for result in executor.map(download_file, downloads):
        print(result)

print("Download step complete.")
