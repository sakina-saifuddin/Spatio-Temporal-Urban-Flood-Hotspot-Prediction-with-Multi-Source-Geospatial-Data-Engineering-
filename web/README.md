# FloodSense 2.0 (web)

FloodSense 2.0 is a single-page Leaflet map, guided by Scout the mascot, showing which parts of Houston have land that looks like places
that flooded in past storms, and why. No backend: it reads precomputed JSON, so it can be hosted on GitHub Pages.

## Run locally

From the repository root:

```bash
python web/scripts/build_data.py          # regenerate web/data/*.json from app/data/
python -m http.server 8000 --directory web
```

Open http://localhost:8000. (Opening `index.html` directly doesn't work because browsers block `fetch` from `file://`.)

## Features

- **Street map + color glow**: Esri street map (OpenStreetMap backup) with a smooth, see-through green → red glow
  blended from ~4,500 ~1 km areas (0.5 = the model's cutoff); scroll or double-click to zoom
- **Click anywhere**: Scout explains the score in plain words (from SHAP), with "what pushed the score" bars,
  311 reports per storm, and the land features
- **Past storms** (floating bar on the map): a small dot for every 311 flood-report location (Memorial Day 2015, Tax Day 2016, Harvey 2017,
  Imelda 2019). Hover a dot for its street name; click it to zoom in, drop a pin and open the area card
- **Top 10 hotspots**: highest-scoring well-sampled areas (≥ 5 sampled spots), click to fly there
- **Place search**: OpenStreetMap Nominatim, limited to the Houston area; unsampled places say so honestly
- **Blue flood on the street**: clicking a report dot zooms to street level; once the zoom ends, water spreads along
  the street from the report (only the stretch within ~90 m of the storm's reports), then shimmers, with ripples
  at the report. Street shape comes back with the same reverse-geocoding request.
- **Street photos** (sidebar): flood photos within 3 km from Wikimedia Commons (credited, distance shown);
  if none, an aerial photo of the street (Esri World Imagery). Always a link to Google Street View.
- **Selected spot** (sidebar): street name, neighborhood and area level for whatever you clicked, plus recently viewed streets
- **Street names**: looked up on demand from the coordinates (OpenStreetMap Nominatim reverse geocoding,
  max 1 request/second, cached in the page). The raw 311 files only contain coordinates, not addresses.
- **Welcome screen**: Scout's pop-up opens every time the map loads
- **Tabs**: 🗺️ Flood Map · 💡 The Project (motivation, question, storms, datasets, cleaning, accuracy, limits,
  official links) · 🛠️ How We Built It (pipeline, data splitting, models and scores, Imelda test,
  tuning, SHAP, tech stack). Linkable as `#map`, `#project`, `#build`. Numbers come from the project docs and
  `reports/`; charts are copied from `reports/figures/` into `assets/figures/`.

## Files

- `index.html`: page structure, welcome and About screens
- `style.css`: look and layout (UH red accent, phone layout below 820 px)
- `app.js`: all behavior, in numbered sections
- `assets/scout.png`: the mascot (copied from `assets/images/mascot.png`)
- `data/cells.json`, `data/storms.json`, `data/hotspots.json`: built by `scripts/build_data.py`
- `scripts/place_names_cache.json`: cached hotspot neighborhood names (avoids re-querying OpenStreetMap)

## Live site (Netlify)

- Live: https://floodsense-2.netlify.app
- Netlify project: `floodsense-2` (admin: https://app.netlify.com/projects/floodsense-2)
- **Auto-deploy:** Netlify is linked to this GitHub repo. Every push to `main` is deployed automatically
  (settings come from `netlify.toml` at the repo root: base `web/`, no build step).

When you change the site, bump the `?v=` numbers on `style.css` / `app.js` in `index.html` so browsers fetch the
new files, then commit and push to `main`.
