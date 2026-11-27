# FloodSense application

FloodSense is an interactive, explainable Streamlit application for exploring **historical flood susceptibility** across the
Houston study area. Its guide is Scout, a water droplet in a cowboy hat. It reads the validated, precomputed datasets in
`app/data/` and never trains models or computes SHAP values at runtime.

> FloodSense provides research-based historical flood susceptibility estimates. It is not a real-time flood forecast,
> official warning system, or property-level flood risk assessment.

The earlier FastAPI + React proposal in this file has been superseded. The approved architecture is **Streamlit**.

## Run locally (Windows / PowerShell, Python 3.11)

From the repository root:

```powershell
.venv\Scripts\Activate.ps1                       # or create a fresh venv
python -m pip install -r app/requirements.txt    # runtime only — no LightGBM/SHAP/PyTorch
python -m streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. Run the tests with:

```powershell
python -m pip install pytest==9.2.0
python -m pytest app/tests -q
```

## Structure

```
app/
  streamlit_app.py        entry point: page config, CSS, sidebar disclaimer, st.navigation (5 pages)
  requirements.txt        runtime dependencies only
  pages/                  home, explore, priority, historical, model_insights
  components/             mascot (Scout + explanation templates), maps (Folium), charts (Plotly),
                          navigation (page registry), disclaimers (notes, limitations, official links)
  utils/                  data_loader (cached loaders, pathlib paths), grid_utils (strict grid matching),
                          metrics (planner ranking/cutoffs, confusion checks), geocoding (Nominatim client)
  styles/custom.css       palette, Poppins/Inter fonts, focus styles, responsive rules
  tests/                  pytest + Streamlit AppTest suites
  data/                   the eight validated datasets (unchanged)
.streamlit/config.toml    light theme
```

All paths are resolved with `pathlib` relative to `app/`, so they work on Linux hosts. Figures are read from
`reports/figures/` and the mascot from `assets/images/mascot.png`, both read-only.

## Pages

| Page | Chapter | Main functionality |
|---|---|---|
| Home | Welcome to Houston | Scout introduction, purpose, Houston context, statistics read from metadata, plain-language susceptibility explanation, chapter navigation, official links |
| Explore Houston | Discover Your Neighborhood | 4 map layers (susceptibility, coverage, dominant factor, report-associated counts), click/coordinate/grid-ID/list/public-place selection, strict cell matching, no-data handling, a separate "nearby sampled cells" action, cell summary, local SHAP chart and table, two-area comparison, reset |
| Priority Areas | Build a More Resilient Houston | Ranking by mean score, max score, share at cutoff or sample count; Top 10/25/50/100/All; filters for minimum samples, score range and coverage; exploratory cutoff recalculated from location scores; summary stats; map and table selection; local SHAP; CSV export; empty state; reset |
| Historical Storms | Journey Through Houston's Flood History | Previous/next and direct storm selection; predicted, recorded-share and classification-share layers; side-by-side synchronized DualMap; classification filters; storm-specific cell bounds; cell breakdown; metrics; confusion matrix; cross-storm comparison; aggregated CSV export |
| Behind the Predictions | Discover the Science | Purpose, data sources, features, model and preprocessing, validation, Imelda test, tuning, rainfall exclusion, original vs. grouped SHAP, 8 research SHAP figures, limitations, team |

## Implementation checklist

- [x] **Batch A — Foundation:** schemas validated, structure, runtime requirements, `st.navigation` (exactly 5 pages), caching, CSS theme, shared components, Home.
- [x] **Batch B — Geographic exploration:** maps and legends, strict ties-to-even grid matching, Nominatim search, coordinate entry, selected-area summaries, SHAP chart, comparison.
- [x] **Batch C — Planner and historical tools:** Priority Areas and Historical Storm Explorer.
- [x] **Batch D — Transparency and polish:** Behind the Predictions, Scout templates, disclaimers, official links, focus styles, responsive CSS.
- [x] **Batch E — Testing and deployment prep:** automated tests, server launch, hash verification, clean-venv dependency check, deployment notes.
- [x] Manual browser review of all five pages and mobile layouts by the team (2026-11-25). Priority Areas, Historical
  Storms and Behind the Predictions passed.
- [x] Final refinement batch (see below).
- [ ] Manual browser re-check of the refinements, especially Explore map behaviour (see below).
- [ ] Public deployment — waiting for authorization and the team decisions listed under *Deployment readiness review*.

## Final refinement batch (2026-11-25)

- **Home:**
  - Primary-button calls to action for Explore Houston and Historical Storm Explorer.
  - Human-readable feature names in place of column names.
  - Scout now says "how much sampled data is available" instead of implying confidence.
  - Shorter sidebar disclaimer; the full disclaimer is still on every page.
  - File-path captions removed.
  - Statistic cards are now one responsive, equal-height grid.
- **Explore Houston:**
  - The legend sits above a full-width map and wraps on narrow screens.
  - Mouse-wheel zoom is off; +/− buttons, double-click and pinch still zoom.
  - Polygon opacity reduced from 0.72 to 0.60.
  - Corrected privacy wording: map clicks and coordinates are never sent to the geocoder, but basemap images come
    from `tile.openstreetmap.org`.
  - A "Selected: cell …" status bar sits under the map with a link to the details.
  - Metrics are wrapping cards, so the score range and strongest downward factor are no longer truncated.
  - SHAP axes use automargin and longer bars get extra room.
  - Environmental and comparison tables wrap without internal scrolling.
  - Buttons renamed "Set as Area A/B" and "↺ Reset map".
  - `paved_pct` (a 0–1 fraction) is shown as "N% of tract near major highways".
  - Area mean scores appear as cards above the comparison table.
- **Explore viewport:** before this batch, changing layers rebuilt the map and reset the viewport. The selection itself
  was kept. Now the cells, selection and comparison outlines go in through streamlit-folium's `feature_group_to_add`.
  Layer changes and map clicks update the map in place, so pan/zoom is kept. Searches, typed selections and reset still
  re-centre the map on purpose.
- **Priority Areas:**
  - The cutoff line is removed from the cell-mean histogram.
  - Only the top 10 ranks are labelled by default, with a toggle for up to 100.
  - Filters are in two columns, with full-width coverage categories that explain their thresholds.
  - The map and table are stacked at full width.
  - "Average cell score" is labelled as an unweighted average of cell means.
  - Sampled cells outside the active filters are shaded grey and labelled "outside current filters". Never-sampled
    areas stay unshaded "No data".
- **Memory:**
  - The large datasets and map GeoJSON now use `st.cache_resource`, so all sessions share one read-only copy.
    `st.cache_data` deserialized a new copy on every call.
  - Explore and Priority share one cached GeoJSON per layer.
  - Pages load only what they use: `historical_predictions.parquet` is not loaded by any page, and
    `location_predictions.parquet` only by Priority Areas.
  - Historical Storms and Behind the Predictions were left as they were, apart from shared chart margins and caching
    internals.

- **Data integrity:** all 8 files load and schemas match the metadata. Counts check out: 62,138 locations, 4,492 cells,
  85,678 historical observations, 15,873 grid-storm summaries and 4 storms. All scores fall in [0, 1], and there are no
  duplicate grid or historical IDs. Coverage counts are 224 / 720 / 3,548. Grid IDs are reproduced from coordinates, and
  baseline + grouped SHAP equals the raw score (max error < 1e-4). The frozen-cutoff flags match. Historical confusion
  counts and derived metrics agree with `storm_metrics.csv`. Imelda matches `reports/final_test_results.csv` exactly.
  Grouped SHAP importance is not the sum of the encoded values.
- **Logic:** ties-to-even rounding, strict matching with no silent substitution, nearby cells and coverage bins.
  Planner shares at 0.50 equal the saved `locations_above_cutoff`, and exploratory cutoffs are computed from location
  scores. Ranking, filters, top-N and the empty state are covered. Geocoding is tested with a mocked network: User-Agent,
  timeout, de-duplication, 429/500/bad JSON/timeout/connection errors, input limits, disable switch and configurability.
- **Pages (AppTest):** Home statistics, links and disclaimer. Explore: all layers, grid-ID selection, SHAP, comparison
  A/B, clear, reset, no-data cell, nearby action and invalid input. Priority: metric, top-25, exploratory cutoff warning,
  filters, empty state, reset and CSV button. Historical: all 4 storms, previous/next wrap-around, 3 layers,
  side-by-side, classification filter and empty state, cell metrics. Insights: all sections and figures.
- **Research protection:** all 28 hashes in `docs/app_source_hashes.csv` are unchanged, `reports/final_test.lock`
  exists, and rendering every page imports none of lightgbm, shap, torch, sklearn or xgboost.
- **Refinement regressions (added in the final batch; full suite now 55 passed):**
  - Home: readable feature names, new Scout wording, no file-path captions, both call-to-action buttons navigate.
  - Explore: selection status bar, percentage display, Area A/B cards, privacy wording names the tile host.
  - Explore: layer changes keep the selection and the base map, so the viewport is not recreated.
  - Priority: "Average cell score" with the unweighted note, the "outside current filters" legend, the rank-label toggle.
  - Charts: no histogram cutoff line, SHAP automargin.
  - Maps: scroll zoom disabled, excluded-cell styling that doesn't mutate the shared GeoJSON, shared cache identity.

**Memory (measured 2026-11-25; one Python process rendering every page through AppTest):**

| Step | Process memory (RSS) |
|---|---|
| Interpreter start | 65 MB |
| Home → Explore → Priority → Historical → Behind the Predictions | 131 → 178 → 222 → 243 → 235 MB |
| After all 4 Explore layers and all 4 storms | 291 MB (peak 311 MB) |
| 30 Explore↔Priority cycles in a fresh process | stays at 221–230 MB (no growth) |

These figures are for the application process alone. The live server was about 235 MB after manual browsing.
Total system memory is separate: this machine had about 0.7–1.2 GB free of 7.7 GB, mostly because of other
applications (Chrome alone used about 1.9 GB). Streamlit's "Manage your app" documentation lists Community Cloud
memory as 690 MB minimum to 2.7 GB maximum per app, and says the limits can change without notice. Even at the
690 MB minimum, the measured ~230–310 MB footprint leaves headroom. Large datasets are shared across sessions, so
memory grows mainly with per-session map payloads.

**Manual and live checks:**

- `python -m streamlit run app/streamlit_app.py` started and `/_stcore/health` returned `ok`.
- One live Nominatim query ("Discovery Green, Houston, TX") succeeded in 2.7 s.
- Browser rendering has **not** been verified by hand.

**Measured server-side script time** (AppTest, local Windows machine; excludes browser rendering and network):

| Interaction | Time |
|---|---|
| Home (cold / warm) | 2.9 s / 0.05 s |
| Explore first map / warm rerun | 3.1 s / 1.7 s |
| Explore cell selection | 2.1 s |
| Explore new layer (first time) | 2.3 s |
| Priority first load / cutoff change | 2.6 s / 2.3 s |
| Historical first load / storm change (cold / warm) | 1.5 s / 1.8 s / 1.3 s |
| Behind the Predictions | 1.2 s |
| Priority CSV build (all 4,492 cells) | 0.23 s |

Cloud cold starts, browser map rendering (about 1.5 MB of GeoJSON per development map) and geocoding latency have not
been measured.

## Differences from the specification and remaining limitations

- **Historical point overlays are not shown.** Following the privacy requirement, the historical explorer uses
  aggregated cell summaries only. CSV exports contain cell summaries without individual observation coordinates. Showing
  points needs team approval.
- **Side-by-side maps use `folium.plugins.DualMap`.** Pan and zoom are synchronized, but clicks are not captured, so
  cells are inspected with the cell selector.
- **Priority map/table sync is one-way for highlighting.** A map click or a table row both select the cell and update the
  map outline and detail panel. A map click does not highlight the matching table row, because Streamlit cannot set a
  dataframe selection after creation.
- **Hover tooltips have alternatives.** Grid-ID entry, a sorted cell list, data tables and text captions cover anything
  shown only on hover.
- **Basemap:** CartoDB tiles now need an API key, so the app uses standard OpenStreetMap tiles
  (`tile.openstreetmap.org`). Leaflet shows the attribution on the map. Heavy public traffic should move to a tile
  provider whose terms fit.
- **Priority map/table layout** stacks the map above the table at all widths, rather than side by side.
- **Geocoding:** only runs on explicit submission, sends at most one request per 1.1 s per app process, and identifies
  itself with the User-Agent `FloodSense-research-app/1.0`. Results stay in process memory only (up to 128 queries) to
  avoid repeat requests; nothing is written to disk. Settings: `FLOODSENSE_GEOCODER_URL`,
  `FLOODSENSE_GEOCODER_USER_AGENT`, `FLOODSENSE_GEOCODER_ENABLED`, as environment variables or Streamlit secrets.
  Before public launch, consider adding a contact URL or email to the User-Agent, as the Nominatim policy recommends.
- **Historical photographs are not used.** Their licences and attribution are not verified (`assets/images/README.md`).
- **Accessibility and mobile:** WCAG 2.2 AA principles were followed (focus outlines, alt text, table alternatives,
  text plus colour, ▲/▼ markers), but no formal audit or mobile-device test has been done.
- **Fonts:** Poppins and Inter load from Google Fonts, with system fallbacks.
- All research limitations in `model_metadata.json` apply and are shown in the app.

## Deployment readiness review (2026-11-25)

**Ready:**

- Entry point `app/streamlit_app.py`. Runtime dependencies are in `app/requirements.txt` (no training libraries), which
  Community Cloud reads from the entry point's folder.
- All paths are relative (`pathlib`). There are no absolute local paths in the application code, no secrets file, and
  no credentials.
- The app's data is about 16 MB. The whole repository pack is about 28 MB.

**Team decisions needed before public launch:**

1. **Precise coordinates in the repository.** `app/data/location_predictions.parquet` (62,138 points) and
   `app/data/historical_predictions.parquet` (85,678 points with `y_true` report labels) store exact latitude and
   longitude. So do the research CSVs already tracked in `data/processed/`. Report-associated points come from City of
   Houston 311 flooding records.
   - The app itself shows and exports only cell-level summaries. Its downloads contain cell centres and bounds, not
     observation coordinates.
   - But anyone who can see the GitHub repository can download the point files.
   - Community Cloud deploys from GitHub. Confirm whether the repository is or will be public, and whether publishing
     point-level coordinates is approved. If not, deploy from a private repository, or prepare coarsened app files in a
     separate, approved step. The validated datasets must not be edited in place.
2. **Basemap tiles.** The OpenStreetMap Foundation tile policy allows public apps, but it is best-effort and access can
   be blocked for heavy use. It requires visible attribution (shown by Leaflet), forbids prefetching, and needs the
   browser's Referer header (sent by default).
   - Fine for a modest class-project audience.
   - Decide whether to keep OSM or move to a provider with explicit terms if traffic may be significant.
3. **Nominatim contact details.** The policy asks for an identifying User-Agent. The default
   (`FloodSense-research-app/1.0`) contains no personal data.
   - Add a team-approved contact URL or email through the `FLOODSENSE_GEOCODER_USER_AGENT` secret before launch.
   - Requests are limited to one every 1.1 s per app process, which is application-wide on a single Community Cloud
     instance.
   - Alternatively set `FLOODSENSE_GEOCODER_ENABLED=false`; map and coordinate search still work.
4. **Licensing and attribution.**
   - The repository has no LICENSE file.
   - Confirm the mascot image's creator and rights.
   - Historical photographs stay unused until their licences are verified.
   - OpenStreetMap-derived data (`paved_pct`, place search and tiles) is credited in the app footer, the map and the
     data-source table.
   - Confirm the City of Houston 311 data terms.
5. **Deployment account.** Pushing the branch and authorizing Streamlit Community Cloud need a GitHub account with
   access to the repository. `floodsense.streamlit.app` availability is unknown until you try.

## Deploy to Streamlit Community Cloud (not yet done — needs authorization)

1. Push the `feature/app-development` branch to GitHub. This needs repository write access, and nothing has been committed or pushed yet.
2. Sign in at https://share.streamlit.io with a GitHub account that can access the repository and authorize the Streamlit GitHub app.
3. **Create app** → choose the repository, branch `feature/app-development`, main file path `app/streamlit_app.py`.
4. **Advanced settings:** Python **3.11**. Optional secrets: `FLOODSENSE_GEOCODER_USER_AGENT = "FloodSense/1.0 (+<contact URL>)"`.
5. Custom subdomain: request `floodsense`. Availability is unknown until you try.
6. Deploy. Community Cloud uses `app/requirements.txt` because it sits next to the entry point, so the research
   `requirements.txt` (PyTorch, LightGBM, SHAP) is not installed. `.streamlit/config.toml` at the repository root applies the theme.

## Resuming development

Read this file's checklist, then run `python -m pytest app/tests -q` from the repository root. Remaining work:
manual browser and mobile checks, team approval for any point-level historical display, then commit, push and deploy
once authorized. Do not rerun `scripts/prepare_app_data.py` or `notebooks/11_final_test.ipynb`.
