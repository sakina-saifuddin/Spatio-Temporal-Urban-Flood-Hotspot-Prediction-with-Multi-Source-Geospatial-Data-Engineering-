# Raw data

Keep downloaded, unmodified source datasets here, grouped by provider. Do not overwrite raw sources with cleaned or derived data.

Expected sources:
- `noaa/` — NOAA coastal DEM (large raster excluded from Git; see root `.gitignore` and the root README).
- `usgs/` — National Hydrography Dataset streams/bayous.
- `mrlc/` — NLCD 2019 raster.
- `noaa_rainfall/` — GHCN-Daily station observations and download metadata.
- `city_of_houston/` — original annual flooding/drainage CSVs.

For every download, record source URL, access date, license/terms, checksum, coverage, CRS, and any changes. Large or restricted source files should not be committed; provide reproducible download instructions instead.
