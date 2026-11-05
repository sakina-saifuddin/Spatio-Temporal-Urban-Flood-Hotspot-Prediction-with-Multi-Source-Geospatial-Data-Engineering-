# Marvin — individual workspace

Branch: `marvin`.

Use this folder for Marvin’s project notes, experiments, and work-in-progress that is not yet part of the shared pipeline. Put reviewed/reusable code in the shared `src/`, `notebooks/`, or `app/` directories through a pull request. Update this README with the agreed task/role; no assignment is assumed yet.

Do not duplicate shared datasets here. Read the data locations and handling rules in `data/README.md`.



# How to Run the Data Processing Pipeline

Follow these steps to execute the data processing pipeline for the **Spatio-Temporal Urban Flood Hotspot Prediction with Multi-Source Geospatial Data Engineering** project.

## 1. Navigate to the Project Directory

Open Terminal and navigate to the project folder:

```bash
cd Spatio-Temporal-Urban-Flood-Hotspot-Prediction-with-Multi-Source-Geospatial-Data-Engineering-
```

## 2. Create the `marvin` Folder

Create a folder named `marvin` inside the `data` directory.

## 3. Add the Input File

Locate `input.csv` in `raw/city_of_houston` and place it in the appropriate location required by the pipeline.

## 4. Calculate Distance to Streams

Navigate to the `distanceToStream` folder and run:

```bash
python getDistanceToSteam.py
```

## 5. Process Elevation Data

Navigate to the `elevation` folder and run the following scripts in order:

```bash
python getFiles.py
python mergeFiles.py
python getElevation.py
```

## 6. Process Imperviousness and Land-Type Data

Navigate to the `imperviousAndLandType` folder and run:

```bash
python getImperviousness.py
```

## 7. Calculate Paved Road Percentage

Navigate to the `pavedPct` folder and run:

```bash
python GetRRoad.py
```

## 8. Process Rainfall Data

Navigate to the `rainfall` folder and run:

```bash
python getRainFall.py
```

## 9. Merge the Processed Data

Run `merge.py` to combine the processed datasets and generate the final `output.csv` file:

```bash
python merge.py
```

## Expected Output

The pipeline generates `output.csv`, containing the merged data for subsequent spatio-temporal urban flood hotspot prediction.

**Note:** Ensure that all required input files, dependencies, and folder paths are correctly configured before running the scripts. Execute each script from the appropriate working directory.
