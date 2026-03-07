# NTL Prophet

Published paper: https://doi.org/10.1016/j.jag.2023.103359

NTL Prophet is a VIIRS nighttime light time-series processing project that covers the full workflow from raw data preprocessing to angle normalization, Prophet-based fitting and gap filling, and conversion of text outputs back into daily GeoTIFF images.

This repository is suitable for the following use cases:

- Processing VNP46A1 and VNP46A2 nighttime light products
- Performing quality control and angle normalization for pixel-level nighttime light time series
- Using Prophet to fit and fill missing or unstable observations
- Reconstructing final text outputs into GeoTIFF files that can be loaded in GIS software

## 1. Features

The project currently provides four core capabilities.

### 1. Data Preprocessing

- Extract key layers from VNP46A2 and VNP46A1 HDF5 files
- Support study-area clipping with a shapefile
- Support automatic mosaicking for multi-tile regions
- Generate standardized pixel-wise time-series text files

Entry points:

- notebook: [1_preprocessing_data.ipynb](1_preprocessing_data.ipynb)
- code: [functions/preprocessing.py](functions/preprocessing.py)

### 2. Angle Normalization

- Read preprocessed pixel time-series text files
- Perform sensor zenith angle normalization for each pixel
- Support outlier filtering, parallel execution, and visualization

Entry points:

- notebook: [2_angle normalization.ipynb](2_angle%20normalization.ipynb)
- code: [functions/angle_normalization.py](functions/angle_normalization.py)
- code: [functions/timeseries_analysis.py](functions/timeseries_analysis.py)

### 3. Prophet Fitting and Gap Filling

- Fit Prophet models on normalized time series
- Support PSO-based search for key Prophet parameters
- Export fitted text outputs and accuracy reports

Entry points:

- notebook: [3_prophet_params_and_run.ipynb](3_prophet_params_and_run.ipynb)
- code: [functions/prophet_pipeline.py](functions/prophet_pipeline.py)

### 4. Text-to-Image Conversion

- Convert Prophet text outputs into daily GeoTIFF files
- Support inheriting spatial metadata from a template raster
- Also support a template-free mode when raster size, transform, and CRS are provided manually

Entry point:

- code: [functions/text_to_img.py](functions/text_to_img.py)

## 2. Project Structure

```text
ntl_prophet/
├── README.md
├── LICENSE
├── requirements.txt
├── __init__.py
├── 1_preprocessing_data.ipynb
├── 2_angle normalization.ipynb
├── 3_prophet_params_and_run.ipynb
├── datasets/
└── functions/
    ├── __init__.py
    ├── preprocessing.py
    ├── angle_normalization.py
    ├── prophet_pipeline.py
    ├── text_to_img.py
    └── timeseries_analysis.py
```

## 3. Recommended Environment

Recommended setup:

- Python 3.10 or 3.11
- Windows 10/11
- Conda for environment management

## 4. Installation

### Option 1. Create a Conda Environment

```bash
conda create -n ntl_prophet python=3.10 -y
conda activate ntl_prophet
pip install -r requirements.txt
```

If GDAL or Rasterio fails to install, try this first:

```bash
conda install -c conda-forge gdal rasterio -y
pip install -r requirements.txt
```

### Option 2. Use pip Directly

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 5. Workflow

The recommended workflow is to run the project in the following order.

### Step 1. Preprocess Raw HDF5 Data

Open [1_preprocessing_data.ipynb](1_preprocessing_data.ipynb) and configure:

- VNP46A2 data folder
- VNP46A1 data folder
- output folder
- shapefile path
- date range and tile filter settings

This step generates standardized time-series text files, for example:

- `processed/ntl_timeseries.txt`

### Step 2. Run Angle Normalization

Open [2_angle normalization.ipynb](2_angle%20normalization.ipynb) and configure:

- input text directory
- output directory
- dataset names to process
- number of workers
- 3-sigma outlier filtering parameters

This step produces:

- normalized time-series text files
- fitted parameters
- R² and related statistics
- optional visualizations

### Step 3. Run Prophet Fitting

Open [3_prophet_params_and_run.ipynb](3_prophet_params_and_run.ipynb) and configure:

- input text path
- output text path
- DOY range
- PSO settings
- Prophet settings
- parallelization settings

This step produces:

- Prophet-fitted text outputs
- accuracy summaries and metrics reports

### Step 4. Convert Text to Daily GeoTIFF

The last code cell in [3_prophet_params_and_run.ipynb](3_prophet_params_and_run.ipynb) already includes a text-to-image conversion example.

The output consists of:

- one TIFF file per day
- filenames in `YYYYMMDD.tif` format

## 6. Input and Output Formats

### 6.1 Time-Series Text Format

The project typically uses the following pixel time-series text format:

```text
point1:lng,lat(lefttop):YYYYMMDD,Zenith,NTLValue;YYYYMMDD,Zenith,NTLValue;...
point2:lng,lat(lefttop):YYYYMMDD,Zenith,NTLValue;...
```

Where:

- `point1`, `point2` are pixel identifiers
- `lng,lat(lefttop)` is the upper-left coordinate of the pixel
- each record stores date, zenith angle, and nighttime light value

### 6.2 GeoTIFF Output

When converting text back to rasters, the program fills values into the 2D grid according to pixel ID order and writes one single-band GeoTIFF per day.

Using a template raster is recommended because it preserves:

- raster width and height
- affine transform
- CRS
- consistent spatial alignment

Template-free mode is also supported, but you must provide:

- `width`
- `height`
- `transform`
- `crs`

## 7. Core Interfaces

### 7.1 Preprocessing

Main functions in [functions/preprocessing.py](functions/preprocessing.py):

- `complete_ntl_preprocessing_pipeline`
- `stage1_extract_and_pair`
- `stage2_generate_time_series`
- `clip_rasters_by_shapefile`
- `mosaic_tiles_by_date`

### 7.2 Angle Normalization

Main functions in [functions/angle_normalization.py](functions/angle_normalization.py):

- `run_angle_normalization`
- `readFile`
- `visScatterAndFitCurve`
- `visTimeSeries`

### 7.3 Prophet Fitting

Main functions in [functions/prophet_pipeline.py](functions/prophet_pipeline.py):

- `run_prophet_pipeline`

### 7.4 Text-to-Image Conversion

Main function in [functions/text_to_img.py](functions/text_to_img.py):

- `txt_to_daily_geotiffs`

Example:

```python
from functions.text_to_img import txt_to_daily_geotiffs

tif_files = txt_to_daily_geotiffs(
    txt_path=r".\output\ntl_timeseries_angle_prophet_pso.txt",
    output_dir=r".\output\prophet_daily_tif",
    start_date="20180101",
    end_date="20190101",
    template_tif=r".\template.tif",
)
```

## 8. Usage Notes

### 8.1 Template Raster Recommendation

If you already have any clipped reference TIFF for your study area, it is strongly recommended to use template mode for text-to-image conversion.

Only use template-free mode when you clearly know:

- the raster width and height
- the upper-left coordinate
- the pixel resolution
- the CRS

### 8.2 Parallel Execution

Several stages support parallel execution, but using more workers is not always better.

Suggested settings:

- small datasets: 4 to 8 workers
- medium datasets: 8 to 16 workers
- on Windows, reduce the worker count if you encounter resource contention or file-handle issues

## 9. Publication Notes

The project name is kept as `ntl_prophet` to stay consistent with import paths, notebook path inference, and GitHub repository naming.

Before publishing updates, it is recommended to confirm:

- the notebooks still use relative paths
- large raw datasets in `datasets/` are not committed unintentionally
- generated outputs remain ignored unless you explicitly want to version them
- screenshots, study-area descriptions, and data-source notes are added if the repository will be public-facing