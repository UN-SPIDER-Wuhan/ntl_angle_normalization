# NTL Angle Normalization

Published paper: https://doi.org/10.1016/j.jag.2023.103359

NTL Angle Normalization is a VIIRS nighttime light time-series processing project that covers the full workflow from raw data preprocessing to angle normalization, Prophet-based fitting and gap filling, and conversion of text outputs back into daily GeoTIFF images.

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

- Python 3.11 (recommended and tested)
- Python 3.10 (compatible)
- Windows 10/11
- Conda for environment management

## 4. Installation

### Option 1. Create a Conda Environment

```bash
conda create -n ntl_prophet311 python=3.11 -y
conda activate ntl_prophet311
pip install -r requirements.txt
```

If GDAL or Rasterio fails to install, try this first:

```bash
conda install -c conda-forge gdal rasterio -y
pip install -r requirements.txt
```

For Windows users, if Prophet reports encoding errors during CmdStan runs,
set an ASCII temp directory before running notebooks or scripts:

```powershell
New-Item -ItemType Directory -Force -Path D:\tmp\cmdstan | Out-Null
$env:TMP='D:\tmp\cmdstan'
$env:TEMP='D:\tmp\cmdstan'
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

## 6. Usage Notes

### 6.1 Template Raster Recommendation

If you already have any clipped reference TIFF for your study area, it is strongly recommended to use template mode for text-to-image conversion.

Only use template-free mode when you clearly know:

- the raster width and height
- the upper-left coordinate
- the pixel resolution
- the CRS

### 6.2 Parallel Execution

Several stages support parallel execution, but using more workers is not always better.

Suggested settings:

- small datasets: 4 to 8 workers
- medium datasets: 8 to 16 workers
- on Windows, reduce the worker count if you encounter resource contention or file-handle issues
