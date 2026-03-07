from .prophet_pipeline import run_prophet_pipeline
from .text_to_img import txt_to_daily_geotiffs
from .angle_normalization import (
    readFile,
    calGoodnessOfFit,
    calCorrelation,
    normalizationZenith,
    visScatterAndFitCurve,
    visTimeSeries,
    run_angle_normalization
)

# Optional preprocessing imports, if available.
try:
    from .preprocessing import (
        stage1_extract_and_pair,
        stage2_generate_time_series,
        clip_rasters_by_shapefile,
        mosaic_tiles_by_date,
        complete_ntl_preprocessing_pipeline,
        parse_h5_name,
        build_key
    )
    _preprocessing_available = True
except ImportError:
    _preprocessing_available = False

__all__ = [
    # Prophet-related exports
    "run_prophet_pipeline",
    "txt_to_daily_geotiffs",
    # Angle normalization exports
    "readFile",
    "calGoodnessOfFit",
    "calCorrelation", 
    "normalizationZenith",
    "visScatterAndFitCurve",
    "visTimeSeries",
    "run_angle_normalization",
]

# Add preprocessing symbols to __all__ when the module is available.
if _preprocessing_available:
    __all__.extend([
        "stage1_extract_and_pair",
        "stage2_generate_time_series",
        "clip_rasters_by_shapefile",
        "mosaic_tiles_by_date",
        "complete_ntl_preprocessing_pipeline",
        "parse_h5_name",
        "build_key"
    ])
