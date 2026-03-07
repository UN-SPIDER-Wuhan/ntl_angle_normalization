# -*- coding: utf-8 -*-
"""
Nighttime light preprocessing module.

This module provides a full preprocessing workflow for VNP46A1/A2 data, including:
- Stage 1: HDF5 layer extraction and pairing, with optional clipping during extraction
- Stage 2: Time-series text generation, with automatic mosaicking for multi-tile regions
- Optional: shapefile clipping and tile mosaicking

Example:
    from functions.preprocessing import (
        stage1_extract_and_pair,
        stage2_generate_time_series,
        clip_rasters_by_shapefile,
        mosaic_tiles_by_date,
        complete_ntl_preprocessing_pipeline
    )

    # Run by stage (multi-tile study areas are mosaicked automatically)
    stage1_info = stage1_extract_and_pair(a2_folder, a1_folder, output_folder,
                                          clip_shapefile=shp_path)
    results = stage2_generate_time_series(stage1_info, output_file)

    # Or run the full workflow in one call
    results = complete_ntl_preprocessing_pipeline(a2_folder, a1_folder, output_folder, output_file)
"""

import os
import re
import sys
import time
import traceback
import shutil
import numpy as np
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Delay GDAL import so the module can still load when GDAL is unavailable
try:
    from osgeo import gdal, ogr, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("⚠️ GDAL is not installed; some features are unavailable")


# =============================================================================
# Constants and regex patterns
# =============================================================================

VIIRS_NAME_RE = re.compile(
    r'^(?P<product>VNP46A[12])\.A(?P<year>\d{4})(?P<doy>\d{3})'
    r'\.h(?P<h>\d{2})v(?P<v>\d{2})\.(?P<collection>\d{3})'
    r'\.(?P<production>\d{13})\.h5$'
)


# =============================================================================
# Helper functions
# =============================================================================

def parse_h5_name(fname: str) -> dict:
    """
    Parse a VNP46A1/A2 HDF5 file name.

    Returns:
        dict: {product, year, doy, h, v, collection, production} or None
    """
    m = VIIRS_NAME_RE.match(fname)
    if not m:
        return None
    g = m.groupdict()
    return {
        'product': g['product'],
        'year': int(g['year']),
        'doy': int(g['doy']),
        'h': int(g['h']),
        'v': int(g['v']),
        'collection': int(g['collection']),
        'production': g['production']
    }


def build_key(meta: dict) -> tuple:
    """Build a unique key from metadata: (year, doy, h, v, collection)"""
    return (meta['year'], meta['doy'], meta['h'], meta['v'], meta['collection'])


def _safe_rmtree(path: str, retries: int = 5, delay: float = 0.6) -> tuple:
    """Delete a directory more robustly to reduce issues caused by transient file locks on Windows."""
    import stat

    def _onerror(func, p, exc_info):
        # If a file is read-only, add write permission first and retry once.
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    last_err = None
    for i in range(max(1, retries)):
        try:
            if not os.path.exists(path):
                return True, None
            shutil.rmtree(path, onerror=_onerror)
            return True, None
        except Exception as e:
            last_err = e
            # Do not wait after the final failed attempt
            if i < retries - 1:
                time.sleep(delay)

    return False, last_err


def _detect_crs(shp_path: str, sample_raster: str):
    """
    Check whether the shapefile CRS matches the sample raster CRS.

    Returns:
        (vec_wkt, ras_wkt, is_same, message)
    """
    vec_ds = ogr.Open(shp_path)
    if vec_ds is None:
        return None, None, False, "Unable to open vector file"
    
    lyr = vec_ds.GetLayer(0)
    vec_srs = lyr.GetSpatialRef()
    vec_wkt = vec_srs.ExportToWkt() if vec_srs else None
    
    ras_ds = gdal.Open(sample_raster, gdal.GA_ReadOnly)
    if ras_ds is None:
        return vec_wkt, None, False, "Unable to open sample raster"
    
    ras_wkt = ras_ds.GetProjection()
    
    if not vec_wkt or not ras_wkt:
        return vec_wkt, ras_wkt, False, "Incomplete CRS information"
    
    vec_srs_norm = osr.SpatialReference()
    ras_srs_norm = osr.SpatialReference()
    vec_srs_norm.ImportFromWkt(vec_wkt)
    ras_srs_norm.ImportFromWkt(ras_wkt)
    
    same = bool(vec_srs_norm.IsSame(ras_srs_norm))
    return vec_wkt, ras_wkt, same, "OK" if same else "CRS mismatch"


def _warp_single(args):
    """Single-file clipping task for parallel execution"""
    (in_path, out_path, shp_path, warp_options, retries) = args
    from osgeo import gdal
    
    last_err = None
    for attempt in range(retries + 1):
        try:
            gdal.UseExceptions()
            gdal.Warp(out_path, in_path, cutlineDSName=shp_path, 
                     cropToCutline=True, **warp_options)
            return True, in_path, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    
    return False, in_path, last_err


# =============================================================================
# Shapefile clipping
# =============================================================================

def clip_rasters_by_shapefile(
        folders: dict,
        shp_path: str,
        output_base: str,
        processes: int = 4,
        overwrite: bool = False,
        warp_options: dict = None,
        parallel_mode: str = 'auto',
        chunk_size: int = 32,
        retries: int = 0,
        fail_fallback_serial: bool = True,
    verbose: bool = True,
    use_bbox: bool = False) -> dict:
    """
    Clip all GeoTIFF files generated by Stage 1 in parallel.

    Parameters:
        folders: folder mapping returned by Stage 1 {'ntl':..., 'quality':..., 'zenith':...}
        shp_path: clipping shapefile path
        output_base: output directory for clipped results
        processes: number of parallel worker processes/threads
        overwrite: whether to overwrite existing files
        warp_options: extra parameters passed to gdal.Warp
        parallel_mode: 'auto'|'process'|'thread'|'none'
        chunk_size: number of tasks submitted per batch
        retries: retry count for each failed file
        fail_fallback_serial: whether to fall back to serial execution after parallel failure
        verbose: whether to print verbose logs

    Returns:
        folders_clipped: mapping of clipped output directories
    """
    if warp_options is None:
        warp_options = {}
    
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Clipping shapefile does not exist: {shp_path}")
    
    os.makedirs(output_base, exist_ok=True)
    
    # Automatically choose the parallel mode
    if parallel_mode == 'auto':
        chosen_mode = 'thread' if sys.platform.startswith('win') else 'process'
    else:
        chosen_mode = parallel_mode
    
    # CRS check
    sample_raster = None
    for key in ('ntl', 'quality', 'zenith'):
        d = folders.get(key)
        if d and os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith('.tif'):
                    sample_raster = os.path.join(d, f)
                    break
        if sample_raster:
            break
    
    if sample_raster and verbose:
        vec_wkt, ras_wkt, same, msg = _detect_crs(shp_path, sample_raster)
        if same:
            print("🧭 CRS check: ✅ matched")
        else:
            print(f"🧭 CRS warning: ❌ mismatch - {msg}")
            print("  Reproject the vector file first or specify dstSRS in warp_options")

    # If bounding-box clipping is requested, try to compute the shapefile extent in the raster CRS
    shp_bbox = None
    if use_bbox:
        try:
            # Get the shapefile bounding box and try to transform it into the sample raster CRS
            def _get_shapefile_bbox(shp_p, target_raster=None):
                ds = ogr.Open(shp_p)
                if ds is None:
                    return None
                lyr = ds.GetLayer(0)
                env = lyr.GetExtent()  # returns (minX, maxX, minY, maxY)
                minx, maxx, miny, maxy = env[0], env[1], env[2], env[3]

                if target_raster:
                    rast = gdal.Open(target_raster)
                    if rast:
                        ras_wkt = rast.GetProjection()
                        vec_srs = lyr.GetSpatialRef()
                        if vec_srs is None:
                            return (minx, miny, maxx, maxy)
                        vec_srs_t = osr.SpatialReference()
                        vec_srs_t.ImportFromWkt(vec_srs.ExportToWkt())
                        ras_srs = osr.SpatialReference()
                        ras_srs.ImportFromWkt(ras_wkt)
                        if not vec_srs_t.IsSame(ras_srs):
                            # Transform the four corner points of the bounding box
                            transform = osr.CoordinateTransformation(vec_srs_t, ras_srs)
                            pts = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
                            xs = []
                            ys = []
                            for x, y in pts:
                                nx, ny, _ = transform.TransformPoint(x, y)
                                xs.append(nx); ys.append(ny)
                            return (min(xs), min(ys), max(xs), max(ys))
                        else:
                            return (minx, miny, maxx, maxy)
                return (minx, miny, maxx, maxy)

            shp_bbox = _get_shapefile_bbox(shp_path, sample_raster)
            if shp_bbox and verbose:
                print(f"🗺️ Using bounding-box clipping: bbox={shp_bbox}")
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to compute bounding box; falling back to cutline clipping: {e}")
            shp_bbox = None
    
    # Collect tasks
    tasks_all = []
    folders_clipped = {}
    
    for key, in_dir in folders.items():
        if not os.path.exists(in_dir):
            continue
        out_dir = os.path.join(output_base, key)
        os.makedirs(out_dir, exist_ok=True)
        
        for fname in os.listdir(in_dir):
            if not fname.lower().endswith('.tif'):
                continue
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            
            if (not overwrite) and os.path.exists(out_path):
                continue
            
            tasks_all.append((in_path, out_path, shp_path, warp_options, retries))
        folders_clipped[key] = out_dir
    
    if not tasks_all:
        if verbose:
            print("⚠️ No clipping tasks found (results may already exist or no TIFF files were found)")
        return folders_clipped
    
    if verbose:
        print(f"✂️ Ready to clip {len(tasks_all)} files, mode={chosen_mode}, "
              f"max_workers={processes}, chunk_size={chunk_size}")
    
    # Progress bar
    try:
        from tqdm import tqdm
        pbar_total = tqdm(total=len(tasks_all), desc="Clipping progress", unit="file")
        use_tqdm = True
    except ImportError:
        pbar_total = None
        use_tqdm = False
    
    def run_batch(batch_tasks, mode):
        success_paths = []
        fail_entries = []
        
        if mode == 'none':
            for t in batch_tasks:
                # If a bounding box is specified, inject outputBounds into warp_options for each task
                if use_bbox and shp_bbox is not None:
                    in_path, out_path, s_path, w_opts, r = t
                    w_opts = dict(w_opts) if w_opts else {}
                    w_opts.update({'outputBounds': (shp_bbox[0], shp_bbox[1], shp_bbox[2], shp_bbox[3])})
                    new_t = (in_path, out_path, None, w_opts, r)
                    ok, p, err = _warp_single(new_t)
                else:
                    ok, p, err = _warp_single(t)
                if ok:
                    success_paths.append(p)
                else:
                    fail_entries.append((p, err))
                if use_tqdm:
                    pbar_total.update(1)
            return success_paths, fail_entries
        
        ExecutorCls = ThreadPoolExecutor if mode == 'thread' else ProcessPoolExecutor
        with ExecutorCls(max_workers=processes) as exe:
            # If bounding-box clipping is used, replace per-task cutline arguments with outputBounds
            tasks_for_submit = []
            for t in batch_tasks:
                if use_bbox and shp_bbox is not None:
                    in_path, out_path, s_path, w_opts, r = t
                    w_opts = dict(w_opts) if w_opts else {}
                    w_opts.update({'outputBounds': (shp_bbox[0], shp_bbox[1], shp_bbox[2], shp_bbox[3])})
                    tasks_for_submit.append((in_path, out_path, None, w_opts, r))
                else:
                    tasks_for_submit.append(t)

            futures = {exe.submit(_warp_single, t): t for t in tasks_for_submit}
            for fut in as_completed(futures):
                try:
                    ok, p, err = fut.result()
                except Exception as e:
                    ok = False
                    p = futures[fut][0]
                    err = f"Runtime exception: {type(e).__name__}: {e}"
                
                if ok:
                    success_paths.append(p)
                else:
                    fail_entries.append((p, err))
                
                if use_tqdm:
                    pbar_total.update(1)
        
        return success_paths, fail_entries
    
    # Execute clipping
    all_success = set()
    failure_detail = {}
    remaining = list(tasks_all)
    
    mode_sequence = [chosen_mode]
    if chosen_mode == 'process':
        mode_sequence.append('thread')
        if fail_fallback_serial:
            mode_sequence.append('none')
    elif chosen_mode == 'thread' and fail_fallback_serial:
        mode_sequence.append('none')
    
    for mode in mode_sequence:
        if not remaining:
            break
        
        if verbose:
            print(f"🚀 Starting mode: {mode} ({len(remaining)} files remaining)")
        
        new_remaining = []
        try:
            while remaining:
                batch = remaining[:chunk_size]
                remaining = remaining[chunk_size:]
                
                try:
                    succ, fails = run_batch(batch, mode)
                    for p in succ:
                        all_success.add(p)
                    for p, err in fails:
                        failure_detail[p] = err
                except Exception as e:
                    if verbose:
                        print(f"  ❌ Fatal batch error in mode {mode}: {type(e).__name__}: {e}")
                    if mode == 'process':
                        new_remaining = [t for t in tasks_all if t[0] not in all_success]
                        break
                    else:
                        raise
            
            if new_remaining:
                remaining = new_remaining
            else:
                remaining = [t for t in tasks_all 
                           if t[0] in failure_detail and t[0] not in all_success]
            
            if remaining and verbose:
                print(f"  ⚠️ {len(remaining)} files still failed after mode {mode}; preparing fallback...")
                
        except Exception as fatal:
            if verbose:
                print(f"💥 Fatal error in mode {mode}: {fatal}")
            if mode == 'process':
                remaining = [t for t in tasks_all if t[0] not in all_success]
                continue
            elif mode == 'thread' and fail_fallback_serial:
                remaining = [t for t in tasks_all if t[0] not in all_success]
                continue
            else:
                break
    
    if pbar_total:
        pbar_total.close()
    
    # Serial fallback
    if remaining and ('none' not in mode_sequence):
        if verbose:
            print(f"🛠️ Running serial fallback for {len(remaining)} files...")
        for t in remaining:
            ok, p, err = _warp_single(t)
            if ok:
                all_success.add(p)
            else:
                failure_detail[p] = err
    
    # Summary statistics
    total = len(tasks_all)
    done = len(all_success)
    final_fail_paths = [t[0] for t in tasks_all if t[0] not in all_success]
    
    if verbose:
        print(f"✅ Clipping finished: {done} succeeded / {total} total, {len(final_fail_paths)} failed")
        
        if final_fail_paths:
            print("Example failed files (first 10):")
            for p in final_fail_paths[:10]:
                print("  -", p, '=>', failure_detail.get(p, 'Unknown error'))
            print("Possible causes: CRS mismatch, invalid shapefile geometry, GDAL driver issues, or insufficient memory")
    
    return folders_clipped


# =============================================================================
# Stage 1: Extract layers and pair files
# =============================================================================

def _get_shapefile_bbox(shp_path: str, target_srs_wkt: str = None):
    """
    Get the shapefile bounding box, optionally transformed to a target CRS.

    Returns:
        (minx, miny, maxx, maxy) or None
    """
    try:
        ds = ogr.Open(shp_path)
        if ds is None:
            return None
        lyr = ds.GetLayer(0)
        env = lyr.GetExtent()  # Returns (minX, maxX, minY, maxY)
        minx, maxx, miny, maxy = env[0], env[1], env[2], env[3]
        
        vec_srs = lyr.GetSpatialRef()
        if vec_srs is None or target_srs_wkt is None:
            return (minx, miny, maxx, maxy)
        
        # Transform into the target coordinate system
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(target_srs_wkt)
        
        if not vec_srs.IsSame(target_srs):
            # Set axis order to the traditional (x, y), i.e. (lon, lat)
            vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
            transform = osr.CoordinateTransformation(vec_srs, target_srs)
            # Note: pts must use the correct (minx, miny) corner combinations
            pts = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
            xs, ys = [], []
            for x, y in pts:
                # TransformPoint returns (x, y, z)
                nx, ny, _ = transform.TransformPoint(x, y)
                xs.append(nx)
                ys.append(ny)
            return (min(xs), min(ys), max(xs), max(ys))
        
        return (minx, miny, maxx, maxy)
    except Exception as e:
        print(f"⚠️ Failed to get shapefile bbox: {e}")
        return None


def _extract_layer_with_clip(subhdf: str, out_path: str, shp_path: str = None, 
                              warp_options: dict = None, use_bbox: bool = False) -> bool:
    """
    Extract a single layer, optionally clipping directly to the shapefile extent.

    Parameters:
        subhdf: HDF5 subdataset path
        out_path: output TIFF path
        shp_path: optional clipping shapefile path
        warp_options: extra gdal.Warp parameters
        use_bbox: whether to clip by bounding box only while preserving all data inside the bbox

    Returns:
        bool: whether extraction succeeded
    """
    try:
        r_layer = gdal.Open(subhdf, gdal.GA_ReadOnly)
        if r_layer is None:
            return False
        
        # Retrieve geospatial metadata
        md = r_layer.GetMetadata_Dict()
        if "HorizontalTileNumber" in md and "VerticalTileNumber" in md:
            hnum = int(md["HorizontalTileNumber"])
            vnum = int(md["VerticalTileNumber"])
            west = 10 * hnum - 180
            north = 90 - 10 * vnum
            east = west + 10
            south = north - 10
            
            if shp_path and os.path.exists(shp_path):
                # First assign georeferencing to the in-memory VRT
                temp_vrt = f"/vsimem/temp_{os.path.basename(out_path)}.vrt"
                opt_txt = f"-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}"
                trans_opts = gdal.TranslateOptions(gdal.ParseCommandLine(opt_txt))
                gdal.Translate(temp_vrt, r_layer, options=trans_opts)
                
                if use_bbox:
                    # Clip using the bounding box only (keep all data inside the bbox)
                    target_srs = osr.SpatialReference()
                    target_srs.ImportFromEPSG(4326)
                    bbox = _get_shapefile_bbox(shp_path, target_srs.ExportToWkt())
                    
                    if bbox:
                        # bbox = (minx, miny, maxx, maxy)
                        # Check whether the bounding box intersects the tile
                        # tile_bbox also uses the (minx, miny, maxx, maxy) format
                        tile_bbox = (west, south, east, north)
                        
                        # Compute the intersection
                        intersect_minx = max(bbox[0], tile_bbox[0])
                        intersect_miny = max(bbox[1], tile_bbox[1])
                        intersect_maxx = min(bbox[2], tile_bbox[2])
                        intersect_maxy = min(bbox[3], tile_bbox[3])
                        
                        if intersect_minx < intersect_maxx and intersect_miny < intersect_maxy:
                            # If there is an intersection, use the intersected extent
                            # GDAL outputBounds format: (ulx, uly, lrx, lry) = (minX, maxY, maxX, minY)
                            warp_opts = {
                                'outputBounds': (intersect_minx, intersect_maxy, intersect_maxx, intersect_miny),
                                'dstSRS': 'EPSG:4326',
                            }
                            if warp_options:
                                warp_opts.update(warp_options)
                            # Remove potentially conflicting cutline arguments
                            warp_opts.pop('cutlineDSName', None)
                            warp_opts.pop('cropToCutline', None)
                            
                            gdal.Warp(out_path, temp_vrt, **warp_opts)
                        else:
                            # No intersection found, clean up and return failure
                            gdal.Unlink(temp_vrt)
                            return False
                    else:
                        # Bounding-box computation failed, fall back to full extraction
                        gdal.Translate(out_path, temp_vrt)
                else:
                    # Clip by shapefile boundary and set pixels outside the boundary to NoData
                    warp_opts = {
                        'cutlineDSName': shp_path,
                        'cropToCutline': True,
                        'dstSRS': 'EPSG:4326',
                    }
                    if warp_options:
                        warp_opts.update(warp_options)
                    
                    gdal.Warp(out_path, temp_vrt, **warp_opts)
                
                # Clean up the in-memory VRT
                gdal.Unlink(temp_vrt)
            else:
                # Extract only, without clipping
                opt_txt = f"-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}"
                opts = gdal.TranslateOptions(gdal.ParseCommandLine(opt_txt))
                gdal.Translate(out_path, r_layer, options=opts)
        else:
            if shp_path and os.path.exists(shp_path):
                if use_bbox:
                    bbox = _get_shapefile_bbox(shp_path)
                    if bbox:
                        # bbox = (minx, miny, maxx, maxy)
                        # GDAL outputBounds format: (ulx, uly, lrx, lry) = (minX, maxY, maxX, minY)
                        warp_opts = {'outputBounds': (bbox[0], bbox[3], bbox[2], bbox[1])}
                        if warp_options:
                            warp_opts.update(warp_options)
                        warp_opts.pop('cutlineDSName', None)
                        warp_opts.pop('cropToCutline', None)
                        gdal.Warp(out_path, r_layer, **warp_opts)
                    else:
                        gdal.Translate(out_path, r_layer)
                else:
                    warp_opts = {'cutlineDSName': shp_path, 'cropToCutline': True}
                    if warp_options:
                        warp_opts.update(warp_options)
                    gdal.Warp(out_path, r_layer, **warp_opts)
            else:
                gdal.Translate(out_path, r_layer)
        
        return True
    except Exception as e:
        print(f"⚠️ Layer extraction failed: {e}")
        return False


def stage1_extract_and_pair(
        input_vnp46a2_folder: str,
        input_vnp46a1_folder: str,
        output_base_folder: str,
        zenith_fill_mode: str = "zero",
        clip_shapefile: str = None,
        clip_warp_options: dict = None,
        clip_shapefile_bbox: bool = False,
        auto_mosaic: bool = True,
        date_start: str = None,
        date_end: str = None,
        tile_filter: list = None,
        verbose: bool = True) -> dict:
    """
    Stage 1: read VNP46A2 / VNP46A1 HDF5 files, extract layers, and pair them.

    Output directory structure:
        <output_base_folder>/temp_processing/
            DNB_BRDF-Corrected_NTL/*.tif
            Mandatory_Quality_Flag/*.tif
            Sensor_Zenith/*.tif

    Parameters:
        input_vnp46a2_folder: directory containing VNP46A2 HDF5 files
        input_vnp46a1_folder: directory containing VNP46A1 HDF5 files
        output_base_folder: base output directory
        zenith_fill_mode: Zenith fill mode ('zero' or 'mean') used in Stage 2
        clip_shapefile: optional clipping shapefile path for direct extraction-time clipping
        clip_warp_options: clipping parameters
        clip_shapefile_bbox: whether to use bounding-box clipping
        auto_mosaic: whether to automatically mosaic cross-tile data
        date_start: start-date filter in 'YYYY-MM-DD' format, inclusive; None disables filtering
        date_end: end-date filter in 'YYYY-MM-DD' format, inclusive; None disables filtering
        tile_filter: tile whitelist in the form ['h27v05', 'h28v05'] or [(27,5), (28,5)]; None disables filtering
        verbose: whether to print verbose logs

    Returns:
        dict: contains temp_work_dir, folders, a2_meta, a1_dict, and summary statistics
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is not installed; layer extraction cannot run")
    
    start_time = time.time()
    
    # Create output directories
    temp_work_dir = os.path.join(output_base_folder, "temp_processing")
    os.makedirs(temp_work_dir, exist_ok=True)
    
    folders = {
        'ntl': os.path.join(temp_work_dir, "DNB_BRDF-Corrected_NTL"),
        'zenith': os.path.join(temp_work_dir, "Sensor_Zenith"),
        'quality': os.path.join(temp_work_dir, "Mandatory_Quality_Flag")
    }
    for p in folders.values():
        os.makedirs(p, exist_ok=True)
    
    clip_mode = "direct clipping" if clip_shapefile else "full extraction"
    if verbose:
        print(f"🚀 [Stage1] Starting extraction and pairing ({clip_mode})")
        if clip_shapefile:
            print(f"   Clipping vector: {clip_shapefile}")
            if clip_shapefile_bbox:
                print("   Clipping mode: bounding-box clipping (bbox)")

    # ---------- Parse filtering conditions ----------
    def _parse_date_to_year_doy(date_str):
        """Convert 'YYYY-MM-DD' to an integer pair of (year, doy)"""
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.year, int(dt.strftime("%j"))

    filter_start_yd = _parse_date_to_year_doy(date_start) if date_start else None
    filter_end_yd   = _parse_date_to_year_doy(date_end)   if date_end   else None

    def _parse_tile_filter(raw):
        """Normalize tile_filter into a set of integer (h, v) pairs"""
        if not raw:
            return None
        # Support passing a single string directly, such as "h27v05"
        if isinstance(raw, str):
            raw = [raw]
        result = set()
        for t in raw:
            if isinstance(t, str):
                m = re.match(r'h(\d+)v(\d+)', t, re.IGNORECASE)
                if m:
                    result.add((int(m.group(1)), int(m.group(2))))
            elif isinstance(t, (tuple, list)) and len(t) == 2:
                result.add((int(t[0]), int(t[1])))
        return result if result else None

    tile_set = _parse_tile_filter(tile_filter)

    def _meta_in_range(meta):
        """Check whether a metadata record satisfies the date/tile filters"""
        yd = (meta['year'], meta['doy'])
        if filter_start_yd and yd < filter_start_yd:
            return False
        if filter_end_yd   and yd > filter_end_yd:
            return False
        if tile_set and (meta['h'], meta['v']) not in tile_set:
            return False
        return True

    if verbose:
        if filter_start_yd or filter_end_yd:
            print(f"   Date filter: {date_start or 'unlimited'} ~ {date_end or 'unlimited'}")
        if tile_set:
            print(f"   Tile filter: {sorted(tile_set)}")
    # ---------- Finished parsing filtering conditions ----------

    # Save the current working directory; if it no longer exists, fall back to a safe directory
    try:
        original_dir = os.getcwd()
        if not os.path.exists(original_dir):
            original_dir = output_base_folder
    except Exception:
        original_dir = output_base_folder
    
    # Collect A2 files
    os.chdir(input_vnp46a2_folder)
    vnp46a2_files = [f for f in os.listdir('.') if f.endswith('.h5') and 'VNP46A2' in f]
    a2_meta = []
    for f in vnp46a2_files:
        meta = parse_h5_name(f)
        if meta:
            if _meta_in_range(meta):
                a2_meta.append((f, meta))
        elif verbose:
            print(f"  ⚠️ Skipping non-standard A2 filename: {f}")
    
    a2_dict = {build_key(meta): f for f, meta in a2_meta}
    
    # Collect A1 files
    a1_dict = {}
    if os.path.exists(input_vnp46a1_folder):
        os.chdir(input_vnp46a1_folder)
        vnp46a1_files = [f for f in os.listdir('.') if f.endswith('.h5') and 'VNP46A1' in f]
        for f in vnp46a1_files:
            meta = parse_h5_name(f)
            if meta:
                a1_dict[build_key(meta)] = f
            elif verbose:
                print(f"  ⚠️ Skipping non-standard A1 filename: {f}")
    else:
        if verbose:
            print("⚠️ VNP46A1 folder not found; only NTL and Quality layers will be produced")
    
    # Layer configuration
    vnp46a2_layer_configs = {
        'ntl': {'index': 0, 'output_folder': folders['ntl'], 
                'prefix': 'DNB_BRDF-Corrected_NTL_'},
        'quality': {'index': 4, 'output_folder': folders['quality'], 
                   'prefix': 'Mandatory_Quality_Flag_'}
    }
    
    total_files = len(a2_meta)
    processed_files = 0
    missing_zenith_keys = []
    failed_files = []
    
    # Progress bar
    try:
        from tqdm import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False
    
    # Process each A2 file
    file_iter = enumerate(a2_meta)
    if use_tqdm:
        file_iter = tqdm(list(file_iter), desc="Stage1 layer extraction", unit="file")
    
    for idx, (hdf_file, meta) in file_iter:
        try:
            os.chdir(input_vnp46a2_folder)
            hdf_layer = gdal.Open(hdf_file, gdal.GA_ReadOnly)
            
            if hdf_layer is None:
                failed_files.append((hdf_file, "Open failed"))
                continue
            
            subdatasets = hdf_layer.GetSubDatasets()
            raster_file_pre = hdf_file[:-17] if len(hdf_file) > 17 else hdf_file[:-3]
            
            layers_extracted = 0
            
            # Extract NTL and Quality layers (with optional clipping during extraction)
            for layer_name, cfg in vnp46a2_layer_configs.items():
                try:
                    if cfg['index'] < len(subdatasets):
                        subhdf = subdatasets[cfg['index']][0]
                        out_name = f"{cfg['prefix']}{raster_file_pre}.tif"
                        out_path = os.path.join(cfg['output_folder'], out_name)

                        w_opts = dict(clip_warp_options) if clip_warp_options else {}

                        if _extract_layer_with_clip(subhdf, out_path, clip_shapefile, w_opts, 
                                                     use_bbox=clip_shapefile_bbox):
                            layers_extracted += 1
                except Exception as e:
                    failed_files.append((hdf_file, f"{layer_name} extraction failed: {e}"))
            
            # Extract Zenith layers (with optional clipping during extraction)
            key = build_key(meta)
            
            if a1_dict:
                match_a1 = a1_dict.get(key)
                if match_a1:
                    try:
                        os.chdir(input_vnp46a1_folder)
                        a1_layer = gdal.Open(match_a1, gdal.GA_ReadOnly)
                        
                        if a1_layer is not None:
                            a1_subs = a1_layer.GetSubDatasets()
                            if 22 < len(a1_subs):
                                zenith_ds = a1_subs[22][0]
                                out_name = f"Sensor_Zenith_{raster_file_pre}.tif"
                                out_path = os.path.join(folders['zenith'], out_name)
                                
                                w_opts = dict(clip_warp_options) if clip_warp_options else {}
                                if _extract_layer_with_clip(zenith_ds, out_path, clip_shapefile, w_opts,
                                                             use_bbox=clip_shapefile_bbox):
                                    layers_extracted += 1
                    except Exception as e:
                        failed_files.append((hdf_file, f"Zenith processing error: {e}"))
                else:
                    missing_zenith_keys.append(key)
            
            # Check completion status
            expected = 3 if a1_dict else 2
            if layers_extracted == expected or (not a1_dict and layers_extracted == 2):
                processed_files += 1
                    
        except Exception as e:
            failed_files.append((hdf_file, str(e)))
    
    try:
        os.chdir(original_dir)
    except Exception:
        pass
    
    stage1_time = time.time() - start_time
    
    if verbose:
        print(f"\n✅ [Stage1] Completed: {processed_files}/{total_files} files, "
              f"missing Zenith pairs: {len(set(missing_zenith_keys))}, failures: {len(failed_files)}")
        print(f"   Elapsed time: {stage1_time:.2f} s")
        if failed_files and len(failed_files) <= 5:
            print("   Failed files:")
            for f, err in failed_files:
                print(f"     - {f}: {err}")
    
    # Automatically detect whether mosaicking is needed for multi-tile study areas
    if auto_mosaic:
        ntl_folder = folders.get('ntl', '')
        if os.path.exists(ntl_folder):
            date_groups = _group_tiffs_by_date(ntl_folder)
            need_mosaic = any(len(files) > 1 for files in date_groups.values())
            
            if need_mosaic:
                if verbose:
                    print("\n🔀 Multi-tile data detected; running mosaicking...")
                mosaic_base = os.path.join(temp_work_dir, "mosaic")
                folders = mosaic_tiles_by_date(folders, mosaic_base, verbose)
                stage1_time = time.time() - start_time
                if verbose:
                    print(f"   Elapsed time after mosaicking: {stage1_time:.2f} s")

    return {
        'temp_work_dir': temp_work_dir,
        'folders': folders,
        'a2_meta': a2_meta,
        'a1_dict': a1_dict,
        'missing_zenith_pairs': len(set(missing_zenith_keys)),
        'processed_files': processed_files,
        'total_files': total_files,
        'stage1_time': stage1_time,
        'failed_files': failed_files
    }


# =============================================================================
# Tile mosaicking for multi-tile study areas
# =============================================================================

def _group_tiffs_by_date(folder: str) -> dict:
    """
    Group TIFF files by date.

    Returns:
        {date_str: [file1, file2, ...]}
    """
    groups = {}
    for f in os.listdir(folder):
        if not f.endswith('.tif'):
            continue
        # Extract the date segment, e.g. VNP46A2.A2015001
        if 'VNP46A2.' in f:
            pos = f.find('VNP46A2.')
            date_part = f[pos:pos + 16]  # VNP46A2.A2015001
            if date_part not in groups:
                groups[date_part] = []
            groups[date_part].append(os.path.join(folder, f))
    return groups


def mosaic_tiles_by_date(
        folders: dict,
        output_base: str,
        verbose: bool = True) -> dict:
    """
    Mosaic multiple tiles from the same date into a single file for cross-tile study areas.

    Parameters:
        folders: folder mapping returned by Stage 1
        output_base: output directory for mosaic results
        verbose: whether to print verbose logs

    Returns:
        mosaic_folders: mapping of mosaic output directories
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is not installed; mosaicking cannot run")
    
    os.makedirs(output_base, exist_ok=True)
    
    mosaic_folders = {}
    
    for layer_name, layer_folder in folders.items():
        if not os.path.exists(layer_folder):
            continue
        
        out_folder = os.path.join(output_base, os.path.basename(layer_folder))
        os.makedirs(out_folder, exist_ok=True)
        mosaic_folders[layer_name] = out_folder
        
        # Group files by date
        date_groups = _group_tiffs_by_date(layer_folder)
        
        # Check whether mosaicking is required
        need_mosaic = any(len(files) > 1 for files in date_groups.values())
        
        if not need_mosaic:
            # No mosaicking required; copy or move directly
            if verbose:
                print(f"   {layer_name}: single tile detected; no mosaicking needed")
            import shutil
            for files in date_groups.values():
                for f in files:
                    shutil.copy2(f, out_folder)
            continue
        
        if verbose:
            print(f"   {layer_name}: multiple tiles detected; running mosaicking...")
        
        try:
            from tqdm import tqdm
            use_tqdm = verbose
        except ImportError:
            use_tqdm = False
        
        date_iter = date_groups.items()
        if use_tqdm:
            date_iter = tqdm(list(date_iter), desc=f"Mosaicking {layer_name}", unit="date")
        
        for date_part, files in date_iter:
            if len(files) == 1:
                # Single-tile case: copy directly
                import shutil
                shutil.copy2(files[0], out_folder)
            else:
                # Multi-tile case: mosaic
                # Extract the filename prefix for the output
                prefix = os.path.basename(files[0]).split('VNP46A2')[0]
                out_name = f"{prefix}{date_part}.tif"
                out_path = os.path.join(out_folder, out_name)
                
                try:
                    # Use gdal.Warp for mosaicking
                    gdal.Warp(out_path, files, format='GTiff')
                except Exception as e:
                    if verbose:
                        print(f"     ⚠️ Mosaicking failed for {date_part}: {e}")
    
    if verbose:
        print(f"   ✅ Mosaicking completed")
    
    return mosaic_folders


# =============================================================================
# Stage 2: Generate the time series
# =============================================================================

def stage2_generate_time_series(
        stage1_info: dict,
        final_output_file: str,
        zenith_fill_mode: str = "zero",
        verbose: bool = True) -> dict:
    """
    Stage 2: read TIFF layers generated by Stage 1 and build a per-pixel time-series text file.

    Parameters:
        stage1_info: info dictionary returned by Stage 1
        final_output_file: output path of the time-series text file
        zenith_fill_mode: Zenith fill mode ('zero' or 'mean')
        auto_mosaic: whether to auto-detect and mosaic multi-tile data (default True)
        verbose: whether to print verbose logs

    Returns:
        results: processing summary dictionary
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is not installed; time-series generation cannot run")
    
    folders = stage1_info['folders']
    total_files = stage1_info['total_files']
    processed_files = stage1_info['processed_files']
    missing_zenith_pairs = stage1_info['missing_zenith_pairs']
    
    if verbose:
        print("\n📊 [Stage2] Generating time series...")
    
    time_series_start = time.time()

    # Do not change cwd to avoid Windows file-handle issues that can block temporary directory cleanup.
    tiff_files = [f for f in os.listdir(folders['ntl']) if f.endswith('.tif')]
    file_num = len(tiff_files)
    
    if file_num == 0:
        raise Exception("Stage2: No NTL TIFF files found")
    
    # Get raster dimensions
    first_tiff = os.path.join(folders['ntl'], tiff_files[0])
    ntl_sample = gdal.Open(first_tiff)
    if ntl_sample is None:
        raise Exception(f"Stage2: Unable to open sample TIFF: {first_tiff}")
    m = ntl_sample.RasterYSize
    n = ntl_sample.RasterXSize
    ntl_sample = None
    
    if verbose:
        print(f"   Raster size: {m} x {n}, time steps: {file_num}")
    
    # Pre-allocate arrays
    point_ntl = -9999 * np.ones((file_num, m, n), dtype='float32')
    point_zenith = -9999 * np.ones((file_num, m, n), dtype='int32')
    point_lon = -9999 * np.ones((m, n), dtype='float32')
    point_lat = -9999 * np.ones((m, n), dtype='float32')
    point_date = np.zeros(shape=(1, file_num)).astype(np.str_)
    zenith_available_mask = np.zeros((file_num, m, n), dtype=bool)
    
    # Progress bar
    try:
        from tqdm import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False
    
    skipped_files = []
    failed_files = []
    
    # Read all time steps
    file_iter = enumerate(tiff_files)
    if use_tqdm:
        file_iter = tqdm(list(file_iter), desc="Stage2 reading TIFFs", unit="file")
    
    for i, tiff_file in file_iter:
        ntl_data = None
        quality_data = None
        zenith_data = None
        try:
            # Parse the date
            if 'VNP46A2.' in tiff_file:
                pos = tiff_file.find('VNP46A2.')
                part = tiff_file[pos + len('VNP46A2.'): pos + len('VNP46A2.') + 8]
                year = int(part[1:5])
                doy = int(part[5:8])
                offset = date(year, 1, 1) + timedelta(doy - 1)
                point_date[0, i] = offset.strftime('%Y%m%d')
            
            # Build file paths
            ntl_file = os.path.join(folders['ntl'], tiff_file)
            zenith_file = os.path.join(folders['zenith'], 
                                      tiff_file.replace('DNB_BRDF-Corrected_NTL_', 'Sensor_Zenith_'))
            quality_file = os.path.join(folders['quality'], 
                                       tiff_file.replace('DNB_BRDF-Corrected_NTL_', 'Mandatory_Quality_Flag_'))
            
            req = [ntl_file, quality_file]
            has_zenith = os.path.exists(zenith_file)
            if has_zenith:
                req.append(zenith_file)
            
            if not all(os.path.exists(f) for f in req):
                skipped_files.append(tiff_file)
                continue
            
            # Read raster data
            ntl_data = gdal.Open(ntl_file)
            quality_data = gdal.Open(quality_file)
            zenith_data = gdal.Open(zenith_file) if has_zenith else None
            
            geotrans = ntl_data.GetGeoTransform()
            ntl_array = ntl_data.ReadAsArray()
            quality_array = quality_data.ReadAsArray()
            zenith_array = zenith_data.ReadAsArray() if zenith_data is not None else None
            
            # Iterate over pixels
            for j in range(m):
                for k in range(n):
                    if i == 0:
                        point_lon[j][k] = geotrans[0] + k * geotrans[1]
                        point_lat[j][k] = geotrans[3] + j * geotrans[5]
                    
                    if (ntl_array[j][k] >= 0) and (ntl_array[j][k] < 5000) and (quality_array[j][k] != 255):
                        point_ntl[i][j][k] = ntl_array[j][k]
                        if zenith_array is not None:
                            point_zenith[i][j][k] = zenith_array[j][k]
                            zenith_available_mask[i, j, k] = True
                
        except Exception as e:
            failed_files.append((tiff_file, str(e)))
        finally:
            # Release GDAL handles promptly to avoid file-lock issues on Windows.
            ntl_data = None
            quality_data = None
            zenith_data = None
    
    # Zenith filling
    if zenith_fill_mode == 'mean':
        zenith_mean = np.zeros((m, n), dtype='float32')
        zenith_count = np.zeros((m, n), dtype='int32')
        
        for i in range(file_num):
            mask = zenith_available_mask[i]
            if mask.any():
                vals = point_zenith[i].astype('float32')
                zenith_mean[mask] += vals[mask]
                zenith_count[mask] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            zenith_mean = np.where(zenith_count > 0, zenith_mean / zenith_count, 0.0)
        
        fill_pixels = 0
        for i in range(file_num):
            need = (point_ntl[i] >= 0) & (point_zenith[i] < 0)
            if need.any():
                point_zenith[i][need] = (zenith_mean[need] * 100).astype('int32')
                fill_pixels += int(need.sum())
    else:
        need_zero = (point_ntl >= 0) & (point_zenith < 0)
        fill_pixels = int(need_zero.sum())
        point_zenith[need_zero] = 0
    
    # Write the output file
    output_dir = os.path.dirname(final_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Output progress bar
    total_pixels = m * n
    point_num = 1
    
    with open(final_output_file, 'w', encoding='utf-8') as f:
        f.write("pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...\n")
        
        pixel_iter = [(j, k) for j in range(m) for k in range(n)]
        if use_tqdm:
            pixel_iter = tqdm(pixel_iter, desc="Stage2 writing pixels", unit="px")
        
        for j, k in pixel_iter:
            valid_mask = point_ntl[:, j, k] >= 0
            if valid_mask.any():
                f.write(f"point {point_num}:{point_lon[j][k]:.6f},{point_lat[j][k]:.6f}:")
                
                series_idxs = np.where(valid_mask)[0]
                entries = []
                for idx_val in series_idxs:
                    zen = point_zenith[idx_val, j, k] / 100.0
                    val = point_ntl[idx_val, j, k]
                    entries.append(f"{point_date[0, idx_val]},{zen:.2f},{val:.1f}")
                
                f.write(";".join(entries) + "\n")
                point_num += 1
    
    time_series_end = time.time()
    stage2_time = time_series_end - time_series_start
    
    if verbose:
        print(f"\n✅ [Stage2] Completed: {point_num - 1} valid pixels, Zenith-filled: {fill_pixels}")
        print(f"   Skipped: {len(skipped_files)}, failed: {len(failed_files)}, elapsed: {stage2_time:.2f} s")
    
    results = {
        'processed_hdf5_files': processed_files,
        'total_hdf5_files': total_files,
        'processed_dates': int(np.sum([1 for i in range(file_num) if (point_ntl[i] >= 0).any()])),
        'valid_points': point_num - 1,
        'output_file': final_output_file,
        'zenith_fill_mode': zenith_fill_mode,
        'missing_zenith_pairs': missing_zenith_pairs,
        'stage2_time': stage2_time,
        'stage1_time': stage1_info['stage1_time'],
        'total_processing_time': stage1_info['stage1_time'] + stage2_time
    }
    
    return results


# =============================================================================
# One-call pipeline interface
# =============================================================================

def complete_ntl_preprocessing_pipeline(
        input_vnp46a2_folder: str,
        input_vnp46a1_folder: str,
        output_base_folder: str,
        final_output_file: str,
        cleanup_intermediate: bool = True,
        zenith_fill_mode: str = "zero",
        # Clipping-related parameters
        use_shape_clip: bool = False,
        shapefile_path: str = None,
        use_shapefile_bbox: bool = False,
        clip_processes: int = 4,
        clip_overwrite: bool = False,
        clip_warp_options: dict = None,
        clip_parallel_mode: str = 'auto',
        # Date and tile filtering parameters
        date_start: str = None,
        date_end: str = None,
        tile_filter: list = None,
        verbose: bool = True) -> dict:
    """
    Run the full preprocessing workflow in one call: Stage 1 -> optional clipping -> Stage 2.

    Parameters:
        input_vnp46a2_folder: directory containing VNP46A2 HDF5 files
        input_vnp46a1_folder: directory containing VNP46A1 HDF5 files
        output_base_folder: base output directory
        final_output_file: final time-series output file path
        cleanup_intermediate: whether to clean up intermediate files
        zenith_fill_mode: Zenith fill mode ('zero' or 'mean')
        use_shape_clip: whether to clip with a shapefile
        shapefile_path: clipping vector path
        clip_processes: number of clipping worker processes
        clip_overwrite: whether to overwrite existing clipping results
        clip_warp_options: extra parameters passed to gdal.Warp
        clip_parallel_mode: clipping parallel mode
        date_start: start-date filter in 'YYYY-MM-DD' format, inclusive; None disables filtering
        date_end: end-date filter in 'YYYY-MM-DD' format, inclusive; None disables filtering
        tile_filter: tile whitelist in the form ['h27v05', 'h28v05'] or [(27,5), (28,5)]; None disables filtering
        verbose: whether to print verbose logs

    Returns:
        results: processing summary dictionary
    """
    # Stage 1: Clip during extraction when a shapefile is provided, then automatically mosaic multi-tile outputs
    stage1_info = stage1_extract_and_pair(
        input_vnp46a2_folder=input_vnp46a2_folder,
        input_vnp46a1_folder=input_vnp46a1_folder, 
        output_base_folder=output_base_folder,
        zenith_fill_mode=zenith_fill_mode,
        clip_shapefile=shapefile_path if use_shape_clip else None,
        clip_warp_options=clip_warp_options,
        clip_shapefile_bbox=use_shapefile_bbox,
        auto_mosaic=True,  # Automatically mosaic multi-tile outputs
        date_start=date_start,
        date_end=date_end,
        tile_filter=tile_filter,
        verbose=verbose
    )
    
    # Stage 2
    results = stage2_generate_time_series(
        stage1_info, final_output_file, zenith_fill_mode, verbose
    )
    
    # Clean up intermediate files
    if cleanup_intermediate:
        ok, err = _safe_rmtree(stage1_info['temp_work_dir'])
        if ok:
            results['temp_folder'] = None
            if verbose:
                print("🧹 Stage1 temporary directory cleaned up")
        else:
            if verbose:
                print(f"⚠️ Failed to clean temporary directory: {err}")
                print("   Hint: a process may still be holding files open; you can remove temp_processing manually later")
            results['temp_folder'] = stage1_info['temp_work_dir']
    else:
        results['temp_folder'] = stage1_info['temp_work_dir']
    
    if verbose:
        print(f"\n✅ All steps completed! Total elapsed time: {results['total_processing_time']:.2f} s")
    
    return results


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == '__main__':
    # Example configuration
    config = {
        'input_vnp46a2_folder': "F:/DATA/Black_Marble/A2",
        'input_vnp46a1_folder': "F:/DATA/Black_Marble/A1",
        'output_base_folder': "F:/DATA/Black_Marble/processed/",
        'final_output_file': "F:/DATA/Black_Marble/processed/ntl_timeseries.txt",
        'zenith_fill_mode': "zero",
        'use_shape_clip': False,
        'shapefile_path': None,
    }
    
    print("🔧 Configuration:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Validate configured paths
    if not os.path.exists(config['input_vnp46a2_folder']):
        print(f"\n❌ VNP46A2 folder does not exist: {config['input_vnp46a2_folder']}")
        print("Update the configuration and run again.")
    else:
        results = complete_ntl_preprocessing_pipeline(**config)
        
        print("\n📊 Processing results:")
        print(f"   Processed files: {results['processed_hdf5_files']}/{results['total_hdf5_files']}")
        print(f"   Time steps: {results['processed_dates']}")
        print(f"   Valid pixels: {results['valid_points']}")
        print(f"   Output file: {results['output_file']}")
