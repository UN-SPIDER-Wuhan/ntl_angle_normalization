# -*- coding: utf-8 -*-
"""
夜光遥感数据预处理模块

本模块提供 VNP46A1/A2 数据的完整预处理功能，包括：
- Stage1: HDF5 图层提取与配对（支持提取时直接裁剪）
- Stage2: 时间序列文本生成（支持自动镶嵌多瓦片）
- 可选: Shapefile 裁剪、瓦片镶嵌

使用示例:
    from functions.preprocessing import (
        stage1_extract_and_pair,
        stage2_generate_time_series,
        clip_rasters_by_shapefile,
        mosaic_tiles_by_date,
        complete_ntl_preprocessing_pipeline
    )
    
    # 分阶段执行（跨瓦片研究区会自动镶嵌）
    stage1_info = stage1_extract_and_pair(a2_folder, a1_folder, output_folder, 
                                          clip_shapefile=shp_path)
    results = stage2_generate_time_series(stage1_info, output_file)
    
    # 或一键执行
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

# GDAL 延迟导入，避免未安装时直接报错
try:
    from osgeo import gdal, ogr, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("⚠️ GDAL 未安装，部分功能不可用")


# =============================================================================
# 常量与正则
# =============================================================================

VIIRS_NAME_RE = re.compile(
    r'^(?P<product>VNP46A[12])\.A(?P<year>\d{4})(?P<doy>\d{3})'
    r'\.h(?P<h>\d{2})v(?P<v>\d{2})\.(?P<collection>\d{3})'
    r'\.(?P<production>\d{13})\.h5$'
)


# =============================================================================
# 辅助函数
# =============================================================================

def parse_h5_name(fname: str) -> dict:
    """
    解析 VNP46A1/A2 HDF5 文件名。
    
    返回:
        dict: {product, year, doy, h, v, collection, production} 或 None
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
    """从元数据构建唯一键 (year, doy, h, v, collection)"""
    return (meta['year'], meta['doy'], meta['h'], meta['v'], meta['collection'])


def _safe_rmtree(path: str, retries: int = 5, delay: float = 0.6) -> tuple:
    """更稳健地删除目录，缓解 Windows 下短暂文件占用问题。"""
    import stat

    def _onerror(func, p, exc_info):
        # 某些文件为只读时，先加写权限再重试一次。
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
            # 末次失败不再等待
            if i < retries - 1:
                time.sleep(delay)

    return False, last_err


def _detect_crs(shp_path: str, sample_raster: str):
    """
    检测 shapefile 与样本栅格 CRS 是否一致。
    
    返回:
        (vec_wkt, ras_wkt, is_same, message)
    """
    vec_ds = ogr.Open(shp_path)
    if vec_ds is None:
        return None, None, False, "无法打开矢量文件"
    
    lyr = vec_ds.GetLayer(0)
    vec_srs = lyr.GetSpatialRef()
    vec_wkt = vec_srs.ExportToWkt() if vec_srs else None
    
    ras_ds = gdal.Open(sample_raster, gdal.GA_ReadOnly)
    if ras_ds is None:
        return vec_wkt, None, False, "无法打开样本栅格"
    
    ras_wkt = ras_ds.GetProjection()
    
    if not vec_wkt or not ras_wkt:
        return vec_wkt, ras_wkt, False, "CRS 读取不完整"
    
    vec_srs_norm = osr.SpatialReference()
    ras_srs_norm = osr.SpatialReference()
    vec_srs_norm.ImportFromWkt(vec_wkt)
    ras_srs_norm.ImportFromWkt(ras_wkt)
    
    same = bool(vec_srs_norm.IsSame(ras_srs_norm))
    return vec_wkt, ras_wkt, same, "OK" if same else "CRS 不一致"


def _warp_single(args):
    """单文件裁剪任务（用于并行）"""
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
# Shapefile 裁剪
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
    并行裁剪 Stage1 生成的所有 GeoTIFF。
    
    参数:
        folders: Stage1 返回的 folders 字典 {'ntl':..., 'quality':..., 'zenith':...}
        shp_path: 裁剪矢量路径
        output_base: 裁剪结果输出目录
        processes: 并行工作进程/线程数
        overwrite: 是否覆盖已存在文件
        warp_options: 传入 gdal.Warp 的额外参数
        parallel_mode: 'auto'|'process'|'thread'|'none'
        chunk_size: 单批提交任务数
        retries: 单文件失败重试次数
        fail_fallback_serial: 并行失败是否串行兜底
        verbose: 是否打印详细信息
    
    返回:
        folders_clipped: 裁剪后的目录字典
    """
    if warp_options is None:
        warp_options = {}
    
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"裁剪矢量不存在: {shp_path}")
    
    os.makedirs(output_base, exist_ok=True)
    
    # 自动选择并行模式
    if parallel_mode == 'auto':
        chosen_mode = 'thread' if sys.platform.startswith('win') else 'process'
    else:
        chosen_mode = parallel_mode
    
    # CRS 检测
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
            print("🧭 CRS 检测: ✅ 一致")
        else:
            print(f"🧭 CRS 警告: ❌ 不一致 - {msg}")
            print("  建议先重投影矢量或在 warp_options 中指定 dstSRS")

    # 如果要求使用外接矩形裁剪，尝试计算 shapefile 在栅格 CRS 下的包络
    shp_bbox = None
    if use_bbox:
        try:
            # 获取 shapefile bbox 并尝试转换到样本栅格 CRS
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
                            # 需要变换 bbox 四个角点
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
                print(f"🗺️ 使用外接矩形裁剪: bbox={shp_bbox}")
        except Exception as e:
            if verbose:
                print(f"⚠️ 计算外接矩形失败，回退到逐像素裁剪: {e}")
            shp_bbox = None
    
    # 收集任务
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
            print("⚠️ 无裁剪任务（可能已存在裁剪结果或无 TIFF 文件）")
        return folders_clipped
    
    if verbose:
        print(f"✂️ 准备裁剪: {len(tasks_all)} 个文件, 模式={chosen_mode}, "
              f"max_workers={processes}, chunk_size={chunk_size}")
    
    # 进度条
    try:
        from tqdm import tqdm
        pbar_total = tqdm(total=len(tasks_all), desc="裁剪进度", unit="file")
        use_tqdm = True
    except ImportError:
        pbar_total = None
        use_tqdm = False
    
    def run_batch(batch_tasks, mode):
        success_paths = []
        fail_entries = []
        
        if mode == 'none':
            for t in batch_tasks:
                # 若指定了 bbox，则在单任务的 warp_options 中注入 outputBounds
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
            # 如果使用 bbox，则替换每个任务的 cutline 参数为 outputBounds
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
                    err = f"执行期异常: {type(e).__name__}: {e}"
                
                if ok:
                    success_paths.append(p)
                else:
                    fail_entries.append((p, err))
                
                if use_tqdm:
                    pbar_total.update(1)
        
        return success_paths, fail_entries
    
    # 执行裁剪
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
            print(f"🚀 启动模式: {mode} (剩余 {len(remaining)} 文件)")
        
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
                        print(f"  ❌ 模式 {mode} 批处理致命错误: {type(e).__name__}: {e}")
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
                print(f"  ⚠️ 模式 {mode} 结束后仍有失败 {len(remaining)} 个，准备回退…")
                
        except Exception as fatal:
            if verbose:
                print(f"💥 模式 {mode} 致命错误: {fatal}")
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
    
    # 串行兜底
    if remaining and ('none' not in mode_sequence):
        if verbose:
            print(f"🛠️ 串行兜底执行 {len(remaining)} 个文件 …")
        for t in remaining:
            ok, p, err = _warp_single(t)
            if ok:
                all_success.add(p)
            else:
                failure_detail[p] = err
    
    # 统计结果
    total = len(tasks_all)
    done = len(all_success)
    final_fail_paths = [t[0] for t in tasks_all if t[0] not in all_success]
    
    if verbose:
        print(f"✅ 裁剪结束: 成功 {done} / 总计 {total}，失败 {len(final_fail_paths)}")
        
        if final_fail_paths:
            print("失败文件示例(前10):")
            for p in final_fail_paths[:10]:
                print("  -", p, '=>', failure_detail.get(p, '未知错误'))
            print("可能原因: CRS不匹配/shapefile几何错误/GDAL驱动问题/内存不足")
    
    return folders_clipped


# =============================================================================
# Stage 1: 提取图层并配对
# =============================================================================

def _get_shapefile_bbox(shp_path: str, target_srs_wkt: str = None):
    """
    获取 shapefile 的外接矩形，可选转换到目标坐标系。
    
    返回:
        (minx, miny, maxx, maxy) 或 None
    """
    try:
        ds = ogr.Open(shp_path)
        if ds is None:
            return None
        lyr = ds.GetLayer(0)
        env = lyr.GetExtent()  # 返回 (minX, maxX, minY, maxY)
        minx, maxx, miny, maxy = env[0], env[1], env[2], env[3]
        
        vec_srs = lyr.GetSpatialRef()
        if vec_srs is None or target_srs_wkt is None:
            return (minx, miny, maxx, maxy)
        
        # 转换到目标坐标系
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(target_srs_wkt)
        
        if not vec_srs.IsSame(target_srs):
            # 设置坐标轴顺序为传统 (x, y) 即 (lon, lat)
            vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
            transform = osr.CoordinateTransformation(vec_srs, target_srs)
            # 注意：这里的 pts 使用正确的 (minx, miny) 等组合
            pts = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
            xs, ys = [], []
            for x, y in pts:
                # TransformPoint 返回 (x, y, z)
                nx, ny, _ = transform.TransformPoint(x, y)
                xs.append(nx)
                ys.append(ny)
            return (min(xs), min(ys), max(xs), max(ys))
        
        return (minx, miny, maxx, maxy)
    except Exception as e:
        print(f"⚠️ 获取 shapefile bbox 失败: {e}")
        return None


def _extract_layer_with_clip(subhdf: str, out_path: str, shp_path: str = None, 
                              warp_options: dict = None, use_bbox: bool = False) -> bool:
    """
    提取单个图层，可选直接裁剪到 shapefile 范围。
    
    参数:
        subhdf: HDF5 子数据集路径
        out_path: 输出 TIFF 路径
        shp_path: 可选的裁剪 shapefile 路径
        warp_options: 额外的 gdal.Warp 参数
        use_bbox: 是否仅使用外接矩形裁剪（保留 bbox 内所有数据）
    
    返回:
        bool: 是否成功
    """
    try:
        r_layer = gdal.Open(subhdf, gdal.GA_ReadOnly)
        if r_layer is None:
            return False
        
        # 获取地理信息
        md = r_layer.GetMetadata_Dict()
        if "HorizontalTileNumber" in md and "VerticalTileNumber" in md:
            hnum = int(md["HorizontalTileNumber"])
            vnum = int(md["VerticalTileNumber"])
            west = 10 * hnum - 180
            north = 90 - 10 * vnum
            east = west + 10
            south = north - 10
            
            if shp_path and os.path.exists(shp_path):
                # 先设置地理参考到内存 VRT
                temp_vrt = f"/vsimem/temp_{os.path.basename(out_path)}.vrt"
                opt_txt = f"-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}"
                trans_opts = gdal.TranslateOptions(gdal.ParseCommandLine(opt_txt))
                gdal.Translate(temp_vrt, r_layer, options=trans_opts)
                
                if use_bbox:
                    # 仅使用外接矩形裁剪（保留 bbox 内所有数据）
                    target_srs = osr.SpatialReference()
                    target_srs.ImportFromEPSG(4326)
                    bbox = _get_shapefile_bbox(shp_path, target_srs.ExportToWkt())
                    
                    if bbox:
                        # bbox = (minx, miny, maxx, maxy)
                        # 检查 bbox 与瓦片是否有交集
                        # tile_bbox 也用 (minx, miny, maxx, maxy) 格式
                        tile_bbox = (west, south, east, north)
                        
                        # 计算交集
                        intersect_minx = max(bbox[0], tile_bbox[0])
                        intersect_miny = max(bbox[1], tile_bbox[1])
                        intersect_maxx = min(bbox[2], tile_bbox[2])
                        intersect_maxy = min(bbox[3], tile_bbox[3])
                        
                        if intersect_minx < intersect_maxx and intersect_miny < intersect_maxy:
                            # 有交集，使用交集范围
                            # GDAL outputBounds 格式: (ulx, uly, lrx, lry) = (minX, maxY, maxX, minY)
                            warp_opts = {
                                'outputBounds': (intersect_minx, intersect_maxy, intersect_maxx, intersect_miny),
                                'dstSRS': 'EPSG:4326',
                            }
                            if warp_options:
                                warp_opts.update(warp_options)
                            # 移除可能冲突的 cutline 参数
                            warp_opts.pop('cutlineDSName', None)
                            warp_opts.pop('cropToCutline', None)
                            
                            gdal.Warp(out_path, temp_vrt, **warp_opts)
                        else:
                            # 无交集，清理并返回失败
                            gdal.Unlink(temp_vrt)
                            return False
                    else:
                        # bbox 计算失败，回退到完整提取
                        gdal.Translate(out_path, temp_vrt)
                else:
                    # 使用 shapefile 边界裁剪（边界外设为 NoData）
                    warp_opts = {
                        'cutlineDSName': shp_path,
                        'cropToCutline': True,
                        'dstSRS': 'EPSG:4326',
                    }
                    if warp_options:
                        warp_opts.update(warp_options)
                    
                    gdal.Warp(out_path, temp_vrt, **warp_opts)
                
                # 清理内存 VRT
                gdal.Unlink(temp_vrt)
            else:
                # 仅提取，不裁剪
                opt_txt = f"-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}"
                opts = gdal.TranslateOptions(gdal.ParseCommandLine(opt_txt))
                gdal.Translate(out_path, r_layer, options=opts)
        else:
            if shp_path and os.path.exists(shp_path):
                if use_bbox:
                    bbox = _get_shapefile_bbox(shp_path)
                    if bbox:
                        # bbox = (minx, miny, maxx, maxy)
                        # GDAL outputBounds 格式: (ulx, uly, lrx, lry) = (minX, maxY, maxX, minY)
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
        print(f"⚠️ 提取图层失败: {e}")
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
    阶段1: 读取 VNP46A2 / VNP46A1 HDF5 文件，提取图层并配对。
    
    输出目录结构:
        <output_base_folder>/temp_processing/
            DNB_BRDF-Corrected_NTL/*.tif
            Mandatory_Quality_Flag/*.tif
            Sensor_Zenith/*.tif
    
    参数:
        input_vnp46a2_folder: VNP46A2 HDF5 文件目录
        input_vnp46a1_folder: VNP46A1 HDF5 文件目录
        output_base_folder: 输出基目录
        zenith_fill_mode: Zenith 填充模式 ('zero' 或 'mean')，Stage2 使用
        clip_shapefile: 可选的裁剪 shapefile 路径（提取时直接裁剪，更高效）
        clip_warp_options: 裁剪参数
        clip_shapefile_bbox: 是否使用外接矩形裁剪
        auto_mosaic: 是否自动镶嵌跨瓦片数据
        date_start: 起始日期过滤，格式 'YYYY-MM-DD'（含），None 表示不过滤
        date_end: 结束日期过滤，格式 'YYYY-MM-DD'（含），None 表示不过滤
        tile_filter: 瓦片白名单，格式为 ['h27v05', 'h28v05'] 或 [(27,5), (28,5)]，None 表示不过滤
        verbose: 是否打印详细信息
    
    返回:
        dict: 包含 temp_work_dir, folders, a2_meta, a1_dict, 统计信息等
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL 未安装，无法执行图层提取")
    
    start_time = time.time()
    
    # 创建输出目录
    temp_work_dir = os.path.join(output_base_folder, "temp_processing")
    os.makedirs(temp_work_dir, exist_ok=True)
    
    folders = {
        'ntl': os.path.join(temp_work_dir, "DNB_BRDF-Corrected_NTL"),
        'zenith': os.path.join(temp_work_dir, "Sensor_Zenith"),
        'quality': os.path.join(temp_work_dir, "Mandatory_Quality_Flag")
    }
    for p in folders.values():
        os.makedirs(p, exist_ok=True)
    
    clip_mode = "直接裁剪" if clip_shapefile else "完整提取"
    if verbose:
        print(f"🚀 [Stage1] 开始提取与配对 ({clip_mode})")
        if clip_shapefile:
            print(f"   裁剪矢量: {clip_shapefile}")
            if clip_shapefile_bbox:
                print("   裁剪方式: 外接矩形裁剪 (bbox)")

    # ---------- 解析过滤条件 ----------
    def _parse_date_to_year_doy(date_str):
        """将 'YYYY-MM-DD' 转换为 (year, doy) 整数对"""
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.year, int(dt.strftime("%j"))

    filter_start_yd = _parse_date_to_year_doy(date_start) if date_start else None
    filter_end_yd   = _parse_date_to_year_doy(date_end)   if date_end   else None

    def _parse_tile_filter(raw):
        """将 tile_filter 统一转为 set of (h, v) int 对"""
        if not raw:
            return None
        # 支持单个字符串直接传入，如 "h27v05"
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
        """检查某条元数据是否满足时间/瓦片过滤条件"""
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
            print(f"   时间过滤: {date_start or '不限'} ~ {date_end or '不限'}")
        if tile_set:
            print(f"   瓦片过滤: {sorted(tile_set)}")
    # ---------- 过滤条件解析完毕 ----------

    # 保存当前工作目录；若 cwd 已不存在（上次运行被 cleanup 删除），回退到安全目录
    try:
        original_dir = os.getcwd()
        if not os.path.exists(original_dir):
            original_dir = output_base_folder
    except Exception:
        original_dir = output_base_folder
    
    # 收集 A2 文件
    os.chdir(input_vnp46a2_folder)
    vnp46a2_files = [f for f in os.listdir('.') if f.endswith('.h5') and 'VNP46A2' in f]
    a2_meta = []
    for f in vnp46a2_files:
        meta = parse_h5_name(f)
        if meta:
            if _meta_in_range(meta):
                a2_meta.append((f, meta))
        elif verbose:
            print(f"  ⚠️ 跳过非标准A2文件名: {f}")
    
    a2_dict = {build_key(meta): f for f, meta in a2_meta}
    
    # 收集 A1 文件
    a1_dict = {}
    if os.path.exists(input_vnp46a1_folder):
        os.chdir(input_vnp46a1_folder)
        vnp46a1_files = [f for f in os.listdir('.') if f.endswith('.h5') and 'VNP46A1' in f]
        for f in vnp46a1_files:
            meta = parse_h5_name(f)
            if meta:
                a1_dict[build_key(meta)] = f
            elif verbose:
                print(f"  ⚠️ 跳过非标准A1文件名: {f}")
    else:
        if verbose:
            print("⚠️ 未找到 VNP46A1 文件夹，将只输出 NTL 和 Quality 图层")
    
    # 图层配置
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
    
    # 进度条
    try:
        from tqdm import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False
    
    # 处理每个 A2 文件
    file_iter = enumerate(a2_meta)
    if use_tqdm:
        file_iter = tqdm(list(file_iter), desc="Stage1 提取图层", unit="file")
    
    for idx, (hdf_file, meta) in file_iter:
        try:
            os.chdir(input_vnp46a2_folder)
            hdf_layer = gdal.Open(hdf_file, gdal.GA_ReadOnly)
            
            if hdf_layer is None:
                failed_files.append((hdf_file, "打开失败"))
                continue
            
            subdatasets = hdf_layer.GetSubDatasets()
            raster_file_pre = hdf_file[:-17] if len(hdf_file) > 17 else hdf_file[:-3]
            
            layers_extracted = 0
            
            # 提取 NTL 和 Quality（支持直接裁剪）
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
                    failed_files.append((hdf_file, f"{layer_name} 提取失败: {e}"))
            
            # 提取 Zenith（支持直接裁剪）
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
                        failed_files.append((hdf_file, f"Zenith 处理异常: {e}"))
                else:
                    missing_zenith_keys.append(key)
            
            # 检查完成度
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
        print(f"\n✅ [Stage1] 完成: {processed_files}/{total_files} 文件, "
              f"缺失Zenith: {len(set(missing_zenith_keys))}, 失败: {len(failed_files)}")
        print(f"   用时: {stage1_time:.2f} 秒")
        if failed_files and len(failed_files) <= 5:
            print("   失败文件:")
            for f, err in failed_files:
                print(f"     - {f}: {err}")
    
    # 自动检测是否需要镶嵌（跨瓦片研究区）
    if auto_mosaic:
        ntl_folder = folders.get('ntl', '')
        if os.path.exists(ntl_folder):
            date_groups = _group_tiffs_by_date(ntl_folder)
            need_mosaic = any(len(files) > 1 for files in date_groups.values())
            
            if need_mosaic:
                if verbose:
                    print("\n🔀 检测到多瓦片数据，执行镶嵌...")
                mosaic_base = os.path.join(temp_work_dir, "mosaic")
                folders = mosaic_tiles_by_date(folders, mosaic_base, verbose)
                stage1_time = time.time() - start_time
                if verbose:
                    print(f"   镶嵌后用时: {stage1_time:.2f} 秒")

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
# 瓦片镶嵌（跨瓦片研究区）
# =============================================================================

def _group_tiffs_by_date(folder: str) -> dict:
    """
    按日期分组 TIFF 文件。
    
    返回:
        {date_str: [file1, file2, ...]}
    """
    groups = {}
    for f in os.listdir(folder):
        if not f.endswith('.tif'):
            continue
        # 提取日期部分: VNP46A2.A2015001
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
    将同一日期的多个瓦片镶嵌为单个文件（用于跨瓦片研究区）。
    
    参数:
        folders: Stage1 返回的 folders 字典
        output_base: 镶嵌结果输出目录
        verbose: 是否打印详细信息
    
    返回:
        mosaic_folders: 镶嵌后的目录字典
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL 未安装，无法执行镶嵌")
    
    os.makedirs(output_base, exist_ok=True)
    
    mosaic_folders = {}
    
    for layer_name, layer_folder in folders.items():
        if not os.path.exists(layer_folder):
            continue
        
        out_folder = os.path.join(output_base, os.path.basename(layer_folder))
        os.makedirs(out_folder, exist_ok=True)
        mosaic_folders[layer_name] = out_folder
        
        # 按日期分组
        date_groups = _group_tiffs_by_date(layer_folder)
        
        # 检查是否需要镶嵌
        need_mosaic = any(len(files) > 1 for files in date_groups.values())
        
        if not need_mosaic:
            # 不需要镶嵌，直接复制/移动
            if verbose:
                print(f"   {layer_name}: 单瓦片，无需镶嵌")
            import shutil
            for files in date_groups.values():
                for f in files:
                    shutil.copy2(f, out_folder)
            continue
        
        if verbose:
            print(f"   {layer_name}: 检测到多瓦片，执行镶嵌...")
        
        try:
            from tqdm import tqdm
            use_tqdm = verbose
        except ImportError:
            use_tqdm = False
        
        date_iter = date_groups.items()
        if use_tqdm:
            date_iter = tqdm(list(date_iter), desc=f"镶嵌 {layer_name}", unit="date")
        
        for date_part, files in date_iter:
            if len(files) == 1:
                # 单瓦片直接复制
                import shutil
                shutil.copy2(files[0], out_folder)
            else:
                # 多瓦片镶嵌
                # 提取前缀用于输出文件名
                prefix = os.path.basename(files[0]).split('VNP46A2')[0]
                out_name = f"{prefix}{date_part}.tif"
                out_path = os.path.join(out_folder, out_name)
                
                try:
                    # 使用 gdal.Warp 进行镶嵌
                    gdal.Warp(out_path, files, format='GTiff')
                except Exception as e:
                    if verbose:
                        print(f"     ⚠️ 镶嵌失败 {date_part}: {e}")
    
    if verbose:
        print(f"   ✅ 镶嵌完成")
    
    return mosaic_folders


# =============================================================================
# Stage 2: 生成时间序列
# =============================================================================

def stage2_generate_time_series(
        stage1_info: dict,
        final_output_file: str,
        zenith_fill_mode: str = "zero",
        verbose: bool = True) -> dict:
    """
    阶段2: 读取 Stage1 生成的 tif 图层，按像素构建时间序列文本。
    
    参数:
        stage1_info: Stage1 返回的信息字典
        final_output_file: 输出时间序列文件路径
        zenith_fill_mode: Zenith 填充模式 ('zero' 或 'mean')
        auto_mosaic: 是否自动检测并镶嵌多瓦片（默认 True）
        verbose: 是否打印详细信息
    
    返回:
        results: 处理结果统计字典
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL 未安装，无法执行时间序列生成")
    
    folders = stage1_info['folders']
    total_files = stage1_info['total_files']
    processed_files = stage1_info['processed_files']
    missing_zenith_pairs = stage1_info['missing_zenith_pairs']
    
    if verbose:
        print("\n📊 [Stage2] 生成时间序列 …")
    
    time_series_start = time.time()

    # 不切换 cwd，避免 Windows 下当前目录句柄占用导致后续无法删除临时目录。
    tiff_files = [f for f in os.listdir(folders['ntl']) if f.endswith('.tif')]
    file_num = len(tiff_files)
    
    if file_num == 0:
        raise Exception("Stage2: 未找到 NTL TIFF 文件")
    
    # 获取影像尺寸
    first_tiff = os.path.join(folders['ntl'], tiff_files[0])
    ntl_sample = gdal.Open(first_tiff)
    if ntl_sample is None:
        raise Exception(f"Stage2: 无法打开样本 TIFF: {first_tiff}")
    m = ntl_sample.RasterYSize
    n = ntl_sample.RasterXSize
    ntl_sample = None
    
    if verbose:
        print(f"   影像尺寸: {m} x {n}, 时间点: {file_num}")
    
    # 预分配数组
    point_ntl = -9999 * np.ones((file_num, m, n), dtype='float32')
    point_zenith = -9999 * np.ones((file_num, m, n), dtype='int32')
    point_lon = -9999 * np.ones((m, n), dtype='float32')
    point_lat = -9999 * np.ones((m, n), dtype='float32')
    point_date = np.zeros(shape=(1, file_num)).astype(np.str_)
    zenith_available_mask = np.zeros((file_num, m, n), dtype=bool)
    
    # 进度条
    try:
        from tqdm import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False
    
    skipped_files = []
    failed_files = []
    
    # 读取所有时间点
    file_iter = enumerate(tiff_files)
    if use_tqdm:
        file_iter = tqdm(list(file_iter), desc="Stage2 读取TIFF", unit="file")
    
    for i, tiff_file in file_iter:
        ntl_data = None
        quality_data = None
        zenith_data = None
        try:
            # 解析日期
            if 'VNP46A2.' in tiff_file:
                pos = tiff_file.find('VNP46A2.')
                part = tiff_file[pos + len('VNP46A2.'): pos + len('VNP46A2.') + 8]
                year = int(part[1:5])
                doy = int(part[5:8])
                offset = date(year, 1, 1) + timedelta(doy - 1)
                point_date[0, i] = offset.strftime('%Y%m%d')
            
            # 构建文件路径
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
            
            # 读取数据
            ntl_data = gdal.Open(ntl_file)
            quality_data = gdal.Open(quality_file)
            zenith_data = gdal.Open(zenith_file) if has_zenith else None
            
            geotrans = ntl_data.GetGeoTransform()
            ntl_array = ntl_data.ReadAsArray()
            quality_array = quality_data.ReadAsArray()
            zenith_array = zenith_data.ReadAsArray() if zenith_data is not None else None
            
            # 遍历像素
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
            # 及时释放 GDAL 句柄，避免 Windows 下文件被占用。
            ntl_data = None
            quality_data = None
            zenith_data = None
    
    # Zenith 填充
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
    
    # 写入输出文件
    output_dir = os.path.dirname(final_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入进度条
    total_pixels = m * n
    point_num = 1
    
    with open(final_output_file, 'w', encoding='utf-8') as f:
        f.write("pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...\n")
        
        pixel_iter = [(j, k) for j in range(m) for k in range(n)]
        if use_tqdm:
            pixel_iter = tqdm(pixel_iter, desc="Stage2 写入像素", unit="px")
        
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
        print(f"\n✅ [Stage2] 完成: {point_num - 1} 有效像素, Zenith填充: {fill_pixels}")
        print(f"   跳过: {len(skipped_files)}, 失败: {len(failed_files)}, 用时: {stage2_time:.2f} 秒")
    
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
# 一键执行接口
# =============================================================================

def complete_ntl_preprocessing_pipeline(
        input_vnp46a2_folder: str,
        input_vnp46a1_folder: str,
        output_base_folder: str,
        final_output_file: str,
        cleanup_intermediate: bool = True,
        zenith_fill_mode: str = "zero",
        # 裁剪相关参数
        use_shape_clip: bool = False,
        shapefile_path: str = None,
        use_shapefile_bbox: bool = False,
        clip_processes: int = 4,
        clip_overwrite: bool = False,
        clip_warp_options: dict = None,
        clip_parallel_mode: str = 'auto',
        # 时间与瓦片过滤参数
        date_start: str = None,
        date_end: str = None,
        tile_filter: list = None,
        verbose: bool = True) -> dict:
    """
    一键执行完整预处理流程: Stage1 -> (可选裁剪) -> Stage2
    
    参数:
        input_vnp46a2_folder: VNP46A2 HDF5 文件目录
        input_vnp46a1_folder: VNP46A1 HDF5 文件目录
        output_base_folder: 输出基目录
        final_output_file: 最终时间序列文件路径
        cleanup_intermediate: 是否清理中间文件
        zenith_fill_mode: Zenith 填充模式 ('zero' 或 'mean')
        use_shape_clip: 是否使用 shapefile 裁剪
        shapefile_path: 裁剪矢量路径
        clip_processes: 裁剪并行进程数
        clip_overwrite: 是否覆盖已存在的裁剪结果
        clip_warp_options: 传入 gdal.Warp 的额外参数
        clip_parallel_mode: 裁剪并行模式
        date_start: 起始日期过滤，格式 'YYYY-MM-DD'（含），None 表示不过滤
        date_end: 结束日期过滤，格式 'YYYY-MM-DD'（含），None 表示不过滤
        tile_filter: 瓦片白名单，格式为 ['h27v05', 'h28v05'] 或 [(27,5), (28,5)]，None 表示不过滤
        verbose: 是否打印详细信息
    
    返回:
        results: 处理结果统计字典
    """
    # Stage 1: 提取时直接裁剪（如果指定了 shapefile），并自动镶嵌多瓦片
    stage1_info = stage1_extract_and_pair(
        input_vnp46a2_folder=input_vnp46a2_folder,
        input_vnp46a1_folder=input_vnp46a1_folder, 
        output_base_folder=output_base_folder,
        zenith_fill_mode=zenith_fill_mode,
        clip_shapefile=shapefile_path if use_shape_clip else None,
        clip_warp_options=clip_warp_options,
        clip_shapefile_bbox=use_shapefile_bbox,
        auto_mosaic=True,  # 跨瓦片自动镶嵌
        date_start=date_start,
        date_end=date_end,
        tile_filter=tile_filter,
        verbose=verbose
    )
    
    # Stage 2
    results = stage2_generate_time_series(
        stage1_info, final_output_file, zenith_fill_mode, verbose
    )
    
    # 清理中间文件
    if cleanup_intermediate:
        ok, err = _safe_rmtree(stage1_info['temp_work_dir'])
        if ok:
            results['temp_folder'] = None
            if verbose:
                print("🧹 已清理 Stage1 临时目录")
        else:
            if verbose:
                print(f"⚠️ 清理临时目录失败: {err}")
                print("   提示: 可能仍有程序占用文件，可稍后手动删除 temp_processing 目录")
            results['temp_folder'] = stage1_info['temp_work_dir']
    else:
        results['temp_folder'] = stage1_info['temp_work_dir']
    
    if verbose:
        print(f"\n✅ 全部完成! 总用时: {results['total_processing_time']:.2f} 秒")
    
    return results


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    # 示例配置
    config = {
        'input_vnp46a2_folder': "F:/DATA/Black_Marble/A2",
        'input_vnp46a1_folder': "F:/DATA/Black_Marble/A1",
        'output_base_folder': "F:/DATA/Black_Marble/processed/",
        'final_output_file': "F:/DATA/Black_Marble/processed/ntl_timeseries.txt",
        'zenith_fill_mode': "zero",
        'use_shape_clip': False,
        'shapefile_path': None,
    }
    
    print("🔧 配置:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # 检查路径
    if not os.path.exists(config['input_vnp46a2_folder']):
        print(f"\n❌ VNP46A2 文件夹不存在: {config['input_vnp46a2_folder']}")
        print("请修改配置后重新运行。")
    else:
        results = complete_ntl_preprocessing_pipeline(**config)
        
        print("\n📊 处理结果:")
        print(f"   处理文件: {results['processed_hdf5_files']}/{results['total_hdf5_files']}")
        print(f"   时间点: {results['processed_dates']}")
        print(f"   有效像素: {results['valid_points']}")
        print(f"   输出文件: {results['output_file']}")
