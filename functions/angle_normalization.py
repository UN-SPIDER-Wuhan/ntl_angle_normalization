# -*- coding: utf-8 -*-
"""
角度归一化处理模块 (Gradient Descent / Least Squares)

本模块提供夜光遥感数据的传感器天顶角归一化功能，包括：
- 数据读取与解析
- 拟合优度与相关性计算
- 角度归一化核心算法
- 时间序列可视化与指标计算
- 并行化处理支持

使用示例:
    from functions.angle_normalization import (
        readFile, visScatterAndFitCurve, visTimeSeries
    )
    
    pointNumLngLatMap, pointsDic, keyList = readFile('input.txt')
    visScatterAndFitCurve(pointsDic, output_file, fit_result_path, 
                          pointNumLngLatMap, output_params_path)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math
import scipy.stats as stats
import scipy.optimize
import datetime as dtime
import matplotlib as mat
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings


# =============================================================================
# 数据读取函数
# =============================================================================

def readFile(filePath: str, min_records: int = 10):
    """
    读取时间序列文本文件，解析为字典结构。
    
    文件格式:
        pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...
    
    参数:
        filePath: 输入文件路径
        min_records: 最少记录数阈值，少于此数的点将被过滤（默认10）
    
    返回:
        pointNumLngLatMap: {pointNum: "lng,lat"} 点位坐标映射
        pointsDic: {pointNum: [[NTLValue, Zenith, YYYYMMDD], ...]} 时间序列数据
        keyList: 所有点位编号列表（含被过滤的）
    """
    with open(filePath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pointNumLngLatMap = {}
    pointsDic = {}
    keyList = []
    
    for i, line in enumerate(lines):
        if i == 0:  # 跳过表头
            continue
        
        temp_list = line.strip().split(":")
        if len(temp_list) < 3:
            continue
            
        pointNum = temp_list[0]
        pointLngLat = temp_list[1]
        pointsDic[pointNum] = []
        pointNumLngLatMap[pointNum] = pointLngLat
        keyList.append(pointNum)
        
        info_list = temp_list[2].split(";")
        for info in info_list:
            value_list = info.split(",")
            if len(value_list) >= 3:
                try:
                    # [NTLValue, Zenith, YYYYMMDD]
                    pointsDic[pointNum].append([
                        float(value_list[2]),  # NTL
                        float(value_list[1]),  # Zenith
                        value_list[0]          # Date
                    ])
                except (ValueError, IndexError):
                    continue
    
    # 过滤记录数不足的点
    new_pointsDic = {}
    new_pointNumLngLatMap = {}
    for key in pointsDic:
        if len(pointsDic[key]) >= min_records:
            new_pointsDic[key] = pointsDic[key]
            new_pointNumLngLatMap[key] = pointNumLngLatMap[key]
    
    return new_pointNumLngLatMap, new_pointsDic, keyList


def filter_points_outliers_3sigma(
        pointsDic: dict,
        pointNumLngLatMap: dict,
        sigma: float = 3.0,
        ntl_upper_bound: float = None,
        min_records_after_filter: int = 10,
        verbose: bool = True):
    """
    在归一化前按点位执行异常值过滤（3-sigma + 可选硬阈值）。

    规则:
    1. 对每个点位的 NTL 序列按 |x-mean| <= sigma*std 过滤（std=0 时跳过 3-sigma）
    2. 若设置 ntl_upper_bound，则同时过滤 NTL > 上限 的记录
    3. 过滤后记录数 < min_records_after_filter 的点位将被剔除

    返回:
        filtered_pointsDic, filtered_pointNumLngLatMap, summary
    """
    filtered_points = {}
    filtered_map = {}

    total_points = len(pointsDic)
    total_records_before = 0
    total_records_after = 0
    removed_by_sigma = 0
    removed_by_upper = 0
    dropped_points = 0

    for key, records in pointsDic.items():
        if not records:
            dropped_points += 1
            continue

        ntl_vals = np.array([row[0] for row in records], dtype='float64')
        total_records_before += len(records)

        keep_mask = np.ones(len(records), dtype=bool)

        # 3-sigma 过滤
        mean_v = float(np.mean(ntl_vals))
        std_v = float(np.std(ntl_vals))
        if std_v > 0:
            sigma_mask = np.abs(ntl_vals - mean_v) <= sigma * std_v
            removed_by_sigma += int((~sigma_mask).sum())
            keep_mask &= sigma_mask

        # 可选硬阈值过滤
        if ntl_upper_bound is not None:
            upper_mask = ntl_vals <= float(ntl_upper_bound)
            removed_by_upper += int((~upper_mask).sum())
            keep_mask &= upper_mask

        kept_records = [rec for rec, keep in zip(records, keep_mask) if keep]

        if len(kept_records) < min_records_after_filter:
            dropped_points += 1
            continue

        filtered_points[key] = kept_records
        if key in pointNumLngLatMap:
            filtered_map[key] = pointNumLngLatMap[key]
        total_records_after += len(kept_records)

    summary = {
        'enabled': True,
        'sigma': float(sigma),
        'ntl_upper_bound': None if ntl_upper_bound is None else float(ntl_upper_bound),
        'points_before': int(total_points),
        'points_after': int(len(filtered_points)),
        'points_dropped': int(dropped_points),
        'records_before': int(total_records_before),
        'records_after': int(total_records_after),
        'records_removed': int(total_records_before - total_records_after),
        'removed_by_sigma': int(removed_by_sigma),
        'removed_by_upper_bound': int(removed_by_upper),
    }

    if verbose:
        print('[info] 归一化前异常值过滤（3-sigma）完成: ' +
              f"点位 {summary['points_before']} -> {summary['points_after']}，" +
              f"记录 {summary['records_before']} -> {summary['records_after']}")

    return filtered_points, filtered_map, summary


# =============================================================================
# 统计与拟合辅助函数
# =============================================================================

def calGoodnessOfFit(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算拟合优度 R²（决定系数）。
    
    参数:
        y_pred: 预测值数组
        y_true: 真实值数组
    
    返回:
        R² 值，范围 (-∞, 1]，越接近1拟合越好
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_mean = np.mean(y_true)
    
    ss_total = np.sum((y_true - y_true_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 0.0
    
    return 1 - (ss_res / ss_total)


def calCorrelation(x_array, y_array) -> float:
    """
    计算 Pearson 相关系数。
    
    参数:
        x_array: X 数组
        y_array: Y 数组
    
    返回:
        相关系数，范围 [-1, 1]；若标准差为0则返回 np.nan
    """
    x = np.array(x_array)
    y = np.array(y_array)
    
    std_x = np.std(x)
    std_y = np.std(y)
    
    if std_x == 0 or std_y == 0:
        return np.nan
    
    cov_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    return cov_xy / (std_x * std_y)


def _func_quadratic(p, x):
    """二次函数: f(x) = a*x² + b*x + c"""
    a, b, c = p
    return a * x ** 2 + b * x + c


def _func_quadratic_c1(p, x):
    """二次函数（截距固定为1）: f(x) = a*x² + b*x + 1"""
    a, b = p
    return a * x ** 2 + b * x + 1


def _error_quadratic(p, x, y):
    """二次拟合的均方误差"""
    return (1 / len(x)) * (_func_quadratic(p, x) - y) ** 2


def _error_quadratic_c1(p, x, y):
    """截距为1的二次拟合均方误差"""
    return (1 / len(x)) * (_func_quadratic_c1(p, x) - y) ** 2


# =============================================================================
# 单点归一化处理函数（用于并行化）
# =============================================================================

def _normalize_single_point(args):
    """
    处理单个点位的角度归一化（用于并行计算）。
    
    参数:
        args: (key, X, Y, time_values) 元组
    
    返回:
        dict: 包含点位信息、归一化结果和统计指标
    """
    key, X, Y, time_values = args
    
    X = np.array(X)
    Y = np.array(Y)
    mean_ntl = float(np.mean(Y)) if len(Y) else 0.0
    
    try:
        # Step 1: 多项式拟合初始参数
        parameters = np.polyfit(X, Y, 2)
        
        # Step 2: 最小二乘优化
        parameter_init = np.array([0, 0, 0])
        Para = leastsq(_error_quadratic, parameter_init, args=(X, Y))
        a, b, c = Para[0]
        
        # 计算拟合指标
        fit_y = parameters[0] * X ** 2 + parameters[1] * X + parameters[2]
        GoodnessOfFit_score = calGoodnessOfFit(fit_y, Y)
        r, p_value = stats.pearsonr(fit_y, Y)
        
        fit_y_leastsq = a * X ** 2 + b * X + c
        GoodnessOfFit_score_leastsq = calGoodnessOfFit(fit_y_leastsq, Y)
        
        # Step 3: 优化归一化参数
        Z = X
        R = Y
        
        # 防止除以 0
        base_c = parameters[2] if parameters[2] != 0 else 1e-6
        p0 = np.array([parameters[0] / base_c, parameters[1] / base_c, 0, 0, base_c])
        
        # 优化目标：最大化校正后数据与趋势的相关性
        def objective(p):
            denom = p[0] * (Z**2) + p[1] * Z + np.ones(len(Z))
            corrected = R / denom
            trend = p[2] * (Z**2) + p[3] * Z + np.full(len(Z), p[4])
            
            n = len(Z)
            sum_xy = np.sum(corrected * trend)
            sum_x = np.sum(corrected)
            sum_y = np.sum(trend)
            sum_x2 = np.sum(corrected ** 2)
            sum_y2 = np.sum(trend ** 2)
            
            numerator = n * sum_xy - sum_x * sum_y
            denom_corr = np.sqrt(n * sum_x2 - sum_x**2) * np.sqrt(n * sum_y2 - sum_y**2)
            
            if denom_corr == 0:
                return 0
            return (numerator / denom_corr) ** 2
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_final = scipy.optimize.fmin(func=objective, x0=p0, disp=False)
        
        # Step 4: 应用归一化
        normalizationY = R / (p_final[0] * (Z**2) + p_final[1] * Z + np.ones(len(Z)))
        
        # 处理负值（保留原始值）
        negative_mask = normalizationY < 0
        normalizationY[negative_mask] = R[negative_mask]
        
        # 归一化后再评估与天顶角的相关性（越低越好）
        try:
            post_params = np.polyfit(Z, normalizationY, 2)
            post_fit = post_params[0] * Z ** 2 + post_params[1] * Z + post_params[2]
            r2_post = calGoodnessOfFit(post_fit, normalizationY)
        except Exception:
            r2_post = np.nan
        
        return {
            'key': key,
            'success': True,
            'normalizationY': normalizationY.tolist(),
            'X': X.tolist(),
            'time_values': time_values,
            'p_final': p_final.tolist(),
            'r_squared': r ** 2,
            'stats': {
                'point': key,
                'r2_polyfit': GoodnessOfFit_score,
                'r2_leastsq': GoodnessOfFit_score_leastsq,
                'pearson_r2': r ** 2,
                'r2_post': r2_post,
                'mean_ntl': mean_ntl
            }
        }
    except Exception as e:
        return {
            'key': key,
            'success': False,
            'error': str(e),
            'stats': {
                'point': key,
                'r2_polyfit': np.nan,
                'r2_leastsq': np.nan,
                'pearson_r2': np.nan,
                'r2_post': np.nan,
                'mean_ntl': mean_ntl
            }
        }


# =============================================================================
# 角度归一化核心函数
# =============================================================================

def normalizationZenith(visX_dic: dict, visY_dic: dict, 
                        outputFile: str, output_params_path: str,
                        plot_scatter: bool = False, verbose: bool = True,
                        progress=None, stats_collector=None):
    """
    对每个点位进行角度归一化处理。
    
    算法流程:
    1. 对每个点的 (Zenith, NTL) 数据进行二次多项式拟合
    2. 使用最小二乘法优化参数
    3. 通过优化目标函数（最大化校正后与趋势的相关性）进一步优化
    4. 应用归一化公式: NTL_corrected = NTL / (a*Z² + b*Z + 1)
    
    参数:
        visX_dic: {pointNum: [Zenith values]} 天顶角数据
        visY_dic: {pointNum: [NTL values]} 夜光亮度数据
        outputFile: 输出拟合结果文件路径（记录 R²）
        output_params_path: 输出参数文件路径
        plot_scatter: 是否绘制散点图（默认 False）
        verbose: 是否打印详细信息（默认 True）
    
    返回:
        normalizationResult_array: 归一化后的 NTL 数组列表
    """
    if stats_collector is None:
        stats_collector = []

    with open(outputFile, 'w', encoding='utf-8') as f, \
         open(output_params_path, 'w', encoding='utf-8') as f1:
        
        normalizationResult_array = []
        
        for key in visX_dic:
            X = np.array(visX_dic[key])
            Y = np.array(visY_dic[key])
            mean_ntl = float(np.mean(Y)) if len(Y) else 0.0
            
            # Step 1: 多项式拟合初始参数
            parameters = np.polyfit(X, Y, 2)
            
            # Step 2: 最小二乘优化
            parameter_init = np.array([0, 0, 0])
            Para = leastsq(_error_quadratic, parameter_init, args=(X, Y))
            a, b, c = Para[0]
            
            if verbose:
                print(f'{key} | polyfit a={parameters[0]:.5f} b={parameters[1]:.5f} c={parameters[2]:.5f}')
            
            # 计算拟合指标
            fit_y = parameters[0] * X ** 2 + parameters[1] * X + parameters[2]
            GoodnessOfFit_score = calGoodnessOfFit(fit_y, Y)
            r, p_value = stats.pearsonr(fit_y, Y)
            
            fit_y_leastsq = a * X ** 2 + b * X + c
            GoodnessOfFit_score_leastsq = calGoodnessOfFit(fit_y_leastsq, Y)
            
            if verbose:
                print(f'R2 polyfit: {GoodnessOfFit_score:.4f} | Pearson r²: {r**2:.4f} p: {p_value:.4e}')
                print(f'R2 leastsq: {GoodnessOfFit_score_leastsq:.4f}')

            # 归一化后再评估与天顶角的相关性（越低越好）
            try:
                post_params = np.polyfit(Z, normalizationY, 2)
                post_fit = post_params[0] * Z ** 2 + post_params[1] * Z + post_params[2]
                r2_post = calGoodnessOfFit(post_fit, normalizationY)
            except Exception:
                r2_post = np.nan

            stats_collector.append({
                'point': key,
                'r2_polyfit': GoodnessOfFit_score,
                'r2_leastsq': GoodnessOfFit_score_leastsq,
                'pearson_r2': r ** 2,
                'r2_post': r2_post,
                'mean_ntl': mean_ntl
            })
            
            # Step 3: 优化归一化参数
            Z = X
            R = Y
            
            # 防止除以 0
            base_c = parameters[2] if parameters[2] != 0 else 1e-6
            p0 = np.array([parameters[0] / base_c, parameters[1] / base_c, 0, 0, base_c])
            
            # 优化目标：最大化校正后数据与趋势的相关性
            def objective(p):
                denom = p[0] * (Z**2) + p[1] * Z + np.ones(len(Z))
                corrected = R / denom
                trend = p[2] * (Z**2) + p[3] * Z + np.full(len(Z), p[4])
                
                n = len(Z)
                sum_xy = np.sum(corrected * trend)
                sum_x = np.sum(corrected)
                sum_y = np.sum(trend)
                sum_x2 = np.sum(corrected ** 2)
                sum_y2 = np.sum(trend ** 2)
                
                numerator = n * sum_xy - sum_x * sum_y
                denom_corr = np.sqrt(n * sum_x2 - sum_x**2) * np.sqrt(n * sum_y2 - sum_y**2)
                
                if denom_corr == 0:
                    return 0
                return (numerator / denom_corr) ** 2
            
            p_final = scipy.optimize.fmin(func=objective, x0=p0, disp=False)
            
            # Step 4: 应用归一化
            normalizationY = R / (p_final[0] * (Z**2) + p_final[1] * Z + np.ones(len(Z)))
            
            # 处理负值（保留原始值）
            negative_mask = normalizationY < 0
            normalizationY[negative_mask] = R[negative_mask]
            
            normalizationResult_array.append(normalizationY)
            
            # 写入结果
            f.write(f'{key}:{r ** 2}\n')
            f1.write(f'{key}:{",".join([str(x) for x in p_final])}\n')
            
            # 可选绘图
            if plot_scatter:
                _plot_scatter_fit(key, X, Y, normalizationY, parameters)

            if progress is not None:
                before_vals = [item['r2_polyfit'] for item in stats_collector if not np.isnan(item['r2_polyfit'])]
                after_vals = [item['r2_post'] for item in stats_collector if not np.isnan(item['r2_post'])]
                avg_before = sum(before_vals) / len(before_vals) if before_vals else 0.0
                avg_after = sum(after_vals) / len(after_vals) if after_vals else 0.0
                progress.set_postfix({
                    '点号': key,
                    '均值': f'{mean_ntl:.2f}',
                    'R2前': f'{GoodnessOfFit_score:.3f}',
                    'R2后': f'{r2_post:.3f}' if not np.isnan(r2_post) else 'nan',
                    '均前': f'{avg_before:.3f}',
                    '均后': f'{avg_after:.3f}'
                })
                progress.update(1)
    
    return normalizationResult_array


# =============================================================================
# 并行化归一化处理
# =============================================================================

def _normalize_batch(batch_args):
    """
    批量处理多个点位（减少进程间通信开销）
    """
    results = []
    for args in batch_args:
        results.append(_normalize_single_point(args))
    return results


def normalizationZenith_parallel(visX_dic: dict, visY_dic: dict, visTime_dic: dict,
                                  n_workers: int = None, show_progress: bool = True,
                                  verbose: bool = False, batch_size: int = None):
    """
    使用多进程并行处理角度归一化。
    
    参数:
        visX_dic: {pointNum: [Zenith values]} 天顶角数据
        visY_dic: {pointNum: [NTL values]} 夜光亮度数据
        visTime_dic: {pointNum: [Date values]} 时间数据
        n_workers: 并行进程数，默认为 CPU 核心数
        show_progress: 是否显示进度条
        verbose: 是否打印详细信息
        batch_size: 每批处理的点位数，默认自动计算
    
    返回:
        results_dict: {key: result} 结果字典
        stats_collector: 统计信息列表
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # 准备任务列表
    tasks = []
    for key in visX_dic:
        tasks.append((key, visX_dic[key], visY_dic[key], visTime_dic[key]))
    
    total_points = len(tasks)
    
    # 自动计算批大小：每个进程至少处理 50 个点，减少进程通信开销
    if batch_size is None:
        batch_size = max(50, total_points // (n_workers * 4))
    
    # 将任务分批
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    
    results_dict = {}
    stats_collector = []
    failed_count = 0
    processed_count = 0
    
    # 使用进程池并行处理批次
    progress = None
    try:
        if show_progress:
            try:
                from tqdm import tqdm
                progress = tqdm(total=total_points, 
                               desc=f'AngleNorm(并行x{n_workers}, 批={batch_size})', 
                               ncols=110)
            except ImportError:
                progress = None
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {executor.submit(_normalize_batch, batch): i for i, batch in enumerate(batches)}
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    
                    for result in batch_results:
                        key = result['key']
                        results_dict[key] = result
                        stats_collector.append(result['stats'])
                        processed_count += 1
                        
                        if not result['success']:
                            failed_count += 1
                    
                    if progress is not None:
                        # 只在批次完成时更新进度
                        before_vals = [s['r2_polyfit'] for s in stats_collector if not np.isnan(s['r2_polyfit'])]
                        after_vals = [s['r2_post'] for s in stats_collector if not np.isnan(s['r2_post'])]
                        avg_before = sum(before_vals) / len(before_vals) if before_vals else 0.0
                        avg_after = sum(after_vals) / len(after_vals) if after_vals else 0.0
                        progress.update(len(batch_results))
                        progress.set_postfix({
                            '失败': failed_count,
                            'R2前': f'{avg_before:.3f}',
                            'R2后': f'{avg_after:.3f}'
                        })
                        
                except Exception as e:
                    batch_size_actual = len(batches[batch_idx])
                    failed_count += batch_size_actual
                    if verbose:
                        print(f'[error] 批次处理异常: batch_{batch_idx} -> {e}')
                    if progress is not None:
                        progress.update(batch_size_actual)
    
    finally:
        if progress is not None:
            progress.close()
    
    if verbose:
        print(f'[info] 并行处理完成: 成功 {len(results_dict) - failed_count}, 失败 {failed_count}')
    
    return results_dict, stats_collector


def visScatterAndFitCurve_parallel(result_dic: dict, outputFile: str, fit_result_path: str,
                                    pointNumLngLatMap: dict, output_params_path: str,
                                    n_workers: int = None, show_progress: bool = True,
                                    verbose: bool = False):
    """
    使用多进程并行执行角度归一化并写入结果。
    
    参数:
        result_dic: readFile 返回的 pointsDic
        outputFile: 拟合结果输出路径（R² 记录）
        fit_result_path: 校正后时间序列输出路径
        pointNumLngLatMap: 点位坐标映射
        output_params_path: 拟合参数输出路径
        n_workers: 并行进程数
        show_progress: 是否显示进度条
        verbose: 是否打印详细信息
    
    返回:
        stats_collector: 统计信息列表
    """
    # 整理数据
    visX_dic = {}   # Zenith
    visY_dic = {}   # NTL
    visTime_dic = {}  # Date
    
    for key in result_dic:
        x_value, y_value, time_value = [], [], []
        for record in result_dic[key]:
            y_value.append(record[0])      # NTL
            x_value.append(record[1])      # Zenith
            time_value.append(int(record[2]))  # Date
        visX_dic[key] = x_value
        visY_dic[key] = y_value
        visTime_dic[key] = time_value
    
    # 并行执行归一化
    results_dict, stats_collector = normalizationZenith_parallel(
        visX_dic, visY_dic, visTime_dic,
        n_workers=n_workers, show_progress=show_progress, verbose=verbose
    )
    
    # 写入结果文件
    with open(fit_result_path, 'w', encoding='utf-8') as f1, \
         open(outputFile, 'w', encoding='utf-8') as f2, \
         open(output_params_path, 'w', encoding='utf-8') as f3:
        
        f1.write('pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...\n')
        
        # 按原始顺序写入
        for key in visX_dic:
            if key not in results_dict or not results_dict[key]['success']:
                continue
            
            result = results_dict[key]
            normalizationY = result['normalizationY']
            X = result['X']
            time_values = result['time_values']
            
            # 写入校正后时间序列
            f1.write(f'{key}:{pointNumLngLatMap[key]}:')
            entries = []
            for k in range(len(normalizationY)):
                entries.append(f'{time_values[k]},{X[k]},{normalizationY[k]}')
            f1.write(';'.join(entries))
            f1.write('\n')
            
            # 写入 R² 和参数
            f2.write(f'{key}:{result["r_squared"]}\n')
            f3.write(f'{key}:{",".join([str(x) for x in result["p_final"]])}\n')
    
    return stats_collector


def _plot_scatter_fit(key: str, X: np.ndarray, Y: np.ndarray, 
                      Y_corrected: np.ndarray, parameters: np.ndarray):
    """绘制散点图与拟合曲线"""
    try:
        x_curve = np.arange(0, 90, 0.02)
        y_curve = parameters[0] * x_curve ** 2 + parameters[1] * x_curve + parameters[2]
        
        plt.figure(figsize=(7, 5))
        plt.scatter(X, Y, color='black', s=8, label='original points')
        plt.scatter(X, Y_corrected, color='green', s=8, label='corrected points')
        plt.plot(x_curve, y_curve, color='red', label='quadratic fit')
        plt.xlabel('Sensor Zenith (degree)')
        plt.ylabel('Nighttime Light Radiance')
        plt.title(f'Scatter & fit - {key}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f'[warn] 绘制散点失败: {key} -> {e}')


# =============================================================================
# 拟合并写入校正结果
# =============================================================================

def visScatterAndFitCurve(result_dic: dict, outputFile: str, fit_result_path: str,
                          pointNumLngLatMap: dict, output_params_path: str,
                          plot_scatter: bool = False, verbose: bool = True,
                          progress=None, stats_collector=None):
    """
    执行角度归一化并写入校正后的时间序列结果。
    
    参数:
        result_dic: readFile 返回的 pointsDic
        outputFile: 拟合结果输出路径（R² 记录）
        fit_result_path: 校正后时间序列输出路径
        pointNumLngLatMap: 点位坐标映射
        output_params_path: 拟合参数输出路径
        plot_scatter: 是否绘制散点图
        verbose: 是否打印详细信息
    """
    with open(fit_result_path, 'w', encoding='utf-8') as f1:
        f1.write('pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...\n')
        
        # 整理数据
        visX_dic = {}   # Zenith
        visY_dic = {}   # NTL
        visTime_dic = {}  # Date
        
        for key in result_dic:
            x_value, y_value, time_value = [], [], []
            for record in result_dic[key]:
                y_value.append(record[0])      # NTL
                x_value.append(record[1])      # Zenith
                time_value.append(int(record[2]))  # Date
            visX_dic[key] = x_value
            visY_dic[key] = y_value
            visTime_dic[key] = time_value
        
        # 执行归一化
        normalizationResult_array = normalizationZenith(
            visX_dic, visY_dic, outputFile, output_params_path,
            plot_scatter=plot_scatter, verbose=verbose,
            progress=progress, stats_collector=stats_collector
        )
        
        # 写入校正结果
        for norm_index, key in enumerate(visX_dic):
            f1.write(f'{key}:{pointNumLngLatMap[key]}:')
            entries = []
            for k in range(len(normalizationResult_array[norm_index])):
                entries.append(f'{visTime_dic[key][k]},{visX_dic[key][k]},{normalizationResult_array[norm_index][k]}')
            f1.write(';'.join(entries))
            f1.write('\n')


# =============================================================================
# 时间序列可视化与指标计算
# =============================================================================

def visTimeSeries(pointsDic: dict, pointsDic_fit: dict, area: str,
                  plot_series: bool = True, verbose: bool = True):
    """
    可视化原始与校正后的时间序列，并计算评价指标。
    
    计算指标:
    - NDHDNTL: (max - min) / (max + min)，衡量动态范围
    - CV: 变异系数 = std / mean * 100，衡量波动程度
    
    参数:
        pointsDic: 原始数据字典
        pointsDic_fit: 校正后数据字典
        area: 区域名称（用于图表标题）
        plot_series: 是否绘制时间序列图
        verbose: 是否打印指标
    
    返回:
        metrics: {pointNum: {ori_ndhdntl, fit_ndhdntl, ori_cv, fit_cv}}
    """
    # 整理原始数据
    pointsDic_sorted = {}
    for key in pointsDic:
        pointsDic_sorted[key] = []
        for item in pointsDic[key]:
            if item[1]:  # 有效 Zenith
                pointsDic_sorted[key].append([item[0], item[1], int(item[2])])
        pointsDic_sorted[key].sort(key=lambda x: x[2])
    
    visNTL_dic = {}
    visTime_dic = {}
    for key in pointsDic_sorted:
        visNTL_dic[key] = [row[0] for row in pointsDic_sorted[key]]
        visTime_dic[key] = [str(row[2]) for row in pointsDic_sorted[key]]
    
    # 整理校正后数据（按原始时间对齐）
    pointsDic_fit_sorted = {}
    for key in pointsDic_fit:
        pointsDic_fit_sorted[key] = []
        for item in pointsDic_fit[key]:
            if str(item[2]) in visTime_dic.get(key, []):
                pointsDic_fit_sorted[key].append([item[0], item[1], int(item[2])])
        pointsDic_fit_sorted[key].sort(key=lambda x: x[2])
    
    visNTL_dic_fit = {}
    for key in pointsDic_fit_sorted:
        visNTL_dic_fit[key] = [row[0] for row in pointsDic_fit_sorted[key]]
    
    # 计算指标并可视化
    metrics = {}
    for key in pointsDic_sorted:
        if key not in visNTL_dic_fit or not visNTL_dic_fit[key]:
            continue
            
        ori_values = np.array(visNTL_dic[key])
        fit_values = np.array(visNTL_dic_fit[key])
        
        # NDHDNTL
        ori_min, ori_max = np.min(ori_values), np.max(ori_values)
        fit_min, fit_max = np.min(fit_values), np.max(fit_values)
        
        ori_ndhdntl = (ori_max - ori_min) / (ori_max + ori_min) if (ori_max + ori_min) != 0 else 0
        fit_ndhdntl = (fit_max - fit_min) / (fit_max + fit_min) if (fit_max + fit_min) != 0 else 0
        
        # CV
        ori_cv = np.std(ori_values) / np.mean(ori_values) * 100 if np.mean(ori_values) != 0 else 0
        fit_cv = np.std(fit_values) / np.mean(fit_values) * 100 if np.mean(fit_values) != 0 else 0
        
        metrics[key] = {
            'ori_ndhdntl': ori_ndhdntl,
            'fit_ndhdntl': fit_ndhdntl,
            'ori_cv': ori_cv,
            'fit_cv': fit_cv
        }
        
        if verbose:
            print(f'{key} original NTL NDHDNTL value : {ori_ndhdntl:.4f}')
            print(f'{key} fit NTL NDHDNTL value : {fit_ndhdntl:.4f}')
            print(f'{key} original NTL cv value : {ori_cv:.2f}')
            print(f'{key} fit NTL cv value : {fit_cv:.2f}')
        
        if plot_series:
            _plot_time_series(key, area, visTime_dic[key], 
                            visNTL_dic[key], visNTL_dic_fit[key])
    
    return metrics


def _plot_time_series(key: str, area: str, time_strs: list,
                      ori_values: list, fit_values: list):
    """绘制时间序列对比图"""
    try:
        time_datetime = [dtime.datetime.strptime(t, '%Y%m%d') for t in time_strs]
        
        plt.figure(figsize=(8, 6))
        plt.plot_date(time_datetime, ori_values, color='red',
                     label='original time series', linestyle='-', marker='')
        plt.plot_date(time_datetime, fit_values, color='#0072B2',
                     label='angle-normalized time series', linestyle='-', marker='')
        plt.title(f'{area} {key}', fontsize=18)
        plt.xlabel('Date')
        plt.ylabel('Nighttime Light Radiance')
        plt.gca().xaxis.set_major_formatter(mat.dates.DateFormatter('%Y-%m'))
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f'[warn] 绘制时间序列失败: {key} -> {e}')


# =============================================================================
# 归一化前后对比可视化
# =============================================================================

def display_comparison_results(results: dict, name: str, show_details: int = 10):
    """
    显示归一化前后的数据对比结果。
    
    参数:
        results: run_angle_normalization 返回的结果字典
        name: 数据集名称
        show_details: 显示详细对比的点位数量（默认10）
    """
    import pandas as pd
    
    print(f'\n{"="*50}')
    print('📊 归一化前后数据对比')
    print(f'{"="*50}')
    
    # 1. R² 统计对比
    r2_summary = results.get('r2_summary', {})
    if r2_summary.get('before') and r2_summary.get('after'):
        before = r2_summary['before']
        after = r2_summary['after']
        
        comparison_data = {
            '指标': ['均值', '中位数', '25%分位', '75%分位', '最小值', '最大值'],
            'R² (处理前)': [before['mean'], before['median'], before['p25'], before['p75'], before['min'], before['max']],
            'R² (处理后)': [after['mean'], after['median'], after['p25'], after['p75'], after['min'], after['max']]
        }
        df_r2 = pd.DataFrame(comparison_data)
        df_r2['变化'] = df_r2['R² (处理后)'] - df_r2['R² (处理前)']
        df_r2['R² (处理前)'] = df_r2['R² (处理前)'].apply(lambda x: f'{x:.4f}')
        df_r2['R² (处理后)'] = df_r2['R² (处理后)'].apply(lambda x: f'{x:.4f}')
        df_r2['变化'] = df_r2['变化'].apply(lambda x: f'{x:+.4f}')
        
        print('\n🔢 R² 统计对比（R²越低表示天顶角影响越小）:')
        try:
            from IPython.display import display
            display(df_r2)
        except ImportError:
            print(df_r2.to_string(index=False))
    
    # 2. CV 和 NDHDNTL 指标对比
    metrics = results.get('metrics', {})
    if metrics:
        metrics_list = []
        for point_id, m in metrics.items():
            metrics_list.append({
                '点号': point_id,
                'CV原始(%)': m['ori_cv'],
                'CV校正(%)': m['fit_cv'],
                'CV变化(%)': m['fit_cv'] - m['ori_cv'],
                'NDHDNTL原始': m['ori_ndhdntl'],
                'NDHDNTL校正': m['fit_ndhdntl'],
                'NDHDNTL变化': m['fit_ndhdntl'] - m['ori_ndhdntl']
            })
        
        df_metrics = pd.DataFrame(metrics_list)
        
        # 计算汇总统计
        summary = {
            '指标': ['平均CV(%)', '平均NDHDNTL'],
            '原始': [df_metrics['CV原始(%)'].mean(), df_metrics['NDHDNTL原始'].mean()],
            '校正': [df_metrics['CV校正(%)'].mean(), df_metrics['NDHDNTL校正'].mean()],
        }
        summary['变化'] = [summary['校正'][i] - summary['原始'][i] for i in range(len(summary['指标']))]
        df_summary = pd.DataFrame(summary)
        df_summary['原始'] = df_summary['原始'].apply(lambda x: f'{x:.4f}')
        df_summary['校正'] = df_summary['校正'].apply(lambda x: f'{x:.4f}')
        df_summary['变化'] = df_summary['变化'].apply(lambda x: f'{x:+.4f}')
        
        print('\n📈 CV & NDHDNTL 汇总对比:')
        try:
            from IPython.display import display
            display(df_summary)
        except ImportError:
            print(df_summary.to_string(index=False))
        
        # 显示前N个点位的详细对比
        if show_details > 0:
            print(f'\n📋 各点位详细对比（显示前{show_details}个）:')
            df_display = df_metrics.head(show_details).copy()
            for col in ['CV原始(%)', 'CV校正(%)', 'CV变化(%)']:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.2f}')
            for col in ['NDHDNTL原始', 'NDHDNTL校正', 'NDHDNTL变化']:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}')
            try:
                from IPython.display import display
                display(df_display)
            except ImportError:
                print(df_display.to_string(index=False))
        
        return df_metrics
    
    return None


def plot_comparison_boxplot(results: dict, name: str):
    """
    绘制归一化前后对比的箱线图。
    
    参数:
        results: run_angle_normalization 返回的结果字典
        name: 数据集名称
    """
    import pandas as pd
    
    metrics = results.get('metrics', {})
    if not metrics:
        print('[warn] 无指标数据，无法绘制对比图')
        return
    
    # 构建 DataFrame
    metrics_list = []
    for point_id, m in metrics.items():
        metrics_list.append({
            'CV原始(%)': m['ori_cv'],
            'CV校正(%)': m['fit_cv'],
            'NDHDNTL原始': m['ori_ndhdntl'],
            'NDHDNTL校正': m['fit_ndhdntl']
        })
    df_metrics = pd.DataFrame(metrics_list)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # CV 对比箱线图
    cv_data = [df_metrics['CV原始(%)'].values, df_metrics['CV校正(%)'].values]
    bp1 = axes[0].boxplot(cv_data, labels=['原始', '校正后'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#FF6B6B')
    bp1['boxes'][1].set_facecolor('#4ECDC4')
    axes[0].set_ylabel('CV (%)')
    axes[0].set_title('变异系数(CV)对比')
    axes[0].grid(True, alpha=0.3)
    
    # NDHDNTL 对比箱线图
    ndh_data = [df_metrics['NDHDNTL原始'].values, df_metrics['NDHDNTL校正'].values]
    bp2 = axes[1].boxplot(ndh_data, labels=['原始', '校正后'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#FF6B6B')
    bp2['boxes'][1].set_facecolor('#4ECDC4')
    axes[1].set_ylabel('NDHDNTL')
    axes[1].set_title('动态范围指数(NDHDNTL)对比')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} 角度归一化前后对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_r2_distribution(results: dict, name: str):
    """
    绘制 R² 分布直方图对比。
    
    参数:
        results: run_angle_normalization 返回的结果字典
        name: 数据集名称
    """
    r2_details = results.get('r2_details', [])
    if not r2_details:
        print('[warn] 无 R² 详细数据，无法绘制分布图')
        return
    
    r2_before = [item['r2_polyfit'] for item in r2_details if not np.isnan(item['r2_polyfit'])]
    r2_after = [item['r2_post'] for item in r2_details if not np.isnan(item['r2_post'])]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 处理前 R² 分布
    axes[0].hist(r2_before, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='white')
    axes[0].axvline(np.mean(r2_before), color='red', linestyle='--', 
                    label=f'均值: {np.mean(r2_before):.4f}')
    axes[0].set_xlabel('R2')
    axes[0].set_ylabel('频数')
    axes[0].set_title('处理前 R2 分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 处理后 R² 分布
    axes[1].hist(r2_after, bins=30, color='#4ECDC4', alpha=0.7, edgecolor='white')
    axes[1].axvline(np.mean(r2_after), color='teal', linestyle='--', 
                    label=f'均值: {np.mean(r2_after):.4f}')
    axes[1].set_xlabel('R2')
    axes[1].set_ylabel('频数')
    axes[1].set_title('处理后 R2 分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} R2分布对比（R2越低表示天顶角影响越小）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 便捷接口
# =============================================================================

def run_angle_normalization(input_file: str, output_dir: str, name: str,
                            plot_scatter: bool = False, plot_series: bool = False,
                            verbose: bool = True, show_progress: bool = True,
                            parallel: bool = False, n_workers: int = None,
                            prefilter_3sigma: bool = False,
                            prefilter_sigma: float = 3.0,
                            prefilter_ntl_upper_bound: float = None,
                            min_records_after_filter: int = 10):
    """
    一键运行角度归一化流程。
    
    参数:
        input_file: 输入时间序列文件路径
        output_dir: 输出目录
        name: 数据集名称（用于输出文件命名）
        plot_scatter: 是否绘制散点图
        plot_series: 是否绘制时间序列图
        verbose: 是否打印详细信息
        show_progress: 是否显示 tqdm 进度条
        parallel: 是否使用多进程并行处理（默认 False）
        n_workers: 并行进程数，默认为 CPU 核心数-1
        prefilter_3sigma: 是否在归一化前执行 3-sigma 异常值过滤
        prefilter_sigma: 3-sigma 过滤阈值（默认 3.0）
        prefilter_ntl_upper_bound: 可选 NTL 硬阈值上限（如 1000）
        min_records_after_filter: 过滤后保留点位所需的最少记录数
    
    返回:
        results: 包含输出路径和评价指标的字典
    """
    import os
    import time
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输出路径
    fit_result_path = os.path.join(output_dir, f'{name}_angle.txt')
    output_r2_path = os.path.join(output_dir, f'{name}_visFitResult.txt')
    output_params_path = os.path.join(output_dir, f'{name}_fitParams.txt')
    
    # 读取数据
    if verbose:
        print(f'读取数据: {input_file}')
    pointNumLngLatMap, pointsDic, keyList = readFile(input_file)
    
    if not pointsDic:
        raise ValueError(f'未找到有效数据点（记录数>=10）: {input_file}')

    prefilter_summary = {
        'enabled': False,
        'points_before': int(len(pointsDic)),
        'points_after': int(len(pointsDic)),
        'records_before': int(sum(len(v) for v in pointsDic.values())),
        'records_after': int(sum(len(v) for v in pointsDic.values())),
        'records_removed': 0,
    }

    if prefilter_3sigma:
        pointsDic, pointNumLngLatMap, prefilter_summary = filter_points_outliers_3sigma(
            pointsDic=pointsDic,
            pointNumLngLatMap=pointNumLngLatMap,
            sigma=prefilter_sigma,
            ntl_upper_bound=prefilter_ntl_upper_bound,
            min_records_after_filter=min_records_after_filter,
            verbose=verbose
        )
        if not pointsDic:
            raise ValueError('3-sigma 过滤后无可用点位，请放宽阈值或关闭过滤')
    
    if verbose:
        print(f'有效点位数: {len(pointsDic)}')
        if parallel:
            cpu_count = multiprocessing.cpu_count()
            actual_workers = n_workers if n_workers else max(1, cpu_count - 1)
            print(f'使用并行处理: {actual_workers} 进程 (CPU核心数: {cpu_count})')
    
    start_time = time.time()
    stats_collector = []
    
    if parallel:
        # 并行处理
        stats_collector = visScatterAndFitCurve_parallel(
            pointsDic, output_r2_path, fit_result_path,
            pointNumLngLatMap, output_params_path,
            n_workers=n_workers, show_progress=show_progress,
            verbose=verbose
        )
    else:
        # 串行处理（原始方式）
        progress = None
        try:
            if show_progress:
                try:
                    from tqdm import tqdm
                    progress = tqdm(total=len(pointsDic), desc='AngleNorm', ncols=100)
                except Exception as exc:
                    if verbose:
                        print(f'[warn] 无法初始化进度条: {exc}')
                    progress = None

            visScatterAndFitCurve(
                pointsDic, output_r2_path, fit_result_path,
                pointNumLngLatMap, output_params_path,
                plot_scatter=plot_scatter, verbose=verbose,
                progress=progress, stats_collector=stats_collector
            )
        finally:
            if progress is not None:
                progress.close()
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f'处理耗时: {elapsed_time:.2f} 秒')
    
    # 读取校正结果并计算指标
    pointNumLngLatMap_fit, pointsDic_fit, _ = readFile(fit_result_path)
    metrics = visTimeSeries(pointsDic, pointsDic_fit, name,
                           plot_series=plot_series, verbose=verbose)

    def _build_r2_summary(values):
        values = [v for v in values if not np.isnan(v)]
        if not values:
            return {}
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }

    r2_summary_before = _build_r2_summary([item['r2_polyfit'] for item in stats_collector])
    r2_summary_after = _build_r2_summary([item['r2_post'] for item in stats_collector])

    if verbose:
        if r2_summary_before:
            print('[info] R2（处理前）: ' + ', '.join([
                f'均值={r2_summary_before["mean"]:.4f}',
                f'中位数={r2_summary_before["median"]:.4f}',
                f'IQR=({r2_summary_before["p25"]:.4f}, {r2_summary_before["p75"]:.4f})',
                f'范围=({r2_summary_before["min"]:.4f}, {r2_summary_before["max"]:.4f})'
            ]))
        if r2_summary_after:
            print('[info] R2（处理后）: ' + ', '.join([
                f'均值={r2_summary_after["mean"]:.4f}',
                f'中位数={r2_summary_after["median"]:.4f}',
                f'IQR=({r2_summary_after["p25"]:.4f}, {r2_summary_after["p75"]:.4f})',
                f'范围=({r2_summary_after["min"]:.4f}, {r2_summary_after["max"]:.4f})'
            ]))
    
    return {
        'input_file': input_file,
        'fit_result_path': fit_result_path,
        'output_r2_path': output_r2_path,
        'output_params_path': output_params_path,
        'num_points': len(pointsDic),
        'prefilter_summary': prefilter_summary,
        'metrics': metrics,
        'r2_summary': {
            'before': r2_summary_before,
            'after': r2_summary_after
        },
        'r2_details': stats_collector
    }


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    import os
    
    # 示例配置
    PLOT_SCATTER = False
    PLOT_SERIES = False
    
    name_list = ['ntl_timeseries']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'processed')
    output_dir = os.path.join(base_dir, 'output')
    
    for name in name_list:
        input_file = os.path.join(input_dir, f'{name}.txt')
        
        if not os.path.exists(input_file):
            print(f'文件不存在, 请检查: {input_file}')
            continue
        
        try:
            results = run_angle_normalization(
                input_file=input_file,
                output_dir=output_dir,
                name=name,
                plot_scatter=PLOT_SCATTER,
                plot_series=PLOT_SERIES
            )
            print(f'\n处理完成: {name}')
            print(f'  校正结果: {results["fit_result_path"]}')
            print(f'  有效点位: {results["num_points"]}')
        except Exception as e:
            print(f'处理失败: {name} -> {e}')
    
    print('\n全部处理完成。')
