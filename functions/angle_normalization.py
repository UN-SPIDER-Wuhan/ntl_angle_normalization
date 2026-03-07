# -*- coding: utf-8 -*-
"""
Angle normalization module (Gradient Descent / Least Squares).

This module provides sensor zenith angle normalization for nighttime light data,
including:
- data loading and parsing
- goodness-of-fit and correlation metrics
- the core angle-normalization algorithm
- time-series visualization and indicator calculation
- parallel processing support

Example:
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
# Data loading functions
# =============================================================================

def readFile(filePath: str, min_records: int = 10):
    """
    Read a time-series text file and parse it into dictionary form.

    File format:
        pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...

    Parameters:
        filePath: input file path
        min_records: minimum number of records required per point (default: 10)

    Returns:
        pointNumLngLatMap: {pointNum: "lng,lat"} point coordinate mapping
        pointsDic: {pointNum: [[NTLValue, Zenith, YYYYMMDD], ...]} time-series data
        keyList: list of all point IDs, including filtered ones
    """
    with open(filePath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pointNumLngLatMap = {}
    pointsDic = {}
    keyList = []
    
    for i, line in enumerate(lines):
        if i == 0:  # Skip the header line
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
    
    # Filter out points with too few records
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
    Apply point-wise outlier filtering before normalization (3-sigma plus an optional hard threshold).

    Rules:
    1. Filter each point's NTL series with |x - mean| <= sigma * std (skip 3-sigma when std = 0)
    2. If ntl_upper_bound is provided, also filter records where NTL exceeds that upper limit
    3. Drop points with fewer than min_records_after_filter records after filtering

    Returns:
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

        # 3-sigma filtering
        mean_v = float(np.mean(ntl_vals))
        std_v = float(np.std(ntl_vals))
        if std_v > 0:
            sigma_mask = np.abs(ntl_vals - mean_v) <= sigma * std_v
            removed_by_sigma += int((~sigma_mask).sum())
            keep_mask &= sigma_mask

        # Optional hard-threshold filtering
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
        print('[info] Pre-normalization outlier filtering (3-sigma) completed: ' +
              f"points {summary['points_before']} -> {summary['points_after']}，" +
              f"records {summary['records_before']} -> {summary['records_after']}")

    return filtered_points, filtered_map, summary


# =============================================================================
# Statistical and fitting helper functions
# =============================================================================

def calGoodnessOfFit(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the goodness of fit R² (coefficient of determination).

    Parameters:
        y_pred: predicted values
        y_true: observed values

    Returns:
        R² in the range (-∞, 1], where values closer to 1 indicate a better fit
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
    Compute the Pearson correlation coefficient.

    Parameters:
        x_array: X array
        y_array: Y array
    
    Returns:
        Correlation coefficient in the range [-1, 1]; returns np.nan when either standard deviation is zero
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
    """Quadratic function: f(x) = a*x² + b*x + c"""
    a, b, c = p
    return a * x ** 2 + b * x + c


def _func_quadratic_c1(p, x):
    """Quadratic function with intercept fixed at 1: f(x) = a*x² + b*x + 1"""
    a, b = p
    return a * x ** 2 + b * x + 1


def _error_quadratic(p, x, y):
    """Mean squared error of the quadratic fit"""
    return (1 / len(x)) * (_func_quadratic(p, x) - y) ** 2


def _error_quadratic_c1(p, x, y):
    """Mean squared error of the quadratic fit with intercept fixed at 1"""
    return (1 / len(x)) * (_func_quadratic_c1(p, x) - y) ** 2


# =============================================================================
# Single-point normalization function for parallel execution
# =============================================================================

def _normalize_single_point(args):
    """
    Normalize a single point's angle effect for parallel execution.

    Parameters:
        args: tuple of (key, X, Y, time_values)

    Returns:
        dict containing point information, normalized values, and statistics
    """
    key, X, Y, time_values = args
    
    X = np.array(X)
    Y = np.array(Y)
    mean_ntl = float(np.mean(Y)) if len(Y) else 0.0
    
    try:
        # Step 1: Initial polynomial fit parameters
        parameters = np.polyfit(X, Y, 2)
        
        # Step 2: Least-squares optimization
        parameter_init = np.array([0, 0, 0])
        Para = leastsq(_error_quadratic, parameter_init, args=(X, Y))
        a, b, c = Para[0]
        
        # Compute fit metrics
        fit_y = parameters[0] * X ** 2 + parameters[1] * X + parameters[2]
        GoodnessOfFit_score = calGoodnessOfFit(fit_y, Y)
        r, p_value = stats.pearsonr(fit_y, Y)
        
        fit_y_leastsq = a * X ** 2 + b * X + c
        GoodnessOfFit_score_leastsq = calGoodnessOfFit(fit_y_leastsq, Y)
        
        # Step 3: Optimize normalization parameters
        Z = X
        R = Y
        
        # Prevent division by zero
        base_c = parameters[2] if parameters[2] != 0 else 1e-6
        p0 = np.array([parameters[0] / base_c, parameters[1] / base_c, 0, 0, base_c])
        
        # Optimization objective: maximize the correlation between corrected data and the fitted trend
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
        
        # Step 4: Apply normalization
        normalizationY = R / (p_final[0] * (Z**2) + p_final[1] * Z + np.ones(len(Z)))
        
        # Handle negative values by keeping the original value
        negative_mask = normalizationY < 0
        normalizationY[negative_mask] = R[negative_mask]
        
        # Re-evaluate correlation with zenith angle after normalization (lower is better)
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
# Core angle-normalization function
# =============================================================================

def normalizationZenith(visX_dic: dict, visY_dic: dict, 
                        outputFile: str, output_params_path: str,
                        plot_scatter: bool = False, verbose: bool = True,
                        progress=None, stats_collector=None):
    """
    Perform angle normalization for each point.

    Workflow:
    1. Fit a quadratic polynomial to each point's (Zenith, NTL) data.
    2. Refine the parameters with least squares.
    3. Further optimize the objective by maximizing the correlation between corrected values and the trend.
    4. Apply the normalization formula: NTL_corrected = NTL / (a*Z² + b*Z + 1)

    Parameters:
        visX_dic: {pointNum: [Zenith values]} zenith-angle data
        visY_dic: {pointNum: [NTL values]} nighttime-light brightness data
        outputFile: output path for fit metrics (records R²)
        output_params_path: output path for fitted parameters
        plot_scatter: whether to plot scatter charts (default False)
        verbose: whether to print verbose logs (default True)

    Returns:
        normalizationResult_array: list of normalized NTL arrays
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
            
            # Step 1: Initial polynomial fit parameters
            parameters = np.polyfit(X, Y, 2)
            
            # Step 2: Least-squares optimization
            parameter_init = np.array([0, 0, 0])
            Para = leastsq(_error_quadratic, parameter_init, args=(X, Y))
            a, b, c = Para[0]
            
            if verbose:
                print(f'{key} | polyfit a={parameters[0]:.5f} b={parameters[1]:.5f} c={parameters[2]:.5f}')
            
            # Compute fit metrics
            fit_y = parameters[0] * X ** 2 + parameters[1] * X + parameters[2]
            GoodnessOfFit_score = calGoodnessOfFit(fit_y, Y)
            r, p_value = stats.pearsonr(fit_y, Y)
            
            fit_y_leastsq = a * X ** 2 + b * X + c
            GoodnessOfFit_score_leastsq = calGoodnessOfFit(fit_y_leastsq, Y)
            
            if verbose:
                print(f'R2 polyfit: {GoodnessOfFit_score:.4f} | Pearson r²: {r**2:.4f} p: {p_value:.4e}')
                print(f'R2 leastsq: {GoodnessOfFit_score_leastsq:.4f}')

            # Re-evaluate correlation with zenith angle after normalization (lower is better)
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
            
            # Step 3: Optimize normalization parameters
            Z = X
            R = Y
            
            # Prevent division by zero
            base_c = parameters[2] if parameters[2] != 0 else 1e-6
            p0 = np.array([parameters[0] / base_c, parameters[1] / base_c, 0, 0, base_c])
            
            # Optimization objective: maximize the correlation between corrected data and the fitted trend
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
            
            # Step 4: Apply normalization
            normalizationY = R / (p_final[0] * (Z**2) + p_final[1] * Z + np.ones(len(Z)))
            
            # Handle negative values by keeping the original value
            negative_mask = normalizationY < 0
            normalizationY[negative_mask] = R[negative_mask]
            
            normalizationResult_array.append(normalizationY)
            
            # Write results
            f.write(f'{key}:{r ** 2}\n')
            f1.write(f'{key}:{",".join([str(x) for x in p_final])}\n')
            
            # Optional plotting
            if plot_scatter:
                _plot_scatter_fit(key, X, Y, normalizationY, parameters)

            if progress is not None:
                before_vals = [item['r2_polyfit'] for item in stats_collector if not np.isnan(item['r2_polyfit'])]
                after_vals = [item['r2_post'] for item in stats_collector if not np.isnan(item['r2_post'])]
                avg_before = sum(before_vals) / len(before_vals) if before_vals else 0.0
                avg_after = sum(after_vals) / len(after_vals) if after_vals else 0.0
                progress.set_postfix({
                    'point': key,
                    'mean': f'{mean_ntl:.2f}',
                    'R2_before': f'{GoodnessOfFit_score:.3f}',
                    'R2_after': f'{r2_post:.3f}' if not np.isnan(r2_post) else 'nan',
                    'avg_before': f'{avg_before:.3f}',
                    'avg_after': f'{avg_after:.3f}'
                })
                progress.update(1)
    
    return normalizationResult_array


# =============================================================================
# Parallel normalization processing
# =============================================================================

def _normalize_batch(batch_args):
    """
    Process multiple points in one batch to reduce inter-process communication overhead.
    """
    results = []
    for args in batch_args:
        results.append(_normalize_single_point(args))
    return results


def normalizationZenith_parallel(visX_dic: dict, visY_dic: dict, visTime_dic: dict,
                                  n_workers: int = None, show_progress: bool = True,
                                  verbose: bool = False, batch_size: int = None):
    """
    Perform angle normalization in parallel with multiple processes.

    Parameters:
        visX_dic: {pointNum: [Zenith values]} zenith-angle data
        visY_dic: {pointNum: [NTL values]} nighttime-light data
        visTime_dic: {pointNum: [Date values]} timestamp data
        n_workers: number of worker processes, defaulting to CPU core count
        show_progress: whether to show a progress bar
        verbose: whether to print verbose logs
        batch_size: number of points per batch; computed automatically by default

    Returns:
        results_dict: {key: result} result dictionary
        stats_collector: list of statistics
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Prepare the task list
    tasks = []
    for key in visX_dic:
        tasks.append((key, visX_dic[key], visY_dic[key], visTime_dic[key]))
    
    total_points = len(tasks)
    
    # Automatically determine batch size: at least 50 points per process to reduce communication overhead
    if batch_size is None:
        batch_size = max(50, total_points // (n_workers * 4))
    
    # Split tasks into batches
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    
    results_dict = {}
    stats_collector = []
    failed_count = 0
    processed_count = 0
    
    # Process batches in parallel using a process pool
    progress = None
    try:
        if show_progress:
            try:
                from tqdm import tqdm
                progress = tqdm(total=total_points, 
                               desc=f'AngleNorm(parallel x{n_workers}, batch={batch_size})', 
                               ncols=110)
            except ImportError:
                progress = None
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all batch tasks
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
                        # Update progress only when an entire batch is finished
                        before_vals = [s['r2_polyfit'] for s in stats_collector if not np.isnan(s['r2_polyfit'])]
                        after_vals = [s['r2_post'] for s in stats_collector if not np.isnan(s['r2_post'])]
                        avg_before = sum(before_vals) / len(before_vals) if before_vals else 0.0
                        avg_after = sum(after_vals) / len(after_vals) if after_vals else 0.0
                        progress.update(len(batch_results))
                        progress.set_postfix({
                            'failed': failed_count,
                            'R2_before': f'{avg_before:.3f}',
                            'R2_after': f'{avg_after:.3f}'
                        })
                        
                except Exception as e:
                    batch_size_actual = len(batches[batch_idx])
                    failed_count += batch_size_actual
                    if verbose:
                        print(f'[error] Batch processing error: batch_{batch_idx} -> {e}')
                    if progress is not None:
                        progress.update(batch_size_actual)
    
    finally:
        if progress is not None:
            progress.close()
    
    if verbose:
        print(f'[info] Parallel processing completed: succeeded {len(results_dict) - failed_count}, failed {failed_count}')
    
    return results_dict, stats_collector


def visScatterAndFitCurve_parallel(result_dic: dict, outputFile: str, fit_result_path: str,
                                    pointNumLngLatMap: dict, output_params_path: str,
                                    n_workers: int = None, show_progress: bool = True,
                                    verbose: bool = False):
    """
    Run angle normalization in parallel and write the output files.

    Parameters:
        result_dic: pointsDic returned by readFile
        outputFile: output path for fit metrics (R² records)
        fit_result_path: output path for corrected time series
        pointNumLngLatMap: point coordinate mapping
        output_params_path: output path for fitted parameters
        n_workers: number of worker processes
        show_progress: whether to show a progress bar
        verbose: whether to print verbose logs

    Returns:
        stats_collector: list of statistics
    """
    # Organize input data
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
    
    # Run normalization in parallel
    results_dict, stats_collector = normalizationZenith_parallel(
        visX_dic, visY_dic, visTime_dic,
        n_workers=n_workers, show_progress=show_progress, verbose=verbose
    )
    
    # Write result files
    with open(fit_result_path, 'w', encoding='utf-8') as f1, \
         open(outputFile, 'w', encoding='utf-8') as f2, \
         open(output_params_path, 'w', encoding='utf-8') as f3:
        
        f1.write('pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...\n')
        
        # Write records in their original order
        for key in visX_dic:
            if key not in results_dict or not results_dict[key]['success']:
                continue
            
            result = results_dict[key]
            normalizationY = result['normalizationY']
            X = result['X']
            time_values = result['time_values']
            
            # Write the corrected time series
            f1.write(f'{key}:{pointNumLngLatMap[key]}:')
            entries = []
            for k in range(len(normalizationY)):
                entries.append(f'{time_values[k]},{X[k]},{normalizationY[k]}')
            f1.write(';'.join(entries))
            f1.write('\n')
            
            # Write R² values and fitted parameters
            f2.write(f'{key}:{result["r_squared"]}\n')
            f3.write(f'{key}:{",".join([str(x) for x in result["p_final"]])}\n')
    
    return stats_collector


def _plot_scatter_fit(key: str, X: np.ndarray, Y: np.ndarray, 
                      Y_corrected: np.ndarray, parameters: np.ndarray):
    """Plot scatter points and the fitted curve"""
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
        print(f'[warn] Failed to plot scatter chart: {key} -> {e}')


# =============================================================================
# Fit and write corrected results
# =============================================================================

def visScatterAndFitCurve(result_dic: dict, outputFile: str, fit_result_path: str,
                          pointNumLngLatMap: dict, output_params_path: str,
                          plot_scatter: bool = False, verbose: bool = True,
                          progress=None, stats_collector=None):
    """
    Run angle normalization and write the corrected time-series results.

    Parameters:
        result_dic: pointsDic returned by readFile
        outputFile: output path for fit metrics (R² records)
        fit_result_path: output path for corrected time series
        pointNumLngLatMap: point coordinate mapping
        output_params_path: output path for fitted parameters
        plot_scatter: whether to draw scatter plots
        verbose: whether to print verbose logs
    """
    with open(fit_result_path, 'w', encoding='utf-8') as f1:
        f1.write('pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...\n')
        
        # Organize input data
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
        
        # Execute normalization
        normalizationResult_array = normalizationZenith(
            visX_dic, visY_dic, outputFile, output_params_path,
            plot_scatter=plot_scatter, verbose=verbose,
            progress=progress, stats_collector=stats_collector
        )
        
        # Write corrected results
        for norm_index, key in enumerate(visX_dic):
            f1.write(f'{key}:{pointNumLngLatMap[key]}:')
            entries = []
            for k in range(len(normalizationResult_array[norm_index])):
                entries.append(f'{visTime_dic[key][k]},{visX_dic[key][k]},{normalizationResult_array[norm_index][k]}')
            f1.write(';'.join(entries))
            f1.write('\n')


# =============================================================================
# Time-series visualization and metric computation
# =============================================================================

def visTimeSeries(pointsDic: dict, pointsDic_fit: dict, area: str,
                  plot_series: bool = True, verbose: bool = True):
    """
    Visualize original and corrected time series and compute evaluation metrics.

    Metrics:
    - NDHDNTL: (max - min) / (max + min), describing dynamic range
    - CV: coefficient of variation = std / mean * 100, describing fluctuation

    Parameters:
        pointsDic: original data dictionary
        pointsDic_fit: corrected data dictionary
        area: area name used in chart titles
        plot_series: whether to draw time-series plots
        verbose: whether to print metrics

    Returns:
        metrics: {pointNum: {ori_ndhdntl, fit_ndhdntl, ori_cv, fit_cv}}
    """
    # Organize the original data
    pointsDic_sorted = {}
    for key in pointsDic:
        pointsDic_sorted[key] = []
        for item in pointsDic[key]:
            if item[1]:  # Valid zenith value
                pointsDic_sorted[key].append([item[0], item[1], int(item[2])])
        pointsDic_sorted[key].sort(key=lambda x: x[2])
    
    visNTL_dic = {}
    visTime_dic = {}
    for key in pointsDic_sorted:
        visNTL_dic[key] = [row[0] for row in pointsDic_sorted[key]]
        visTime_dic[key] = [str(row[2]) for row in pointsDic_sorted[key]]
    
    # Organize corrected data aligned to the original timestamps
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
    
    # Compute metrics and generate visualizations
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
    """Plot a time-series comparison chart"""
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
        print(f'[warn] Failed to plot time series: {key} -> {e}')


# =============================================================================
# Visualization comparing results before and after normalization
# =============================================================================

def display_comparison_results(results: dict, name: str, show_details: int = 10):
    """
    Display comparison results before and after normalization.

    Parameters:
        results: result dictionary returned by run_angle_normalization
        name: dataset name
        show_details: number of points to show in the detailed comparison (default 10)
    """
    import pandas as pd
    
    print(f'\n{"="*50}')
    print('📊 Comparison before and after normalization')
    print(f'{"="*50}')
    
    # 1. R² statistical comparison
    r2_summary = results.get('r2_summary', {})
    if r2_summary.get('before') and r2_summary.get('after'):
        before = r2_summary['before']
        after = r2_summary['after']
        
        comparison_data = {
            'Metric': ['Mean', 'Median', '25th percentile', '75th percentile', 'Minimum', 'Maximum'],
            'R² (before)': [before['mean'], before['median'], before['p25'], before['p75'], before['min'], before['max']],
            'R² (after)': [after['mean'], after['median'], after['p25'], after['p75'], after['min'], after['max']]
        }
        df_r2 = pd.DataFrame(comparison_data)
        df_r2['Change'] = df_r2['R² (after)'] - df_r2['R² (before)']
        df_r2['R² (before)'] = df_r2['R² (before)'].apply(lambda x: f'{x:.4f}')
        df_r2['R² (after)'] = df_r2['R² (after)'].apply(lambda x: f'{x:.4f}')
        df_r2['Change'] = df_r2['Change'].apply(lambda x: f'{x:+.4f}')
        
        print('\n🔢 R² summary comparison (lower R² indicates weaker zenith-angle impact):')
        try:
            from IPython.display import display
            display(df_r2)
        except ImportError:
            print(df_r2.to_string(index=False))
    
    # 2. CV and NDHDNTL metric comparison
    metrics = results.get('metrics', {})
    if metrics:
        metrics_list = []
        for point_id, m in metrics.items():
            metrics_list.append({
                'Point': point_id,
                'CV original (%)': m['ori_cv'],
                'CV corrected (%)': m['fit_cv'],
                'CV change (%)': m['fit_cv'] - m['ori_cv'],
                'NDHDNTL original': m['ori_ndhdntl'],
                'NDHDNTL corrected': m['fit_ndhdntl'],
                'NDHDNTL change': m['fit_ndhdntl'] - m['ori_ndhdntl']
            })
        
        df_metrics = pd.DataFrame(metrics_list)
        
        # Compute summary statistics
        summary = {
            'Metric': ['Mean CV (%)', 'Mean NDHDNTL'],
            'Original': [df_metrics['CV original (%)'].mean(), df_metrics['NDHDNTL original'].mean()],
            'Corrected': [df_metrics['CV corrected (%)'].mean(), df_metrics['NDHDNTL corrected'].mean()],
        }
        summary['Change'] = [summary['Corrected'][i] - summary['Original'][i] for i in range(len(summary['Metric']))]
        df_summary = pd.DataFrame(summary)
        df_summary['Original'] = df_summary['Original'].apply(lambda x: f'{x:.4f}')
        df_summary['Corrected'] = df_summary['Corrected'].apply(lambda x: f'{x:.4f}')
        df_summary['Change'] = df_summary['Change'].apply(lambda x: f'{x:+.4f}')
        
        print('\n📈 CV and NDHDNTL summary comparison:')
        try:
            from IPython.display import display
            display(df_summary)
        except ImportError:
            print(df_summary.to_string(index=False))
        
        # Display detailed comparisons for the first N points
        if show_details > 0:
            print(f'\n📋 Detailed comparison by point (showing the first {show_details}):')
            df_display = df_metrics.head(show_details).copy()
            for col in ['CV original (%)', 'CV corrected (%)', 'CV change (%)']:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.2f}')
            for col in ['NDHDNTL original', 'NDHDNTL corrected', 'NDHDNTL change']:
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
    Draw boxplots comparing metrics before and after normalization.

    Parameters:
        results: result dictionary returned by run_angle_normalization
        name: dataset name
    """
    import pandas as pd
    
    metrics = results.get('metrics', {})
    if not metrics:
        print('[warn] No metric data available; cannot draw comparison plots')
        return
    
    # Build the DataFrame
    metrics_list = []
    for point_id, m in metrics.items():
        metrics_list.append({
            'CV original (%)': m['ori_cv'],
            'CV corrected (%)': m['fit_cv'],
            'NDHDNTL original': m['ori_ndhdntl'],
            'NDHDNTL corrected': m['fit_ndhdntl']
        })
    df_metrics = pd.DataFrame(metrics_list)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # CV comparison boxplot
    cv_data = [df_metrics['CV original (%)'].values, df_metrics['CV corrected (%)'].values]
    bp1 = axes[0].boxplot(cv_data, labels=['Original', 'Corrected'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#FF6B6B')
    bp1['boxes'][1].set_facecolor('#4ECDC4')
    axes[0].set_ylabel('CV (%)')
    axes[0].set_title('Coefficient of Variation (CV) comparison')
    axes[0].grid(True, alpha=0.3)
    
    # NDHDNTL comparison boxplot
    ndh_data = [df_metrics['NDHDNTL original'].values, df_metrics['NDHDNTL corrected'].values]
    bp2 = axes[1].boxplot(ndh_data, labels=['Original', 'Corrected'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#FF6B6B')
    bp2['boxes'][1].set_facecolor('#4ECDC4')
    axes[1].set_ylabel('NDHDNTL')
    axes[1].set_title('Dynamic range index (NDHDNTL) comparison')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} comparison before and after angle normalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_r2_distribution(results: dict, name: str):
    """
    Draw histogram comparisons of the R² distribution.

    Parameters:
        results: result dictionary returned by run_angle_normalization
        name: dataset name
    """
    r2_details = results.get('r2_details', [])
    if not r2_details:
        print('[warn] No detailed R² data available; cannot draw distribution plots')
        return
    
    r2_before = [item['r2_polyfit'] for item in r2_details if not np.isnan(item['r2_polyfit'])]
    r2_after = [item['r2_post'] for item in r2_details if not np.isnan(item['r2_post'])]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # R² distribution before processing
    axes[0].hist(r2_before, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='white')
    axes[0].axvline(np.mean(r2_before), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(r2_before):.4f}')
    axes[0].set_xlabel('R2')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('R2 distribution before processing')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R² distribution after processing
    axes[1].hist(r2_after, bins=30, color='#4ECDC4', alpha=0.7, edgecolor='white')
    axes[1].axvline(np.mean(r2_after), color='teal', linestyle='--', 
                    label=f'Mean: {np.mean(r2_after):.4f}')
    axes[1].set_xlabel('R2')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('R2 distribution after processing')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} R2 distribution comparison (lower R2 indicates weaker zenith-angle impact)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Convenience interface
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
    Run the full angle-normalization workflow in one call.

    Parameters:
        input_file: input time-series file path
        output_dir: output directory
        name: dataset name used for output file naming
        plot_scatter: whether to draw scatter plots
        plot_series: whether to draw time-series plots
        verbose: whether to print verbose logs
        show_progress: whether to show a tqdm progress bar
        parallel: whether to use multiprocessing (default False)
        n_workers: number of worker processes, defaulting to CPU cores minus one
        prefilter_3sigma: whether to run 3-sigma outlier filtering before normalization
        prefilter_sigma: sigma threshold for 3-sigma filtering (default 3.0)
        prefilter_ntl_upper_bound: optional hard upper bound for NTL values (for example 1000)
        min_records_after_filter: minimum records required to keep a point after filtering

    Returns:
        results: dictionary containing output paths and evaluation metrics
    """
    import os
    import time
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    fit_result_path = os.path.join(output_dir, f'{name}_angle.txt')
    output_r2_path = os.path.join(output_dir, f'{name}_visFitResult.txt')
    output_params_path = os.path.join(output_dir, f'{name}_fitParams.txt')
    
    # Read input data
    if verbose:
        print(f'Reading data: {input_file}')
    pointNumLngLatMap, pointsDic, keyList = readFile(input_file)
    
    if not pointsDic:
        raise ValueError(f'No valid data points found (records >= 10): {input_file}')

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
            raise ValueError('No usable points remain after 3-sigma filtering; relax the threshold or disable filtering')
    
    if verbose:
        print(f'Valid points: {len(pointsDic)}')
        if parallel:
            cpu_count = multiprocessing.cpu_count()
            actual_workers = n_workers if n_workers else max(1, cpu_count - 1)
            print(f'Using parallel processing: {actual_workers} workers (CPU cores: {cpu_count})')
    
    start_time = time.time()
    stats_collector = []
    
    if parallel:
        # Parallel processing
        stats_collector = visScatterAndFitCurve_parallel(
            pointsDic, output_r2_path, fit_result_path,
            pointNumLngLatMap, output_params_path,
            n_workers=n_workers, show_progress=show_progress,
            verbose=verbose
        )
    else:
        # Serial processing (original behavior)
        progress = None
        try:
            if show_progress:
                try:
                    from tqdm import tqdm
                    progress = tqdm(total=len(pointsDic), desc='AngleNorm', ncols=100)
                except Exception as exc:
                    if verbose:
                        print(f'[warn] Failed to initialize progress bar: {exc}')
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
        print(f'Elapsed time: {elapsed_time:.2f} s')
    
    # Read corrected results and compute metrics
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
            print('[info] R2 (before): ' + ', '.join([
                f'mean={r2_summary_before["mean"]:.4f}',
                f'median={r2_summary_before["median"]:.4f}',
                f'IQR=({r2_summary_before["p25"]:.4f}, {r2_summary_before["p75"]:.4f})',
                f'range=({r2_summary_before["min"]:.4f}, {r2_summary_before["max"]:.4f})'
            ]))
        if r2_summary_after:
            print('[info] R2 (after): ' + ', '.join([
                f'mean={r2_summary_after["mean"]:.4f}',
                f'median={r2_summary_after["median"]:.4f}',
                f'IQR=({r2_summary_after["p25"]:.4f}, {r2_summary_after["p75"]:.4f})',
                f'range=({r2_summary_after["min"]:.4f}, {r2_summary_after["max"]:.4f})'
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
# Main entry point
# =============================================================================

if __name__ == '__main__':
    import os
    
    # Example configuration
    PLOT_SCATTER = False
    PLOT_SERIES = False
    
    name_list = ['ntl_timeseries']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'processed')
    output_dir = os.path.join(base_dir, 'output')
    
    for name in name_list:
        input_file = os.path.join(input_dir, f'{name}.txt')
        
        if not os.path.exists(input_file):
            print(f'File does not exist, please check: {input_file}')
            continue
        
        try:
            results = run_angle_normalization(
                input_file=input_file,
                output_dir=output_dir,
                name=name,
                plot_scatter=PLOT_SCATTER,
                plot_series=PLOT_SERIES
            )
            print(f'\nProcessing completed: {name}')
            print(f'  Corrected output: {results["fit_result_path"]}')
            print(f'  Valid points: {results["num_points"]}')
        except Exception as e:
            print(f'Processing failed: {name} -> {e}')
    
    print('\nAll processing completed.')
