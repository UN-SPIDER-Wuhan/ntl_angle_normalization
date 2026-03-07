from datetime import datetime
import numpy as np
import pandas as pd
import logging
import warnings

"""
Compatibility shims for NumPy 1.20+ where aliases like np.int/np.float were removed.
Some older dependencies (e.g., fbprophet) still reference these names.
This ensures runtime compatibility without requiring users to pin old NumPy.
"""
# Use a warnings filter to avoid FutureWarning when checking hasattr
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
try:
    # Prefer the modern package name first
    from prophet import Prophet  # type: ignore
    _PROPHET_LOGGER_NAME = "prophet"
except Exception:
    # Fall back to the legacy package name
    from fbprophet import Prophet  # type: ignore
    _PROPHET_LOGGER_NAME = "fbprophet"
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# tqdm progress bar fallback (use a no-op implementation if unavailable)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


class PSO(object):
    """Particle swarm optimization for searching three key Prophet parameters:
    [seasonality_prior_scale, changepoint_prior_scale, n_changepoints]

    The fitness function is the mean absolute validation error |y - yhat|
    computed at the provided validation indices.
    """

    def __init__(
        self,
        particle_num: int,
        particle_dim: int,
        iter_num: int,
        c1: float,
        c2: float,
        w: float,
        max_value: float,
        min_value: float,
        df: pd.DataFrame,
        fdays: int,
        testIndex_list_allDst: List[int],
        base_params: Dict[str, Any],
        enable_plot: bool = False,
        quiet: bool = False,
        progress: bool = True,
    ):
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.max_value = max_value
        self.min_value = min_value
        self.df = df
        self.fdays = fdays
        self.testIndex_list_allDst = testIndex_list_allDst
        self.base_params = base_params
        self.enable_plot = enable_plot
        self.quiet = quiet
        self.progress = progress

    def swarm_origin(self) -> Tuple[List[List[float]], List[List[float]]]:
        particle_loc: List[List[float]] = []
        particle_dir: List[List[float]] = []
        for _ in range(self.particle_num):
            tmp1 = []
            tmp2 = []
            for _ in range(self.particle_dim):
                a = random.random()
                b = random.random()
                tmp1.append(a * (self.max_value - self.min_value) + self.min_value)
                tmp2.append(b)
            particle_loc.append(tmp1)
            particle_dir.append(tmp2)
        return particle_loc, particle_dir

    def fitness(self, particle_loc: List[List[float]]):
        fitness_value: List[float] = []
        for i in range(self.particle_num):
            error_value = find_params(
                self.df,
                self.fdays,
                self.testIndex_list_allDst,
                particle_loc[i][0],
                particle_loc[i][1],
                particle_loc[i][2],
                self.base_params,
            )
            fitness_value.append(error_value)
        current_fitness = float("inf")
        current_parameter: List[float] = []
        for i in range(self.particle_num):
            if current_fitness > fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]
        return fitness_value, current_fitness, current_parameter

    def updata(
        self,
        particle_loc: List[List[float]],
        particle_dir: List[List[float]],
        gbest_parameter: List[float],
        pbest_parameters: List[List[float]],
    ):
        for i in range(self.particle_num):
            a1 = [x * self.w for x in particle_dir[i]]
            a2 = [y * self.c1 * random.random() for y in list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
            a3 = [z * self.c2 * random.random() for z in list(np.array(gbest_parameter) - np.array(particle_dir[i]))]
            particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
            particle_loc[i] = list(np.array(particle_loc[i]) + np.array(particle_dir[i]))

        parameter_list: List[List[float]] = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)

        value: List[List[float]] = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)

        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                # Linearly normalize values back into [min_value, max_value]
                denom = (value[j][0] - value[j][1]) or 1.0
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1]) / denom * (self.max_value - self.min_value) + self.min_value

        return particle_loc, particle_dir

    def plot(self, results: List[float]):
        X = list(range(1, self.iter_num + 1))
        Y = [results[i] for i in range(self.iter_num)]
        if plt is not None:
            plt.plot(X, Y)
            plt.xlabel('Number of iteration', size=12)
            plt.ylabel('Value of Error', size=12)
            plt.title('PSO parameter optimization')
            plt.show()

    def main(self) -> List[float]:
        results: List[float] = []
        best_fitness = float("inf")
        particle_loc, particle_dir = self.swarm_origin()
        gbest_parameter = [0.0 for _ in range(self.particle_dim)]
        pbest_parameters = [[0.0 for _ in range(self.particle_dim)] for _ in range(self.particle_num)]
        fitness_value = [float("inf") for _ in range(self.particle_num)]

        iter_range = range(self.iter_num)
        if self.progress:
            iter_range = tqdm(iter_range, desc="PSO", leave=False, total=self.iter_num)

        for i in iter_range:
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(particle_loc)
            for j in range(self.particle_num):
                if current_fitness_value[j] < fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter
            if not self.quiet:
                print('iteration:', (i + 1 if isinstance(i, int) else i), '; best params:', gbest_parameter, '; loss:', best_fitness)
            results.append(best_fitness)
            fitness_value = current_fitness_value
            particle_loc, particle_dir = self.updata(particle_loc, particle_dir, gbest_parameter, pbest_parameters)

        if self.enable_plot:
            self.plot(results)
        if not self.quiet:
            print('Final parameters:', gbest_parameter)
        return gbest_parameter


def readFile(filePath: str, min_obs_per_point: int) -> Tuple[Dict[str, str], Dict[str, List[List[Any]]]]:
    with open(filePath, 'r') as f:
        lines = f.readlines()
    pointNumLngLatMap: Dict[str, str] = {}
    pointsDic: Dict[str, List[List[Any]]] = {}
    for i in range(len(lines)):
        if i == 0:
            continue
        temp_list = lines[i].split(":")
        pointNum = temp_list[0]
        pointLngLat = temp_list[1]
        pointsDic[pointNum] = []
        pointNumLngLatMap[pointNum] = pointLngLat
        info_list = temp_list[2].split(";")
        for k in range(len(info_list)):
            value_list = info_list[k].split(",")
            # [NTLValue, Zenith, YYYYMMDD]
            pointsDic[pointNum].append([float(value_list[2]), float(value_list[1]), value_list[0]])

    new_pointsDic: Dict[str, List[List[Any]]] = {}
    new_pointNumLngLatMap: Dict[str, str] = {}
    for key in pointsDic:
        if len(pointsDic[key]) < min_obs_per_point:
            continue
        new_pointsDic[key] = pointsDic[key]
        new_pointNumLngLatMap[key] = pointNumLngLatMap[key]
    return new_pointNumLngLatMap, new_pointsDic


def d_to_jd(time_str: str, fmt: str = "%Y.%m.%d") -> int:
    dt = datetime.strptime(time_str, fmt)
    tt = dt.timetuple()
    return tt.tm_year * 1000 + tt.tm_yday


def jd_to_time(jd_str: str, in_fmt: str = "%Y%j", out_fmt: str = "%Y.%m.%d") -> str:
    dt = datetime.strptime(jd_str, in_fmt).date()
    return dt.strftime(out_fmt)


def getRandomIndex(total_size: int, select_size: int) -> List[int]:
    randomIndex_list: List[int] = []
    while True:
        if len(randomIndex_list) == select_size:
            break
        randomNum = int(random.random() * total_size)
        if randomNum not in randomIndex_list:
            randomIndex_list.append(randomNum)
    return randomIndex_list


def getRandomIndex_crossValidationK(total_size: int, select_size: int, crossValidation_count: int) -> List[List[int]]:
    randomIndex_list: List[List[int]] = []
    randomIndex_list_temp: List[int] = []
    alreadyUseIndex: List[int] = []
    for _ in range(crossValidation_count):
        while True:
            if len(randomIndex_list_temp) == select_size:
                break
            randomNum = int(random.random() * total_size)
            if randomNum not in alreadyUseIndex:
                randomIndex_list_temp.append(randomNum)
                alreadyUseIndex.append(randomNum)
        randomIndex_list.append(randomIndex_list_temp)
        randomIndex_list_temp = []
    return randomIndex_list


def find_params(
    df: pd.DataFrame,
    fdays: int,
    testIndex_list: List[int],
    sp_scale: float,
    cp_scale: float,
    nPoints: int,
    base_params: Dict[str, Any],
) -> float:
    params = {
        "seasonality_mode": base_params.get("seasonality_mode", "additive"),
        "seasonality_prior_scale": sp_scale,
        "changepoint_prior_scale": cp_scale,
        "n_changepoints": int(max(1, round(nPoints))),
        "daily_seasonality": base_params.get("daily_seasonality", False),
        "weekly_seasonality": base_params.get("weekly_seasonality", False),
        "changepoint_range": base_params.get("changepoint_range", 1),
    }
    avg_err = predict_one(df, fdays, testIndex_list, params)
    return avg_err


def predict_one(train_df: pd.DataFrame, fdays: int, testIndex_list: List[int], params: Dict[str, Any]) -> float:
    m = Prophet(**params)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=fdays)
    forecast = m.predict(future)
    return mean_forecast_err(train_df, testIndex_list, forecast[["ds", "yhat"]])


def mean_forecast_err(train_df: pd.DataFrame, testIndex_list: List[int], df_dsAndyhat: pd.DataFrame) -> float:
    mean_value = 0.0
    for tsIndex in testIndex_list:
        yhat = df_dsAndyhat.loc[tsIndex, 'yhat']
        y = train_df.loc[tsIndex, 'y']
        mean_value += math.fabs(y - yhat)
    mean_value = mean_value / max(1, len(testIndex_list))
    return float(mean_value)


def mean_forecast_err1(test_NTLValue_series: List[float], testIndex_list: List[int], df_dsAndyhat: pd.DataFrame) -> float:
    mean_value = 0.0
    count_index = 0
    for tsIndex in testIndex_list:
        yhat = df_dsAndyhat.loc[tsIndex, 'yhat']
        y = test_NTLValue_series[count_index]
        count_index += 1
        mean_value += math.fabs(y - yhat)
    mean_value = mean_value / max(1, len(testIndex_list))
    return float(mean_value)


def writeDatafillingResult(
    f,
    key_words: str,
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    rows: int,
    point_lng_lat: str,
    include_coords: bool = False,
) -> List[float]:
    if include_coords:
        f.write(key_words + ":" + str(point_lng_lat) + ":")
    else:
        f.write(key_words + ":")
    temp_arr: List[float] = []
    for k in range(rows):
        date_str = str(forecast.loc[k, 'ds']).split(" ")[0]
        if k == 0:
            if (df.loc[k, 'y'] is None):
                f.write(date_str + "," + str(forecast.loc[k, 'yhat']))
                temp_arr.append(forecast.loc[k, 'yhat'])
            else:
                f.write(date_str + "," + str(df.loc[k, 'y']))
                temp_arr.append(df.loc[k, 'y'])
        else:
            if (df.loc[k, 'y'] is None):
                f.write(";" + date_str + "," + str(forecast.loc[k, 'yhat']))
                temp_arr.append(forecast.loc[k, 'yhat'])
            else:
                f.write(";" + date_str + "," + str(df.loc[k, 'y']))
                temp_arr.append(df.loc[k, 'y'])
    f.write("\n")
    return temp_arr


def formatDatafillingResult(
    key_words: str,
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    rows: int,
    point_lng_lat: str,
    include_coords: bool = False,
) -> Tuple[str, List[float]]:
    """Return the same output format as writeDatafillingResult, but as a string for parallel collection and ordered writing."""
    parts: List[str] = []
    if include_coords:
        header = f"{key_words}:{point_lng_lat}:"
    else:
        header = f"{key_words}:"
    parts.append(header)
    temp_arr: List[float] = []
    for k in range(rows):
        date_str = str(forecast.loc[k, 'ds']).split(" ")[0]
        if k == 0:
            if (df.loc[k, 'y'] is None):
                parts.append(f"{date_str},{forecast.loc[k, 'yhat']}")
                temp_arr.append(forecast.loc[k, 'yhat'])
            else:
                parts.append(f"{date_str},{df.loc[k, 'y']}")
                temp_arr.append(df.loc[k, 'y'])
        else:
            if (df.loc[k, 'y'] is None):
                parts.append(f";{date_str},{forecast.loc[k, 'yhat']}")
                temp_arr.append(forecast.loc[k, 'yhat'])
            else:
                parts.append(f";{date_str},{df.loc[k, 'y']}")
                temp_arr.append(df.loc[k, 'y'])
    line = "".join(parts) + "\n"
    return line, temp_arr


def _process_point_worker(args: Tuple[Any, ...]) -> Tuple[int, str, str, float, float]:
    """Parallel worker for a single point.

    Returns: (key_words, line, mape, relative_error)
    """
    (
        idx,
        key_words,
        time_series,
        ntl_values,
        point_lng_lat,
        time_1,
        time_2,
        pso_cfg,
        prophet_base,
        future_days,
        flags,
        base_seed,
    ) = args

    # 0) Deterministic random seed for reproducible but point-specific runs
    try:
        s = None if base_seed is None else int(base_seed) + int(idx)
    except Exception:
        s = None
    if s is not None:
        try:
            random.seed(s)
            np.random.seed(s)
        except Exception:
            pass

    # 1) Rebuild a continuous date sequence
    new_time_series: List[str] = []
    new_NTLValue_series: List[Optional[float]] = []
    i_count = 0

    current_time = d_to_jd(time_series[0][:4] + "." + time_series[0][4:6] + "." + time_series[0][6:])
    if (str(current_time)[-3:] != time_1):
        before_time = int(str(current_time)[:-3] + time_1)
        diff_count = current_time - before_time
        for j in range(diff_count):
            YYYYMMDD_list = jd_to_time(str(before_time + j)).split('.')
            new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
            new_NTLValue_series.append(None)

    for i in range(len(ntl_values) - 1):
        i_count = i
        last_time = d_to_jd(time_series[i + 1][:4] + "." + time_series[i + 1][4:6] + "." + time_series[i + 1][6:])
        pre_time = d_to_jd(time_series[i][:4] + "." + time_series[i][4:6] + "." + time_series[i][6:])
        diff_count = int(last_time) - int(pre_time)

        if (diff_count > 1 and diff_count < 365):
            new_time_series.append(time_series[i])
            new_NTLValue_series.append(ntl_values[i])
            for j in range(1, diff_count):
                YYYYMMDD_list = jd_to_time(str(int(pre_time) + j)).split('.')
                new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                new_NTLValue_series.append(None)
        elif (diff_count > 365):
            new_time_series.append(time_series[i])
            new_NTLValue_series.append(ntl_values[i])

            year_temp = int(str(pre_time)[0:4])
            day_temp = int(str(pre_time)[4:])
            if (year_temp % 4 == 0):
                if (day_temp < 366):
                    for ii in range(366 - day_temp):
                        YYYYMMDD_list = jd_to_time(str(int(pre_time) + ii + 1)).split('.')
                        new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                        new_NTLValue_series.append(None)
            else:
                if (day_temp < 365):
                    for ii in range(365 - day_temp):
                        YYYYMMDD_list = jd_to_time(str(int(pre_time) + ii + 1)).split('.')
                        new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                        new_NTLValue_series.append(None)

            if (str(last_time)[-3:] != "001"):
                before_time = int(str(last_time)[:-3] + "001")
                diff_count2 = last_time - before_time
                for j in range(diff_count2):
                    YYYYMMDD_list = jd_to_time(str(before_time + j)).split('.')
                    new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                    new_NTLValue_series.append(None)
        else:
            new_time_series.append(time_series[i])
            new_NTLValue_series.append(ntl_values[i])

    new_time_series.append(time_series[i_count + 1])
    new_NTLValue_series.append(ntl_values[i_count + 1])
    end_time = d_to_jd(time_series[i_count + 1][:4] + "." + time_series[i_count + 1][4:6] + "." + time_series[i_count + 1][6:])

    if (str(end_time)[-3:] != time_2):
        after_time = int(str(end_time)[:-3] + time_2)
        before_time = int(end_time)
        diff_count = after_time - before_time
        for j in range(diff_count):
            YYYYMMDD_list = jd_to_time(str(before_time + j + 1)).split('.')
            new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
            new_NTLValue_series.append(None)

    # 2) Build the DataFrame
    column_name = ['ds', 'y']
    df = pd.DataFrame(np.vstack((new_time_series, new_NTLValue_series)).T, columns=column_name)
    # Training indices
    testIndex_list_allDst_rest: List[int] = []
    for tsTime in time_series:
        testIndex_list_allDst_rest.append(df[df.ds == tsTime].index.tolist()[0])

    # 3) PSO parameter bounds
    max_value = int(len(ntl_values) * float(pso_cfg.get("max_n_changepoints_ratio", 0.6)))
    min_value = float(pso_cfg.get("min_value", 0.01))

    # 4) PSO
    pso = PSO(
        particle_num=int(pso_cfg.get("particle_num", 3)),
        particle_dim=int(pso_cfg.get("particle_dim", 3)),
        iter_num=int(pso_cfg.get("iter_num", 3)),
        c1=float(pso_cfg.get("c1", 2.0)),
        c2=float(pso_cfg.get("c2", 2.0)),
        w=float(pso_cfg.get("w", 0.5)),
        max_value=max_value,
        min_value=min_value,
        df=df,
        fdays=int(future_days),
        testIndex_list_allDst=testIndex_list_allDst_rest,
        base_params=prophet_base,
        enable_plot=False,
        quiet=True,
        progress=False,
    )
    glo_gbest_parameter = pso.main()

    # 5) Fit and forecast
    m = Prophet(
        seasonality_mode=prophet_base.get("seasonality_mode", "additive"),
        seasonality_prior_scale=glo_gbest_parameter[0],
        changepoint_prior_scale=glo_gbest_parameter[1],
        n_changepoints=int(max(1, round(glo_gbest_parameter[2]))),
        changepoint_range=prophet_base.get("changepoint_range", 1),
        daily_seasonality=prophet_base.get("daily_seasonality", False),
        weekly_seasonality=prophet_base.get("weekly_seasonality", False),
    )
    # Prophet fitting: pass a seed when supported and fall back otherwise
    try:
        m.fit(df, seed=s if s is not None else None)
    except TypeError:
        m.fit(df)
    future = m.make_future_dataframe(periods=int(future_days), freq='D')
    forecast = m.predict(future)

    # 6) Metrics and serialization
    mape = mean_forecast_err1(ntl_values, testIndex_list_allDst_rest, forecast[['ds', 'yhat']])
    try:
        relative_error_once = mape / (np.mean(ntl_values) or 1.0)
    except Exception:
        relative_error_once = float('nan')

    line, _ = formatDatafillingResult(
        key_words,
        df,
        forecast,
        len(new_time_series),
        point_lng_lat,
        include_coords=bool(flags.get("output_include_coords", False)),
    )
    return int(idx), key_words, line, float(mape), float(relative_error_once)


def run_prophet_pipeline(
    input_dirs: List[str],
    output_dirs: List[str],
    loop_max_freq: int = 1,
    doy_start: str = "001",
    doy_end: str = "237",
    min_obs_per_point: int = 10,
    pso_config: Optional[Dict[str, Any]] = None,
    prophet_config: Optional[Dict[str, Any]] = None,
    future_days: int = 0,
    flags: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full pipeline from reading TXT data and PSO search to Prophet fitting and filled-output export.

    Returns a summary dictionary with basic statistics.
    """

    t_start = time.time()

    if random_seed is not None:
        try:
            random.seed(random_seed)
            np.random.seed(int(random_seed))
        except Exception:
            pass

    # Default PSO configuration
    pso_cfg = {
        "particle_num": 3,
        "particle_dim": 3,
        "iter_num": 3,
        "c1": 2.0,
        "c2": 2.0,
        "w": 0.5,
        "min_value": 0.01,
        "max_n_changepoints_ratio": 0.6,
    }
    if pso_config:
        pso_cfg.update(pso_config)

    # Default fixed Prophet parameters
    prophet_base = {
        "seasonality_mode": "additive",
        "daily_seasonality": False,
        "weekly_seasonality": False,
        "changepoint_range": 1,
    }
    if prophet_config:
        prophet_base.update(prophet_config)

    # Default feature flags
    flg = {
        "enable_pso_plot": False,
        "enable_forecast_plot": False,
        "enable_simple_series_plot": False,
        "output_include_coords": False,
        "progress_bar": True,
        "quiet": True,
        "parallel_points": False,
        "num_workers": None,  # None means use the system default
        "write_metrics_report": True,  # Whether to write a per-point metrics CSV report
    }
    if flags:
        flg.update(flags)

    # Silence Prophet INFO logs
    try:
        logging.getLogger(_PROPHET_LOGGER_NAME).setLevel(logging.WARNING)
    except Exception:
        pass

    processed_points = 0
    outputs: List[str] = []
    all_metrics: List[Dict[str, Any]] = []  # Summary across all processed files

    freq = 0
    while freq < loop_max_freq:
        fit_result_path = input_dirs[freq]
        outputFile = output_dirs[freq]
        pointNumLngLatMap_fit, pointsDic_fit = readFile(fit_result_path, min_obs_per_point)

        time_1 = doy_start
        time_2 = doy_end

        time_series_dic: Dict[str, List[str]] = {}
        NTLValue_series_dic: Dict[str, List[float]] = {}
        for key in pointsDic_fit:
            time_series_dic[key] = []
            NTLValue_series_dic[key] = []
            for i in range(len(pointsDic_fit[key])):
                time_series_dic[key].append(pointsDic_fit[key][i][2])
                NTLValue_series_dic[key].append(pointsDic_fit[key][i][0])

        outputs.append(outputFile)

        # Collect per-point metrics for this file
        per_file_metrics: List[Tuple[str, float, float]] = []  # (point_id, mae, rel_err)

        key_list = list(time_series_dic.keys())

        # Parallel or serial execution
        if flg.get("parallel_points", False):
            max_workers = flg.get("num_workers") or max(1, (os.cpu_count() or 2) - 1)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for idx, key_words in enumerate(key_list):
                    futures.append(
                        ex.submit(
                            _process_point_worker,
                            (
                                idx,
                                key_words,
                                time_series_dic[key_words],
                                NTLValue_series_dic[key_words],
                                pointNumLngLatMap_fit[key_words],
                                time_1,
                                time_2,
                                pso_cfg,
                                prophet_base,
                                future_days,
                                flg,
                                int(random_seed) if random_seed is not None else None,
                            ),
                        )
                    )
                pbar = tqdm(total=len(futures), desc=f"Points {freq+1}/{loop_max_freq}") if flg.get("progress_bar", True) else None

                # Incrementally write results while preserving original order via a buffer and next_idx pointer
                next_idx = 0
                buffer: Dict[int, str] = {}
                with open(outputFile, 'a') as f:
                    for fut in as_completed(futures):
                        idx, key_words, line, mape, rel = fut.result()
                        buffer[idx] = line
                        # Record metrics (historically named mape but numerically MAE)
                        try:
                            per_file_metrics.append((str(key_words), float(mape), float(rel)))
                        except Exception:
                            pass
                        # Try writing consecutive buffered records in order
                        while next_idx in buffer:
                            f.write(buffer.pop(next_idx))
                            processed_points += 1
                            next_idx += 1
                        if pbar is not None:
                            pbar.update(1)
                if pbar is not None:
                    pbar.close()
        else:
            # Original serial path with progress bar and quiet-mode control preserved
            key_iter = key_list
            if flg.get("progress_bar", True):
                key_iter = tqdm(key_list, desc=f"Points {freq+1}/{loop_max_freq}")
            with open(outputFile, 'a') as f:
                for idx, key_words in enumerate(key_iter):
                    # Set a deterministic random seed for this point
                    s = None
                    try:
                        s = int(random_seed) + int(idx) if random_seed is not None else None
                    except Exception:
                        s = None
                    if s is not None:
                        try:
                            random.seed(s)
                            np.random.seed(s)
                        except Exception:
                            pass
                    # Extract the training data for fitting
                    test_time_series_rest: List[str] = []
                    test_NTLValue_series_rest: List[float] = []
                    for index in range(len(time_series_dic[key_words])):
                        test_time_series_rest.append(time_series_dic[key_words][index])
                        test_NTLValue_series_rest.append(NTLValue_series_dic[key_words][index])

                    # Build a continuous sequence and fill missing values with None (same logic as the worker)
                    new_time_series: List[str] = []
                    new_NTLValue_series: List[Optional[float]] = []
                    i_count = 0

                    current_time = d_to_jd(time_series_dic[key_words][0][:4] + "." + time_series_dic[key_words][0][4:6] + "." + time_series_dic[key_words][0][6:])
                    if (str(current_time)[-3:] != time_1):
                        before_time = int(str(current_time)[:-3] + time_1)
                        diff_count = current_time - before_time
                        for j in range(diff_count):
                            YYYYMMDD_list = jd_to_time(str(before_time + j)).split('.')
                            new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                            new_NTLValue_series.append(None)

                    for i in range(len(NTLValue_series_dic[key_words]) - 1):
                        i_count = i
                        last_time = d_to_jd(time_series_dic[key_words][i + 1][:4] + "." + time_series_dic[key_words][i + 1][4:6] + "." + time_series_dic[key_words][i + 1][6:])
                        pre_time = d_to_jd(time_series_dic[key_words][i][:4] + "." + time_series_dic[key_words][i][4:6] + "." + time_series_dic[key_words][i][6:])
                        diff_count = int(last_time) - int(pre_time)

                        if (diff_count > 1 and diff_count < 365):
                            new_time_series.append(time_series_dic[key_words][i])
                            new_NTLValue_series.append(NTLValue_series_dic[key_words][i])
                            for j in range(1, diff_count):
                                YYYYMMDD_list = jd_to_time(str(int(pre_time) + j)).split('.')
                                new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                                new_NTLValue_series.append(None)
                        elif (diff_count > 365):
                            new_time_series.append(time_series_dic[key_words][i])
                            new_NTLValue_series.append(NTLValue_series_dic[key_words][i])

                            year_temp = int(str(pre_time)[0:4])
                            day_temp = int(str(pre_time)[4:])
                            if (year_temp % 4 == 0):
                                if (day_temp < 366):
                                    for ii in range(366 - day_temp):
                                        YYYYMMDD_list = jd_to_time(str(int(pre_time) + ii + 1)).split('.')
                                        new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                                        new_NTLValue_series.append(None)
                            else:
                                if (day_temp < 365):
                                    for ii in range(365 - day_temp):
                                        YYYYMMDD_list = jd_to_time(str(int(pre_time) + ii + 1)).split('.')
                                        new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                                        new_NTLValue_series.append(None)

                            if (str(last_time)[-3:] != "001"):
                                before_time = int(str(last_time)[:-3] + "001")
                                diff_count2 = last_time - before_time
                                for j in range(diff_count2):
                                    YYYYMMDD_list = jd_to_time(str(before_time + j)).split('.')
                                    new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                                    new_NTLValue_series.append(None)
                        else:
                            new_time_series.append(time_series_dic[key_words][i])
                            new_NTLValue_series.append(NTLValue_series_dic[key_words][i])

                    new_time_series.append(time_series_dic[key_words][i_count + 1])
                    new_NTLValue_series.append(NTLValue_series_dic[key_words][i_count + 1])
                    end_time = d_to_jd(time_series_dic[key_words][i_count + 1][:4] + "." + time_series_dic[key_words][i_count + 1][4:6] + "." + time_series_dic[key_words][i_count + 1][6:])

                    if (str(end_time)[-3:] != time_2):
                        after_time = int(str(end_time)[:-3] + time_2)
                        before_time = int(end_time)
                        diff_count = after_time - before_time
                        for j in range(diff_count):
                            YYYYMMDD_list = jd_to_time(str(before_time + j + 1)).split('.')
                            new_time_series.append(str(YYYYMMDD_list[0]) + str(YYYYMMDD_list[1]) + str(YYYYMMDD_list[2]))
                            new_NTLValue_series.append(None)

                    # Build the DataFrame
                    column_name = ['ds', 'y']
                    df = pd.DataFrame(np.vstack((new_time_series, new_NTLValue_series)).T, columns=column_name)
                    # Training indices
                    testIndex_list_allDst_rest: List[int] = []
                    for tsTime in test_time_series_rest:
                        testIndex_list_allDst_rest.append(df[df.ds == tsTime].index.tolist()[0])

                    # PSO parameter bounds
                    max_value = int(len(NTLValue_series_dic[key_words]) * float(pso_cfg.get("max_n_changepoints_ratio", 0.6)))
                    min_value = float(pso_cfg.get("min_value", 0.01))

                    pso = PSO(
                        particle_num=int(pso_cfg.get("particle_num", 3)),
                        particle_dim=int(pso_cfg.get("particle_dim", 3)),
                        iter_num=int(pso_cfg.get("iter_num", 3)),
                        c1=float(pso_cfg.get("c1", 2.0)),
                        c2=float(pso_cfg.get("c2", 2.0)),
                        w=float(pso_cfg.get("w", 0.5)),
                        max_value=max_value,
                        min_value=min_value,
                        df=df,
                        fdays=int(future_days),
                        testIndex_list_allDst=testIndex_list_allDst_rest,
                        base_params=prophet_base,
                        enable_plot=bool(flg.get("enable_pso_plot", False)),
                        quiet=bool(flg.get("quiet", True)),
                        progress=bool(flg.get("progress_bar", True)),
                    )
                    glo_gbest_parameter = pso.main()

                    # Fit using the best parameters
                    m = Prophet(
                        seasonality_mode=prophet_base.get("seasonality_mode", "additive"),
                        seasonality_prior_scale=glo_gbest_parameter[0],
                        changepoint_prior_scale=glo_gbest_parameter[1],
                        n_changepoints=int(max(1, round(glo_gbest_parameter[2]))),
                        changepoint_range=prophet_base.get("changepoint_range", 1),
                        daily_seasonality=prophet_base.get("daily_seasonality", False),
                        weekly_seasonality=prophet_base.get("weekly_seasonality", False),
                    )
                    # Fit Prophet with a seed to keep runs reproducible
                    try:
                        m.fit(df, seed=s if s is not None else None)
                    except TypeError:
                        m.fit(df)
                    future = m.make_future_dataframe(periods=int(future_days), freq='D')
                    forecast = m.predict(future)

                    if flg.get("enable_forecast_plot", False) and plt is not None:
                        m.plot(forecast)
                        plt.xlabel("date")
                        plt.ylabel("NTL")
                        plt.title(str(key_words))
                        plt.show()

                    mape = mean_forecast_err1(test_NTLValue_series_rest, testIndex_list_allDst_rest, forecast[['ds', 'yhat']])
                    try:
                        relative_error_once = mape / (np.mean(NTLValue_series_dic[key_words]) or 1.0)
                    except Exception:
                        relative_error_once = float('nan')

                    if not flg.get("quiet", True):
                        print(key_words, "mape:", mape, "relative error:", relative_error_once)

                    point_lng_lat = pointNumLngLatMap_fit[key_words]
                    temp_series = writeDatafillingResult(
                        f,
                        key_words,
                        df,
                        forecast,
                        len(new_time_series),
                        point_lng_lat,
                        include_coords=bool(flg.get("output_include_coords", False)),
                    )

                    # Record per-point metrics
                    try:
                        per_file_metrics.append((str(key_words), float(mape), float(relative_error_once)))
                    except Exception:
                        pass

                    if flg.get("enable_simple_series_plot", False) and plt is not None:
                        plt.plot(new_time_series, temp_series, color="cornflowerblue", alpha=0.7)
                        plt.xticks(rotation=300)
                        plt.xlabel("date")
                        plt.ylabel("NTL")
                        plt.title("Filled series: " + str(key_words))
                        plt.show()

                    processed_points += 1

        # After processing this file, write the metrics report (CSV) and compute summary statistics
        metrics_summary: Dict[str, Any] = {}
        if per_file_metrics:
            mae_list = [m[1] for m in per_file_metrics if m[1] is not None]
            rel_list = [m[2] for m in per_file_metrics if m[2] is not None]
            if mae_list:
                try:
                    mae_avg = float(np.mean(mae_list))
                    mae_med = float(np.median(mae_list))
                except Exception:
                    mae_avg = float('nan'); mae_med = float('nan')
            else:
                mae_avg = float('nan'); mae_med = float('nan')
            if rel_list:
                try:
                    rel_avg = float(np.mean(rel_list))
                    rel_med = float(np.median(rel_list))
                except Exception:
                    rel_avg = float('nan'); rel_med = float('nan')
            else:
                rel_avg = float('nan'); rel_med = float('nan')

            metrics_summary = {
                "file": outputFile,
                "points": len(per_file_metrics),
                "mae_avg": mae_avg,
                "mae_med": mae_med,
                "rel_avg": rel_avg,
                "rel_med": rel_med,
            }

            # Write the CSV report
            if flg.get("write_metrics_report", True):
                report_path = os.path.splitext(outputFile)[0] + "_metrics.csv"
                try:
                    with open(report_path, 'w', newline='') as rf:
                        writer = csv.writer(rf)
                        writer.writerow(["point_id", "mae", "relative_error"])
                        writer.writerows(per_file_metrics)
                    metrics_summary["report_csv"] = report_path
                except Exception:
                    metrics_summary["report_csv"] = None

        all_metrics.append(metrics_summary)

        # Increment loop counter
        freq += 1

    # Finish execution and return summary information
    t_end = time.time()
    # Aggregate accuracy summaries for all files
    summary_metrics = all_metrics

    return {
        "processed_points": processed_points,
        "output_files": outputs,
        "elapsed_time": t_end - t_start,
        "metrics": summary_metrics,
    }

