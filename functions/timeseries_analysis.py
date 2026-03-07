"""时间序列分析与可视化工具。"""

import datetime as dtime
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _read_point_timeseries(file_path: str) -> Dict[str, List[Tuple[str, float, float]]]:
	"""
	读取预处理/归一化输出文本。

	文件格式:
		pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...

	返回:
		{point_id: [(date_str, zenith, ntl), ...]}
	"""
	points: Dict[str, List[Tuple[str, float, float]]] = {}

	with open(file_path, "r", encoding="utf-8") as f:
		for idx, line in enumerate(f):
			if idx == 0:
				continue

			parts = line.strip().split(":")
			if len(parts) < 3:
				continue

			point_id = parts[0]
			payload = parts[2]
			records: List[Tuple[str, float, float]] = []

			for item in payload.split(";"):
				vals = item.split(",")
				if len(vals) < 3:
					continue
				try:
					date_str = vals[0]
					zenith = float(vals[1])
					ntl = float(vals[2])
					records.append((date_str, zenith, ntl))
				except (TypeError, ValueError):
					continue

			if records:
				points[point_id] = records

	return points


def _daily_mean_ntl(points: Dict[str, List[Tuple[str, float, float]]]) -> Dict[str, float]:
	"""按日期聚合所有像元 NTL，计算每日均值。"""
	date_values: Dict[str, List[float]] = {}

	for records in points.values():
		for date_str, _, ntl in records:
			date_values.setdefault(date_str, []).append(ntl)

	return {d: float(np.mean(v)) for d, v in date_values.items() if v}


def plot_mean_ntl_before_after(
	original_file: str,
	normalized_file: str,
	title: str = "夜光均值时序（归一化前后）",
	figsize: Tuple[int, int] = (12, 5),
) -> dict:
	"""
	绘制归一化前后夜光均值时序图。

	参数:
		original_file: 归一化前时序文件路径
		normalized_file: 归一化后时序文件路径
		title: 图标题
		figsize: 图尺寸

	返回:
		dict: 包含对齐日期、前后均值序列与变化统计
	"""
	ori_points = _read_point_timeseries(original_file)
	fit_points = _read_point_timeseries(normalized_file)

	if not ori_points:
		raise ValueError(f"原始文件无可用数据: {original_file}")
	if not fit_points:
		raise ValueError(f"归一化文件无可用数据: {normalized_file}")

	ori_daily = _daily_mean_ntl(ori_points)
	fit_daily = _daily_mean_ntl(fit_points)

	common_dates = sorted(set(ori_daily.keys()) & set(fit_daily.keys()))
	if not common_dates:
		raise ValueError("原始与归一化数据无重叠日期，无法对比")

	x_dt = [dtime.datetime.strptime(d, "%Y%m%d") for d in common_dates]
	y_ori = np.array([ori_daily[d] for d in common_dates], dtype="float64")
	y_fit = np.array([fit_daily[d] for d in common_dates], dtype="float64")

	plt.figure(figsize=figsize)
	plt.plot(x_dt, y_ori, color="#e4572e", linewidth=2.0, label="归一化前均值")
	plt.plot(x_dt, y_fit, color="#2e86ab", linewidth=2.0, label="归一化后均值")
	plt.title(title)
	plt.xlabel("日期")
	plt.ylabel("夜光均值")
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
	plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
	plt.grid(True, alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.show()

	delta = y_fit - y_ori
	stats = {
		"n_dates": int(len(common_dates)),
		"mean_before": float(np.mean(y_ori)),
		"mean_after": float(np.mean(y_fit)),
		"mean_delta": float(np.mean(delta)),
		"median_delta": float(np.median(delta)),
	}

	return {
		"dates": common_dates,
		"mean_before_series": y_ori.tolist(),
		"mean_after_series": y_fit.tolist(),
		"stats": stats,
	}
