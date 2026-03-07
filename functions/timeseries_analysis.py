"""Time-series analysis and visualization utilities."""

import datetime as dtime
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _read_point_timeseries(file_path: str) -> Dict[str, List[Tuple[str, float, float]]]:
	"""
	Read preprocessing or normalization output text.

	File format:
		pointNum:lng,lat(left top):YYYYMMDD,Zenith,NTLValue;...

	Returns:
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
	"""Aggregate all pixel NTL values by date and compute the daily mean."""
	date_values: Dict[str, List[float]] = {}

	for records in points.values():
		for date_str, _, ntl in records:
			date_values.setdefault(date_str, []).append(ntl)

	return {d: float(np.mean(v)) for d, v in date_values.items() if v}


def plot_mean_ntl_before_after(
	original_file: str,
	normalized_file: str,
	title: str = "Mean Nighttime Light Time Series (Before vs After Normalization)",
	figsize: Tuple[int, int] = (12, 5),
) -> dict:
	"""
	Plot the mean nighttime light time series before and after normalization.

	Parameters:
		original_file: Path to the pre-normalization time-series file
		normalized_file: Path to the post-normalization time-series file
		title: Plot title
		figsize: Figure size

	Returns:
		dict: Aligned dates, mean series before/after normalization, and summary statistics
	"""
	ori_points = _read_point_timeseries(original_file)
	fit_points = _read_point_timeseries(normalized_file)

	if not ori_points:
		raise ValueError(f"No valid data found in original file: {original_file}")
	if not fit_points:
		raise ValueError(f"No valid data found in normalized file: {normalized_file}")

	ori_daily = _daily_mean_ntl(ori_points)
	fit_daily = _daily_mean_ntl(fit_points)

	common_dates = sorted(set(ori_daily.keys()) & set(fit_daily.keys()))
	if not common_dates:
		raise ValueError("The original and normalized datasets do not share any overlapping dates")

	x_dt = [dtime.datetime.strptime(d, "%Y%m%d") for d in common_dates]
	y_ori = np.array([ori_daily[d] for d in common_dates], dtype="float64")
	y_fit = np.array([fit_daily[d] for d in common_dates], dtype="float64")

	plt.figure(figsize=figsize)
	plt.plot(x_dt, y_ori, color="#e4572e", linewidth=2.0, label="Mean before normalization")
	plt.plot(x_dt, y_fit, color="#2e86ab", linewidth=2.0, label="Mean after normalization")
	plt.title(title)
	plt.xlabel("Date")
	plt.ylabel("Mean nighttime light")
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
