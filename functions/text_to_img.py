from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import os
import re

import numpy as np
import rasterio
from affine import Affine


DateLike = Union[str, datetime]


def _as_datetime(value: DateLike, accepted_formats: Sequence[str]) -> datetime:
    if isinstance(value, datetime):
        return value
    for fmt in accepted_formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {value}. Supported formats: {accepted_formats}")


def _parse_text_entries(txt_path: str) -> List[Tuple[int, str]]:
    entries: List[Tuple[int, str]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            if line.lower().startswith("pointnum:"):
                continue

            head = line.split(":", 1)[0].strip()
            match = re.search(r"(\d+)$", head)
            if not match:
                continue

            entries.append((int(match.group(1)), line))

    entries.sort(key=lambda x: x[0])
    return entries


def _extract_payload(content: str) -> str:
    parts = content.split(":")
    if len(parts) >= 3:
        return parts[2]
    if len(parts) == 2:
        return parts[1]
    return ""


def _extract_value(fields: Sequence[str]) -> float:
    if len(fields) < 2:
        raise ValueError("Each record must contain at least date and value")
    return float(fields[-1])


def txt_to_daily_geotiffs(
    txt_path: str,
    output_dir: str,
    start_date: DateLike,
    end_date: DateLike,
    template_tif: Optional[str] = None,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    transform: Optional[Union[Affine, Sequence[float]]] = None,
    crs: Optional[Union[str, dict]] = None,
    nodata_value: float = -9999.0,
    date_formats: Optional[Iterable[str]] = None,
) -> List[str]:
    """Convert point-based time-series text into daily GeoTIFF files.

    Parameters:
    - template_tif: Path to a template raster. When provided, spatial metadata,
      resolution, width, and height are inherited automatically.
    - template-free mode: Requires width, height, transform, and crs.
        - supported txt format:
            pointN:lng,lat:YYYYMMDD,Zenith,NTL;...
            pointN:lng,lat:YYYY-MM-DD,0.0,NTL;...
            pointN:YYYY-MM-DD,NTL;... (legacy output without coordinates)
    """
    accepted_formats = tuple(date_formats) if date_formats else ("%Y%m%d", "%Y-%m-%d")
    start_dt = _as_datetime(start_date, accepted_formats)
    end_dt = _as_datetime(end_date, accepted_formats)
    if end_dt < start_dt:
        raise ValueError("end_date cannot be earlier than start_date")

    meta = {}
    if template_tif:
        with rasterio.open(template_tif) as src:
            meta = src.meta.copy()
            height = src.height
            width = src.width
    else:
        if width is None or height is None or transform is None or crs is None:
            raise ValueError("In template-free mode, width, height, transform, and crs are required")
        if isinstance(transform, (list, tuple)):
            if len(transform) != 6:
                raise ValueError("transform must contain 6 values in Affine(a, b, c, d, e, f) order")
            transform = Affine(*transform)
        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "crs": crs,
            "transform": transform,
        }

    interval_day = (end_dt - start_dt).days + 1
    cube = np.full((height, width, interval_day), nodata_value, dtype=np.float32)

    entries = _parse_text_entries(txt_path)
    entry_by_id = {idx: content for idx, content in entries}

    total_pixels = height * width
    pixel_id = 1
    for r in range(height):
        for c in range(width):
            content = entry_by_id.get(pixel_id)
            if content:
                payload = _extract_payload(content)
                rows = payload.split(";") if payload else []
                for row in rows:
                    row = row.strip()
                    if not row:
                        continue
                    fields = [field.strip() for field in row.split(",")]
                    if len(fields) < 2:
                        continue
                    date_dt = _as_datetime(fields[0], accepted_formats)
                    if start_dt <= date_dt <= end_dt:
                        cube[r, c, (date_dt - start_dt).days] = _extract_value(fields)
            pixel_id += 1
            if pixel_id > total_pixels and r == height - 1 and c == width - 1:
                break

    os.makedirs(output_dir, exist_ok=True)
    meta.update({"dtype": "float32", "count": 1, "nodata": nodata_value})

    outputs: List[str] = []
    for i in range(interval_day):
        out_file = os.path.join(output_dir, f"{(start_dt + timedelta(days=i)).strftime('%Y%m%d')}.tif")
        with rasterio.open(out_file, "w", **meta) as dst:
            dst.write(cube[:, :, i], 1)
        outputs.append(out_file)

    return outputs
