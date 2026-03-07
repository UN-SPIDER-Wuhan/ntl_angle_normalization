from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import os

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
    raise ValueError(f"无法解析日期: {value}. 支持格式: {accepted_formats}")


def _parse_text_entries(txt_path: str) -> List[Tuple[int, str]]:
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    raw = raw.replace("pointNum:lng,lat(lefttop):YYYY-MM-DD,Zenith,NTLValue;...", "")

    chunks = raw.split("point")
    entries: List[Tuple[int, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        idx_text = chunk.split(":", 1)[0].strip()
        if not idx_text.isdigit():
            continue
        entries.append((int(idx_text), chunk))

    entries.sort(key=lambda x: x[0])
    return entries


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
    """将点位时序 txt 转为逐日 GeoTIFF。

    参数说明:
    - template_tif: 模板影像路径。提供后自动继承空间参考、分辨率和宽高。
    - 无模板模式: 需手动传 width、height、transform、crs。
    - txt 格式默认兼容: pointN:lng,lat:YYYYMMDD,Zenith,NTL;...
    """
    accepted_formats = tuple(date_formats) if date_formats else ("%Y%m%d", "%Y-%m-%d")
    start_dt = _as_datetime(start_date, accepted_formats)
    end_dt = _as_datetime(end_date, accepted_formats)
    if end_dt < start_dt:
        raise ValueError("end_date 不能早于 start_date")

    meta = {}
    if template_tif:
        with rasterio.open(template_tif) as src:
            meta = src.meta.copy()
            height = src.height
            width = src.width
    else:
        if width is None or height is None or transform is None or crs is None:
            raise ValueError("无模板模式下，必须提供 width、height、transform、crs")
        if isinstance(transform, (list, tuple)):
            if len(transform) != 6:
                raise ValueError("transform 序列长度必须为 6，格式为 Affine(a,b,c,d,e,f)")
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
                parts = content.split(":")
                if len(parts) >= 3:
                    rows = parts[2].split(";")
                    for row in rows:
                        row = row.strip()
                        if not row:
                            continue
                        fields = row.split(",")
                        if len(fields) < 3:
                            continue
                        date_dt = _as_datetime(fields[0].strip(), accepted_formats)
                        if start_dt <= date_dt <= end_dt:
                            cube[r, c, (date_dt - start_dt).days] = float(fields[2])
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
