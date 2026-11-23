"""Report signed height over an independently chosen rectangular region, including missing measurements."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path, roi, expected):
    with np.load(path, allow_pickle=False) as m:
        height = m['height_mm']
        supported = m['measured'] if 'measured' in m else m['confirmed']
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > height.shape[1] or y+h > height.shape[0]:
            raise ValueError("Прямоугольник вне карты")
        region = np.s_[y:y+h, x:x+w]
        values = height[region][supported[region] & np.isfinite(height[region])]
    if not np.isfinite(expected).all() or expected[0] > expected[1]:
        raise ValueError("Некорректный интервал эталона")
    result = dict(path=str(path), roi=list(roi), expected_mm=list(expected), count=len(values),
                  total=w*h, coverage=len(values)/(w*h))
    if len(values):
        error = values - np.clip(values, *expected)
        result.update(p10_median_p90_mm=np.percentile(values, [10, 50, 90]).tolist(),
                      median_signed_error_mm=float(np.median(error)),
                      p95_abs_error_mm=float(np.percentile(abs(error), 95)),
                      max_abs_error_mm=float(abs(error).max()), within_010=int(np.sum(abs(error)<=.1)))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('map', type=Path)
    ap.add_argument('--roi', type=int, nargs=4, required=True, metavar=('X','Y','W','H'))
    ap.add_argument('--expected', type=float, nargs=2, required=True, metavar=('MIN','MAX'))
    args = ap.parse_args()
    print(json.dumps(summarize(args.map, args.roi, args.expected), indent=2))
