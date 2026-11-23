"""Reproduce the offline reference-frame comparison for scan_20261123-171208.

Run from any folder with the project requirements installed. Only coverage.png,
next to this script, is written. XY areas are in squared command millimetres.
"""
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.stats import binned_statistic

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "z")]
from measure_height import ScanResult, add_height_args, load_scan
from stereo import Rectifier


def main():
    frames, meta = load_scan(ROOT / "scans/scan_20261123-171208")
    poses = np.array(meta["positions"])
    center = poses[meta["ref_index"]]
    V = np.array(meta["V_cam"])
    K, D = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    n, d = np.array(meta["guess"]["n"]), meta["guess"]["d"]
    rect = Rectifier(K, D, frames[0].shape[::-1], meta["model"])
    pitch = 0.5
    xs, ys = np.arange(-110, 111, pitch), np.arange(-77, 78, pitch)
    xx, yy = np.meshgrid(xs, ys)
    base = np.stack([xx, yy], axis=-1) @ V.T

    def surface(normal, distance, height):
        world = base.copy()
        world[..., 2] += (distance - height - base @ normal) / normal[2]
        return world

    def visible(world, offset):
        projected = (world - offset) @ K.T
        uv = (projected[..., :2] / projected[..., 2:]).astype(np.float32)
        return cv2.remap(rect.valid.astype(np.uint8), uv, None, cv2.INTER_NEAREST) > 0

    upper = surface(n, d, 30)
    masks = [visible(upper, V @ (pos - center)) for pos in poses]
    count = np.sum(masks, axis=0)
    center_mask = masks[meta["ref_index"]]
    to_xy = np.linalg.inv(np.column_stack([V, [0, 0, 1]]))[:2]
    args = add_height_args(argparse.ArgumentParser()).parse_args([])
    args.fill_planes = False
    maps = []
    for idx in [meta["ref_index"], 0, 4, 20, 24]:
        offset = V @ (poses[idx] - center)
        distance = d - n @ offset
        scan = ScanResult(frames, poses, idx, K, D, meta["model"], V,
                          (n, distance, meta["guess"]["kind"]), (n, distance), args, rect=rect)
        if not maps:
            table_n, table_d = scan.n_t.copy(), scan.d_t
        world = scan.X[scan.valid] + offset
        xy = world @ to_xy.T
        heights = table_d - world @ table_n
        ix = np.rint((xy[:, 0] - xs[0]) / pitch).astype(int)
        iy = np.rint((xy[:, 1] - ys[0]) / pitch).astype(int)
        keep = (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
        cells = iy[keep] * xs.size + ix[keep]
        per_cell = binned_statistic(cells, heights[keep], statistic="median",
                                    bins=np.arange(xx.size + 1) - 0.5)
        maps.append(per_cell.statistic.reshape(xx.shape))
        del scan

    stack = np.array(maps)
    votes = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        span = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        median = np.nanmedian(stack, axis=0)
    agreed = (votes >= 2) & (span <= 0.35)
    center_visible = visible(surface(table_n, table_d, np.nan_to_num(median)), np.zeros(3))
    outside = agreed & ~center_visible
    area = pitch ** 2
    print("At h=30, central / visible from >=5:", center_mask.sum()*area, (count >= 5).sum()*area)
    print("Measured central / union / consistent / consistent outside:",
          np.isfinite(stack[0]).sum()*area, (votes > 0).sum()*area, agreed.sum()*area, outside.sum()*area)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    extent = (xs[0]-pitch/2, xs[-1]+pitch/2, ys[0]-pitch/2, ys[-1]+pitch/2)
    geometric = np.select([center_mask & (count >= 5), count >= 5, count > 0], [3, 2, 1], 0)
    empirical = np.select([outside, agreed, votes > 0], [3, 2, 1], 0)
    panels = [
        (geometric, ["#f4f5f6", "#b7ccd5", "#468a9a", "#143f52"],
         "Видимость плоскости на высоте 30 мм",
         ["1–4 ракурса", "≥5 ракурсов", "≥5 и виден из центра"]),
        (empirical, ["#f4f5f6", "#b7ccd5", "#257c76", "#d08029"],
         "Прямые оценки по пяти опорным кадрам",
         ["Есть оценка", "≥2 оценки, размах ≤0,35 мм", "Согласие за границей центрального кадра"]),
    ]
    for ax, (data, palette, title, labels) in zip(axes, panels):
        ax.imshow(data, origin="lower", extent=extent, cmap=ListedColormap(palette),
                  vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set(title=title, xlabel="X относительно центра, командные мм", ylabel="Y, командные мм",
               xlim=(-105, 105), ylim=(-76, 76))
        ax.legend(handles=[Patch(color=c, label=s) for c, s in zip(palette[1:], labels)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9)
    axes[0].plot(*(poses-center).T, ".", ms=2, color="white")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.28, wspace=0.20)
    fig.savefig(HERE / "coverage.png", dpi=150, bbox_inches="tight", pad_inches=0.18)


if __name__ == "__main__":
    main()
