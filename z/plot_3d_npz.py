"""Plot a 2.5D mesh directly from an existing cv-metrology area.npz / surfaces.npz.

No camera, stage, scan or height reconstruction is used.

Examples:
    python z/plot_3d_npz.py bench/area-robust-035/area.npz --show
    python z/plot_3d_npz.py bench/area-robust-035/area.npz --mode measured --show
"""
import argparse
from pathlib import Path

import numpy as np


def make_triangles(height, allowed, max_z_jump):
    """Mesh existing neighbouring XY grid samples, never interpolate missing pixels."""
    h, w = height.shape
    base = np.arange(h * w, dtype=np.int64).reshape(h, w)
    a = base[:-1, :-1].reshape(-1)
    b = base[:-1, 1:].reshape(-1)
    c = base[1:, :-1].reshape(-1)
    d = base[1:, 1:].reshape(-1)
    triangles = np.concatenate((np.column_stack((a, c, b)),
                                np.column_stack((b, c, d))), axis=0)
    flat_ok = allowed.reshape(-1)
    ok = flat_ok[triangles].all(axis=1)
    triangles = triangles[ok]
    if max_z_jump is not None and triangles.size:
        values = height.reshape(-1)[triangles]
        triangles = triangles[np.ptp(values, axis=1) <= max_z_jump]
    return triangles


def main():
    parser = argparse.ArgumentParser(description="Show a 3D surface from an EXISTING NPZ height map, without re-computing heights.")
    parser.add_argument("npz", type=Path, help="Existing area.npz or surfaces.npz")
    parser.add_argument("--show", action="store_true", help="Open interactive 3D window")
    parser.add_argument("--out", type=Path, help="PNG file (default: artifacts/<scan_folder>_<input>_<mode>_3d.png)")
    parser.add_argument("--mode", choices=("confirmed", "measured", "model"), default="confirmed",
                        help="confirmed: trusted samples only (default); measured: all measured samples, incl. conflicts; "
                             "model: model_mm if present, potentially inferred")
    parser.add_argument("--max-z-jump", type=float, default=3.0,
                        help="Do not join adjacent samples with vertical difference greater than N mm (default: 3); use 0 to disable")
    parser.add_argument("--z-exaggeration", type=float, default=1.0,
                        help="Vertical exaggeration for visualisation only (default: 1)")
    args = parser.parse_args()
    if args.z_exaggeration <= 0 or args.max_z_jump < 0:
        parser.error("--z-exaggeration must be positive and --max-z-jump non-negative")
    if not args.npz.is_file():
        parser.error(f"NPZ file not found: {args.npz}")

    with np.load(args.npz, allow_pickle=False) as d:
        for field in ("height_mm", "xy_origin", "cell_mm"):
            if field not in d:
                parser.error(f"{args.npz} is missing required field {field!r}")
        raw_height = np.asarray(d["height_mm"], dtype=np.float64)
        origin = np.asarray(d["xy_origin"], dtype=np.float64).reshape(-1)
        cell = float(d["cell_mm"])
        if args.mode == "model" and "model_mm" in d:
            height = np.asarray(d["model_mm"], dtype=np.float64)
        else:
            height = raw_height
        if "confirmed" in d:
            confirmed = np.asarray(d["confirmed"], dtype=bool)
        else:
            confirmed = np.isfinite(raw_height)

    if height.ndim != 2 or height.shape != raw_height.shape or confirmed.shape != height.shape:
        parser.error("Height and confirmed grids must be 2D and have the same shape")
    if len(origin) != 2 or not np.all(np.isfinite(origin)) or not np.isfinite(cell) or cell <= 0:
        parser.error("Invalid xy_origin or cell_mm")
    if args.mode == "confirmed":
        valid = np.isfinite(height) & confirmed
    else:
        valid = np.isfinite(height)
    triangles = make_triangles(height, valid, args.max_z_jump or None)
    if len(triangles) == 0:
        parser.error("No adjacent valid samples could form a surface. Try --mode measured or --max-z-jump 0")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ny, nx = height.shape
    gy, gx = np.indices(height.shape)
    x = origin[0] + gx.reshape(-1) * cell
    y = origin[1] + gy.reshape(-1) * cell
    z = height.reshape(-1)
    vertices = np.stack([x[triangles], y[triangles], z[triangles]], axis=-1)
    finite_values = height[valid]
    lo, hi = np.percentile(finite_values, [2, 98])
    if hi <= lo:
        lo, hi = float(finite_values.min()), float(finite_values.max())
        if hi <= lo:
            hi = lo + 1
    norm = Normalize(vmin=lo, vmax=hi, clip=True)
    cmap = plt.get_cmap("turbo")
    facecolors = cmap(norm(vertices[:, :, 2].mean(axis=1)))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices, facecolors=facecolors, edgecolors="none", linewidths=0, alpha=1.0)
    ax.add_collection3d(mesh)
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    z0, z1 = float(finite_values.min()), float(finite_values.max())
    dz = max(z1 - z0, cell)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0 - 0.05 * dz, z1 + 0.05 * dz)
    ax.set_box_aspect((max(x1-x0,cell), max(y1-y0,cell), dz * args.z_exaggeration))
    ax.set_xlabel("X, мм")
    ax.set_ylabel("Y, мм")
    ax.set_zlabel("Z, мм")
    ax.set_title(f"3D поверхность из NPZ — {args.mode} ({len(triangles):,} треугольников)")
    ax.view_init(elev=30, azim=-60)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=.55, pad=.12)
    bar.set_label("Высота, мм")
    out = args.out or (Path("artifacts") / f"{args.npz.parent.name}_{args.npz.stem}_{args.mode}_3d.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"NPZ: {args.npz}")
    print(f"Режим: {args.mode}; исходных ячеек: {height.size}; показано точек: {valid.sum()}; треугольников: {len(triangles)}")
    print(f"Картинка: {out}")
    print("Высоты не пересчитывались. Пропуски не заполнялись.")
    if args.show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
