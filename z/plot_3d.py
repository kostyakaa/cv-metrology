import sys
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from measure_height import load_scan, ScanResult

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--max-height", type=float, default=15.0)
    ap.add_argument("--min-score", type=float, default=0.35, help="Порог ZNCC при расчете карты")
    ap.add_argument("--min-point-score", type=float, default=0.5, help="Отсечь готовые точки с уверенностью ниже этой")
    ap.add_argument("--win", type=int, default=9)
    
    # Настройки SOR (фильтра выбросов)
    ap.add_argument("--sor-k", type=int, default=20, help="Сколько соседей проверять (больше = строже)")
    ap.add_argument("--sor-scale", type=float, default=1.5, help="Порог отсечения (меньше = строже)")
    
    args_cli = ap.parse_args()

    print(f"Загрузка скана {args_cli.scan_dir}...")
    frames, meta = load_scan(args_cli.scan_dir)

    args = argparse.Namespace(
        sweep_step=0.25, below=3.0, max_height=args_cli.max_height, map_views=0,
        win=args_cli.win, min_texture=2.0, min_score=args_cli.min_score, min_views=2,
        table_near=3.0, min_height=0.5, min_area=4.0, click_range=1.0,
        patch=31, max_sigma=0.05,
    )

    g = meta["guess"]
    gp = meta.get("global_plane")
    
    print("Пересчет карты высот...")
    scan = ScanResult(
        frames, meta["positions"], meta["ref_index"],
        meta["camera_matrix"], np.array(meta["dist_coeffs"]), meta["model"],
        meta["V_cam"], (np.array(g["n"]), g["d"], g["kind"]),
        None if gp is None else (np.array(gp["n"]), gp["d"]),
        args, log=lambda m: print(f"  [ScanResult] {m}")
    )

    valid = scan.valid
    if not np.any(valid):
        print("Нет валидных точек!")
        return

    # Извлекаем точки и их уверенность
    X_3d = scan.X[valid]
    H = scan.ht[valid]
    Scores = scan.score[valid] # <-- Достаем уверенность ZNCC

    # Отсекаем по высоте и уверенности
    mask = (H > -1.0) & (H < args_cli.max_height) & (Scores >= args_cli.min_point_score)
    X_3d = X_3d[mask]
    H = H[mask]

    # Прореживаем
    step = args_cli.downsample
    X_3d = X_3d[::step] 
    H = H[::step]

    print(f"Точек до SOR: {len(X_3d)}")

    # --- SOR Фильтр ---
    if len(X_3d) > 0:
        print(f"Фильтрация выбросов (SOR, k={args_cli.sor_k}, scale={args_cli.sor_scale})...")
        pts_3d = np.column_stack((X_3d, H))
        tree = cKDTree(pts_3d)
        dists, _ = tree.query(pts_3d, k=args_cli.sor_k)
        mean_dists = np.mean(dists[:, 1:], axis=1)
        threshold = np.median(mean_dists) * args_cli.sor_scale
        good_mask = mean_dists < threshold
        
        X_3d = X_3d[good_mask]
        H = H[good_mask]
        print(f"Удалено выбросов: {len(good_mask) - np.sum(good_mask)}")
        print(f"Осталось точек: {len(X_3d)}")

    if len(X_3d) == 0:
        print("После фильтрации не осталось точек.")
        return

    # Отрисовка
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(X_3d[:, 0], -X_3d[:, 1], H, 
                    c=H, cmap='turbo', s=3, alpha=0.8)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Высота, мм')
    
    ax.set_xlabel('X, мм')
    ax.set_ylabel('-Y, мм')
    ax.set_zlabel('Высота, мм')
    ax.set_zlim(-1, args_cli.max_height)
    
    x_range = max(X_3d[:, 0].max() - X_3d[:, 0].min(), 1.0)
    y_range = max(X_3d[:, 1].max() - X_3d[:, 1].min(), 1.0)
    z_range = max(args_cli.max_height - (-1), 0.2 * x_range)
    max_range = max(x_range, y_range, z_range)
    
    ax.set_box_aspect((
        x_range / max_range,
        y_range / max_range,
        z_range / max_range,
    ))
    
    ax.view_init(elev=25, azim=-60)

    out_png = Path(args_cli.scan_dir) / "plot_3d_clean.png"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"УСПЕХ! График сохранен в: {out_png}")

if __name__ == "__main__":
    main()