import sys
import argparse
from pathlib import Path
import numpy as np

import matplotlib
if "--show" not in sys.argv:  # окно - только по --show, иначе картинка без экрана
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from measure_height import load_scan, ScanResult
import make_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--max-height", type=float, default=15.0)
    ap.add_argument("--min-score", type=float, default=0.35, help="Порог ZNCC при расчете карты")
    ap.add_argument("--min-point-score", type=float, default=0.5, help="Отсечь готовые точки с уверенностью ниже этой")
    ap.add_argument("--win", type=int, default=9)
    ap.add_argument("--map-views", type=int, default=0, help="лимит дополнительных ракурсов (0 — все)")
    ap.add_argument("--min-map-views", type=int, default=4, help="минимум ракурсов для точки карты")

    # Настройки SOR (фильтра выбросов)
    ap.add_argument("--sor-k", type=int, default=20, help="Сколько соседей проверять (больше = строже)")
    ap.add_argument("--sor-scale", type=float, default=1.5, help="Порог отсечения (меньше = строже)")
    ap.add_argument("--no-filled", action="store_true",
                    help="Не показывать достроенное по плоскости (однотонная маска платы, места под бликом)")
    ap.add_argument("--show", action="store_true",
                    help="Открыть окно с 3D-картой: её можно вращать мышью (PNG сохраняется в любом случае)")
    ap.add_argument("--show-thin", type=int, default=4,
                    help="Окно (--show): показывать каждую N-ю точку, чтобы поворот шёл плавнее; PNG - полный")
    ap.add_argument("--view", choices=("points", "model"), default="points",
                    help="points - облако точек, как раньше (для него --downsample, SOR, --no-filled, --show-thin); "
                         "model - достроенная 2.5D-модель: верх целиком, стенки до стола (STL - z/make_model.py)")

    args_cli = ap.parse_args()
    if args_cli.downsample < 1 or args_cli.sor_k < 2 or not np.isfinite(args_cli.sor_scale) or args_cli.sor_scale <= 0:
        ap.error("--downsample >= 1, --sor-k >= 2, --sor-scale > 0")

    print(f"Загрузка скана {args_cli.scan_dir}...")
    frames, meta = load_scan(args_cli.scan_dir)

    args = argparse.Namespace(
        sweep_step=0.25, below=3.0, max_height=args_cli.max_height, map_views=args_cli.map_views,
        win=args_cli.win, min_texture=2.0, min_score=args_cli.min_score, min_map_views=args_cli.min_map_views,
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

    if args_cli.view == "model":
        top = make_model.build_model(scan)
        if top is None:
            print("Объекта выше стола не найдено.")
            return
        for line in make_model.describe(top, scan):
            print(line)
        title = f"2.5D-модель\n{args_cli.scan_dir}"
        out_png = Path(args_cli.scan_dir) / "plot_3d_model.png"
        make_model.render(top, out_png, title=title)
        print(f"УСПЕХ! Модель сохранена в: {out_png}")
        if args_cli.show:
            make_model.render(top, show=True, title=title)
        return

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
    if len(X_3d) > 1:
        print(f"Фильтрация выбросов (SOR, k={args_cli.sor_k}, scale={args_cli.sor_scale})...")
        tree = cKDTree(X_3d)
        dists, _ = tree.query(X_3d, k=min(len(X_3d), args_cli.sor_k))
        mean_dists = np.mean(dists[:, 1:], axis=1)
        threshold = np.median(mean_dists) * args_cli.sor_scale
        good_mask = mean_dists <= threshold

        X_3d = X_3d[good_mask]
        H = H[good_mask]
        print(f"Удалено выбросов: {len(good_mask) - np.sum(good_mask)}")
        print(f"Осталось точек: {len(X_3d)}")

    if len(X_3d) == 0:
        print("После фильтрации не осталось точек.")
        return

    # Достроенное по плоскости (не замер): серым, реже и прозрачнее измеренного
    filled = getattr(scan, "filled", None)
    Xf = np.empty((0, 3))
    Hf = np.empty(0)
    if filled is not None and filled.any() and not args_cli.no_filled:
        Xf = scan.X[filled][::step * 4]
        Hf = scan.ht[filled][::step * 4]
        keep = (Hf > -1.0) & (Hf < args_cli.max_height)
        Xf, Hf = Xf[keep], Hf[keep]
        print(f"Достроенных точек (показано): {len(Xf)}")

    # Размах осей - по всей карте, чтобы копия для окна выглядела так же, как полная
    X_all = np.vstack([X_3d, Xf]) if len(Xf) else X_3d
    x_range = max(X_all[:, 0].max() - X_all[:, 0].min(), 1.0)
    y_range = max(X_all[:, 1].max() - X_all[:, 1].min(), 1.0)
    z_range = max(args_cli.max_height - (-1), 0.2 * x_range)
    max_range = max(x_range, y_range, z_range)
    aspect = (x_range / max_range, y_range / max_range, z_range / max_range)

    # Отрисовка
    def draw(Xp, Hp, Xq, Hq):
        """Измеренное (Xp, Hp) цветом по высоте, достроенное (Xq, Hq) серым."""
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if len(Xq):
            ax.scatter(Xq[:, 0], -Xq[:, 1], Hq, c='0.55', s=1, alpha=0.25, depthshade=False,
                       label='достроено по плоскости (не замер)')
        sc = ax.scatter(Xp[:, 0], -Xp[:, 1], Hp,
                        c=Hp, cmap='turbo', s=3, alpha=0.8)
        if len(Xq):
            ax.legend(loc='upper left')

        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Высота, мм')

        ax.set_xlabel('X, мм')
        ax.set_ylabel('-Y, мм')
        ax.set_zlabel('Высота, мм')
        ax.set_zlim(-1, args_cli.max_height)
        ax.set_box_aspect(aspect)
        ax.view_init(elev=25, azim=-60)
        return fig

    out_png = Path(args_cli.scan_dir) / "plot_3d_clean.png"
    fig = draw(X_3d, H, Xf, Hf)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"УСПЕХ! График сохранен в: {out_png}")
    if args_cli.show:
        if matplotlib.get_backend().lower() == "agg":
            print("Окна нет: matplotlib не нашёл GUI-бэкенд (нужен tkinter или PyQt). PNG сохранён.")
        else:
            # В окне точек меньше, поворот идёт плавнее; PNG выше остаётся полным
            k = max(1, args_cli.show_thin)
            plt.close(fig)
            draw(X_3d[::k], H[::k], Xf[::k], Hf[::k])
            n_show = len(X_3d[::k]) + len(Xf[::k])
            print(f"Окно открыто: {n_show} точек (каждая {k}-я), вращать мышью. Закрыть окно - вернуться в терминал")
            plt.show()

if __name__ == "__main__":
    main()
