"""
3D-модель объекта по скану: верхняя поверхность целиком + стенки до стола + дно -> замкнутое тело, STL (мм).

Сверху камера видит только верх объекта, поэтому модель "2.5D": под каждой точкой верха - сплошной материал
до стола, боковые стенки вертикальные, нависаний и полостей сбоку в ней нет.

Как строится:
  1. Высоты над столом (плоскость меток; если её нет - стол в кадре) по карте скана: измеренное и
     достроенное плоскостью. Одиночные ложные пятна отбрасываются.
  2. Объект: карта режется по скачкам высоты на куски. От куска у центра кадра (камеру при скане ставят над
     объектом) растим: вниз - касается и ниже (опора: плата под объективом), вверх - касается и лежит внутри
     контура (детали на плате). Ложные полосы в пазах стола и мусор у краёв кадра в объект не попадают.
  3. Дыры внутри объекта (однотонное, блик, края деталей) интерполируются по краям. Где сквозь дыру виден
     стол - это настоящее отверстие, оно остаётся.
  4. Вид строго сверху на сетке --res мм. Закрытое от камеры высокими деталями - уровнем фона рядом.
  5. Сетка: верх, стенки по контуру, дно на столе.

Оси модели: x - вправо по кадру, y - вверх по кадру, z - вверх от стола; начало - точка стола под камерой
в опорной точке скана.

Запуск:
    python z/make_model.py scans/scan_<время>            # model.stl и model.png в папке скана
    python z/make_model.py scans/scan_<время> --show     # и окно, модель можно вращать мышью
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from measure_height import ScanResult, add_height_args, load_scan  # noqa: E402
from stereo import points_from_heights  # noqa: E402

UNKNOWN = -32000


def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(mask):
    """Маска вместе с дырами (тем, что не связано с краем кадра)."""
    m = np.pad(mask.astype(np.uint8), 1)
    cv2.floodFill(m, None, (0, 0), 2)
    return m[1:-1, 1:-1] != 2


def outer_ids(lab, n, extra=None):
    """Метки кусков, касающихся края кадра (и тех, где extra)."""
    out = np.zeros(n, bool)
    out[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
    if extra is not None:
        out[np.unique(lab[extra])] = True
    return out


# ---------- высоты и объект (оси опорного кадра, половинное разрешение)
def table_heights(sc, speckle_mm2=1.0, speckle_diff=0.5):
    """Высоты над тем же столом, который выбрал ScanResult. Одиночные ложные пятна
    (кусок, не стыкующийся по высоте с соседями, меньше speckle_mm2)
    выкидываются. Возвращает (высоты, известно, (n, d) стола, px на мм)."""
    nb, db = np.asarray(sc.n_t, float), float(sc.d_t)
    px_mm = sc.Ks[0, 0] / db
    known = (sc.valid | sc.filled) & np.isfinite(sc.X[..., 0])
    h = np.where(known, db - sc.X @ nb, np.nan).astype(np.float32)
    q = np.where(known, np.round(h * 100), UNKNOWN).astype(np.int16)
    cv2.filterSpeckles(q, UNKNOWN, max(1, int(speckle_mm2 * px_mm ** 2)), int(round(speckle_diff * 100)))
    return h, q != UNKNOWN, (nb, db), px_mm


def height_pieces(h, mask, jump):
    """Куски: связные области mask, внутри которых соседние высоты отличаются не больше jump (пиксели на скачке ни
    в какой кусок не входят). Возвращает (метки, число кусков, статистика cv2.connectedComponentsWithStats)."""
    hk = np.where(mask, h, np.nan)
    g = np.zeros(h.shape, np.float32)
    rows, cols = h.shape
    for dy, dx in ((0, 1), (1, 0)):
        d = np.nan_to_num(np.abs(hk[:rows - dy, :cols - dx] - hk[dy:, dx:]), nan=0)
        g[:rows - dy, :cols - dx] = np.maximum(g[:rows - dy, :cols - dx], d)
        g[dy:, dx:] = np.maximum(g[dy:, dx:], d)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask & (g <= jump)).astype(np.uint8), connectivity=4)
    return lab, n - 1, stats


def select_object(h, known, px_mm, above=0.3, jump=1.0, adj_mm=1.0, tol=0.5, support_share=0.1, min_half_width=0.75):
    """Объект от центра кадра. Основа - самый большой кусок, чья рамка накрывает центр (кольцо вокруг дырки -
    тоже), иначе ближайший к центру. Дальше, пока растётся, берётся кусок, касающийся выбранного, если он
      - внутри контура выбранного и не ниже соседнего выбранного (деталь сверху), или
      - ниже соседнего выбранного и широкий: площадь не меньше support_share выбранного, толщина не меньше
        2 * min_half_width мм (опора под деталью, дно кармана). Узкие полоски (пазы стола) и низкий мусор в
        дырках - нет.
    Маска объекта или None."""
    lab, n, stats = height_pieces(h, known & (h > above), jump)
    if n == 0:
        return None
    area = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    area[0] = 0
    cy, cx = h.shape[0] / 2, h.shape[1] / 2
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    w, hh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    half_width = np.zeros(n + 1, np.float32)
    np.maximum.at(half_width, lab.ravel(), cv2.distanceTransform((lab > 0).astype(np.uint8), cv2.DIST_L2, 3).ravel())
    areal = (area >= 4*px_mm**2) & (half_width >= min_half_width*px_mm)
    candidates = areal if areal.any() else area > 0
    cover = np.nonzero((x <= cx) & (cx < x + w) & (y <= cy) & (cy < y + hh) & candidates)[0]
    if len(cover):
        seed = cover[np.argmax(area[cover])]
    else:
        distance = np.hypot(np.maximum(np.maximum(x-cx,cx-(x+w)),0),
                            np.maximum(np.maximum(y-cy,cy-(y+hh)),0))
        # The centre can be inside a U-shaped opening. A nearby sliver must not win
        # against a well supported surface with a slightly more distant boundary.
        seed = np.argmax(np.where(candidates,area/(1+distance/(8*px_mm))**2,0))
    kept = np.zeros(n + 1, bool)
    kept[seed] = True
    ker = disk(adj_mm * px_mm)
    surface_ker = disk(max(adj_mm,4.) * px_mm)
    heights = np.zeros(n+1)
    for i in np.flatnonzero(candidates):
        heights[i] = np.median(h[lab == i])
    hz = np.nan_to_num(h, nan=0).astype(np.float32)
    for _ in range(100):
        km = kept[lab] & (lab > 0)
        near = cv2.dilate(km.astype(np.uint8), ker) > 0
        touch = np.bincount(lab[near & ~km], minlength=n + 1) > 0
        surface = cv2.dilate(km.astype(np.uint8),surface_ker)>0
        near_surface = np.bincount(lab[surface & ~km],minlength=n+1)>0
        inside = np.bincount(lab[fill_holes(km)], minlength=n + 1) / np.maximum(area, 1)
        top = cv2.dilate(np.where(km, hz, -1e9).astype(np.float32), ker)  # высота выбранного рядом
        contact = (lab > 0) & ~km & (top > -1e8)
        lower = np.bincount(lab[contact & (hz < top - tol)], minlength=n + 1) / \
            np.maximum(np.bincount(lab[contact], minlength=n + 1), 1)
        wide = (area >= max(4.0 * px_mm ** 2, support_share * area[kept].sum())) & \
            (half_width >= min_half_width * px_mm)
        add = touch & ~kept & (((inside >= 0.5) & (lower <= 0.5)) | ((lower > 0.5) & wide))
        add |= near_surface & candidates & ~kept & (np.abs(heights-heights[seed]) <= tol)
        add[0] = False
        if not add.any():
            break
        kept |= add
    return kept[lab] & (lab > 0)


def pull_push(val, known):
    """Гладкое заполнение неизвестного по известному: пирамида вверх - средние по известному, вниз - там, где своих
    данных мало, подставляется значение с более грубого уровня."""
    v = np.where(known, val, 0).astype(np.float32)
    w = known.astype(np.float32)
    pyr = [(v, w)]
    while min(v.shape) > 4:
        vw, w = cv2.pyrDown(v * w), cv2.pyrDown(w)
        v = np.where(w > 1e-6, vw / np.maximum(w, 1e-6), 0).astype(np.float32)
        pyr.append((v, w))
    out = pyr[-1][0]
    for v, w in reversed(pyr[:-1]):
        a = np.clip(2 * w, 0, 1)
        out = a * v + (1 - a) * cv2.pyrUp(out, dstsize=(v.shape[1], v.shape[0]))
    return out


def fill_object(h, known, obj, above=0.3):
    """Дыры внутри объекта: где сквозь дыру виден стол - настоящее отверстие, остаётся; иначе (однотонное, блик,
    скачок на краю детали) - часть объекта, высоты гладко интерполируются по объекту вокруг (pull_push).
    Возвращает (объект с дырами, высоты, интерполировано, виден стол)."""
    table = known & (h <= above) & ~obj
    n, lab = cv2.connectedComponents((~obj).astype(np.uint8), connectivity=4)
    out = outer_ids(lab, n, table)
    full = obj | ((lab > 0) & ~out[lab])
    miss = full & ~obj
    hf = np.where(obj, h, 0).astype(np.float32)
    if miss.any():
        hf[miss] = pull_push(h, obj)[miss]
    return full, hf, miss, table


# ---------- вид сверху
def top_view(sc, h, full, interp, table, plane, res=0.25, above=0.3, min_part=0.75):
    """Вид строго сверху: точки верха -> клетки res мм на столе, в клетке - самая высокая точка. Пустые клетки,
    окружённые объектом (закрыто от камеры высокими деталями, щели), - уровнем фона: минимум соседних клеток
    (видимый стол - 0); что не выше above - стол. Шум - медиана 3x3 внутри объекта; полоски тоньше 2 * min_part мм
    и мелкие обрывки убираются.
    Возвращает dict: z (NaN - пусто), src (2 - измерено, 1 - достроено плоскостью, 0 - интерполировано),
    x0, y1 (координаты края сетки), res, axes (начало и оси x, y, z модели в осях опорной камеры)."""
    nb, db = plane
    ez = -nb
    ex = np.array([1.0, 0.0, 0.0]) - nb[0] * nb
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    p0 = np.array([0.0, 0.0, db / nb[2]])  # стол на оси объектива
    X = points_from_heights(np.where(full, h, 0).astype(np.float32), sc.Ks, nb, db) - p0
    ys, xs = np.nonzero(full)
    P = X[ys, xs]
    x, y, z = P @ ex, P @ ey, h[ys, xs].astype(np.float64)
    src = np.where(interp[ys, xs], 0, np.where(sc.valid[ys, xs], 2, 1)).astype(np.uint8)
    x0, y1 = x.min(), y.max()
    W, H = int((x.max() - x0) / res) + 1, int((y1 - y.min()) / res) + 1
    cell = ((y1 - y) / res).astype(np.int64) * W + ((x - x0) / res).astype(np.int64)
    order = np.lexsort((-z, cell))  # по клетке, в клетке - сверху вниз
    cells, first = np.unique(cell[order], return_index=True)
    zg = np.full(H * W, np.nan)
    zg[cells] = z[order[first]]
    sg = np.zeros(H * W, np.uint8)
    sg[cells] = src[order[first]]
    zg, sg = zg.reshape(H, W), sg.reshape(H, W)
    obs = np.isfinite(zg)
    # где виден стол (настоящие дыры и край объекта)
    Pt = X[table]
    tj = np.floor((Pt @ ex - x0) / res).astype(np.int64)
    ti = np.floor((y1 - Pt @ ey) / res).astype(np.int64)
    ok = (tj >= 0) & (tj < W) & (ti >= 0) & (ti < H)
    tg = np.zeros((H, W), bool)
    tg[ti[ok], tj[ok]] = True
    tg &= ~obs
    # пустые клетки внутри - минимумом соседей
    n, gl = cv2.connectedComponents((~obs & ~tg).astype(np.uint8), connectivity=4)
    inner = ~outer_ids(gl, n)
    inner[0] = False
    gl = np.where(inner[gl], gl, 0)
    if gl.any():
        near = cv2.dilate(gl.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int64)
        around = (obs | tg) & (near > 0)
        zmin = np.full(n, np.inf)
        np.minimum.at(zmin, near[around], np.where(obs, zg, 0.0)[around])
        fill = zmin[gl]
        put = (gl > 0) & np.isfinite(fill) & (fill > above)
        zg[put] = fill[put]
        sg[put] = 0
    occ = np.isfinite(zg)
    inner3 = cv2.erode(occ.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    zm = cv2.medianBlur(np.nan_to_num(zg, nan=0).astype(np.float32), 3)
    zg = np.where(inner3, zm, zg)
    zg[occ & (zg <= above)] = np.nan
    # тоньше ~2 * min_part мм (ложные полоски вдоль пазов стола) - прочь, мелкие обрывки - тоже
    occ = np.isfinite(zg).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, disk(min_part / res))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    if n == 1:
        zg[:] = np.nan
    else:
        area = stats[:, cv2.CC_STAT_AREA]
        keep = area >= max(4.0 / res ** 2, 0.01 * area[1:].max())
        keep[0] = False
        zg[~keep[lab]] = np.nan
    return {"z": zg, "src": sg, "x0": x0, "y1": y1, "res": res, "axes": (p0, ex, ey, ez)}


# ---------- сетка
def grid_mesh(z, x0, y1, res, solid=True):
    """Сетка по клеткам: квадрат из 4 соседних заполненных клеток - 2 треугольника верха. solid - ещё дно на столе
    (z = 0) и стенки по краю: замкнутое тело, обход треугольников - наружу. Возвращает (вершины (N, 3), (M, 3))."""
    occ = np.isfinite(z)
    q = occ[:-1, :-1] & occ[:-1, 1:] & occ[1:, :-1] & occ[1:, 1:]
    for _ in range(20):  # квадраты, касающиеся только углом, - убрать один (иначе ребро на 4 треугольника)
        p1 = q[:-1, :-1] & q[1:, 1:] & ~q[:-1, 1:] & ~q[1:, :-1]
        p2 = q[:-1, 1:] & q[1:, :-1] & ~q[:-1, :-1] & ~q[1:, 1:]
        if not (p1.any() or p2.any()):
            break
        q[1:, 1:][p1] = False
        q[1:, :-1][p2] = False
    used = np.zeros_like(occ)
    used[:-1, :-1] |= q
    used[:-1, 1:] |= q
    used[1:, :-1] |= q
    used[1:, 1:] |= q
    n = int(used.sum())
    vid = np.full(z.shape, -1, np.int64)
    vid[used] = np.arange(n)
    ii, jj = np.nonzero(used)
    xy = np.column_stack([x0 + (jj + 0.5) * res, y1 - (ii + 0.5) * res])
    V = np.column_stack([xy, z[ii, jj]])
    qi, qj = np.nonzero(q)
    a, b = vid[qi, qj], vid[qi, qj + 1]            # ряд квадрата с большим y
    d, c = vid[qi + 1, qj], vid[qi + 1, qj + 1]    # ряд с меньшим y
    top = np.concatenate([np.column_stack([d, c, b]), np.column_stack([d, b, a])])  # против часовой сверху
    if not solid:
        return V, top
    V = np.concatenate([V, np.column_stack([xy, np.zeros(n)])])
    bottom = top[:, ::-1] + n
    e = np.concatenate([np.column_stack([d, c]), np.column_stack([c, b]), np.column_stack([b, a]),
                        np.column_stack([a, d])])
    rim = e[~np.isin(e[:, 1] * n + e[:, 0], e[:, 0] * n + e[:, 1])]  # ребро края: нет пары в обратную сторону
    u, v = rim[:, 0], rim[:, 1]
    walls = np.concatenate([np.column_stack([u, u + n, v]), np.column_stack([v, u + n, v + n])])
    return V, np.concatenate([top, bottom, walls])


def write_stl(path, V, F, name="cv-metrology model"):
    """Двоичный STL (мм)."""
    tri = V[F].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as f:
        f.write(name.encode("ascii", "replace")[:80].ljust(80, b" "))
        f.write(np.uint32(len(F)).tobytes())
        rec.tofile(f)


def build_model(sc, res=0.25, above=0.3, min_part=0.75):
    """Скан -> вид сверху (dict top_view) + сетка замкнутого тела (V, F) или None, если объекта нет."""
    h, known, plane, px_mm = table_heights(sc)
    obj = select_object(h, known, px_mm, above=above)
    if obj is None or not obj.any():
        return None
    full, hf, interp, table = fill_object(h, known, obj, above=above)
    top = top_view(sc, hf, full, interp, table, plane, res=res, above=above, min_part=min_part)
    if not np.isfinite(top["z"]).any():
        return None
    top["V"], top["F"] = grid_mesh(top["z"], top["x0"], top["y1"], res)
    return top


def describe(top, sc):
    """Строки сводки: размер объекта, откуда верх (измерено / достроено плоскостью / интерполировано), оси."""
    z, src, res = top["z"], top["src"], top["res"]
    occ = np.isfinite(z)
    rows, cols = np.nonzero(occ)
    share = [100 * ((src == k) & occ).sum() / occ.sum() for k in (2, 1, 0)]
    return [f"объект {np.ptp(cols) * res:.1f} x {np.ptp(rows) * res:.1f} мм, высота до {np.nanmax(z):.2f} мм",
            f"отсчёт высоты: {sc.ref_kind}",
            f"верх: измерено {share[0]:.0f}%, достроено плоскостью {share[1]:.0f}%, интерполировано {share[2]:.0f}%",
            f"начало координат: стол под камерой в опорной точке скана (станок X{sc.c_ref[0]:.3f} Y{sc.c_ref[1]:.3f}); "
            f"x - вправо по кадру, y - вверх по кадру, z - вверх от стола"]


def model_figure(top, max_quads=20000, title=""):
    """Фигура matplotlib: модель как тело - верх цветом по высоте, стенки серые, тень от света сверху-слева.
    Сетка прорежена до ~max_quads квадратов верха (matplotlib рисует медленно)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    z = top["z"]
    step = max(1, int(np.ceil(np.sqrt(np.isfinite(z).sum() / max_quads))))
    V, F = grid_mesh(z[::step, ::step], top["x0"], top["y1"], top["res"] * step)
    n = len(V) // 2
    F = F[(F < n).any(axis=1)]  # дно снизу не видно - не рисуем
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    norm = colors.Normalize(0, float(np.nanmax(z)))
    cmap = plt.get_cmap("turbo")
    fc = cmap(norm(tri[:, :, 2].mean(axis=1)))
    fc[(F >= n).any(axis=1), :3] = 0.6  # стенки
    light = np.array([-0.4, 0.6, 1.0])
    light /= np.linalg.norm(light)
    fc[:, :3] *= (0.45 + 0.55 * np.clip(nrm @ light, 0, 1))[:, None]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, linewidths=0))
    lo, hi = V.min(axis=0), V.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(0, hi[2])
    rng = hi - lo
    ax.set_box_aspect((rng[0], rng[1], max(rng[2], 0.05 * max(rng[0], rng[1]))))  # масштаб по осям одинаковый
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, pad=0.08).set_label("Высота над столом, мм")
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.set_zlabel("z, мм")
    ax.set_title(title)
    ax.view_init(elev=35, azim=-60)
    return fig


def render(top, path=None, show=False, title="", max_quads=20000):
    """Картинка модели в файл и/или окно (вращать мышью; ~20 тыс. квадратов перерисовываются за ~0.07 с).
    Бэкенд matplotlib выбирает вызывающий: без окна - Agg до импорта pyplot."""
    import matplotlib
    import matplotlib.pyplot as plt
    if path is not None:
        fig = model_figure(top, max_quads, title)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    if show:
        if matplotlib.get_backend().lower() == "agg":
            print("окна нет: matplotlib не нашёл GUI-бэкенд (нужен tkinter или PyQt)")
            return
        model_figure(top, max_quads, title)
        print("окно открыто: вращать мышью, закрыть окно - вернуться в терминал")
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", help="папка скана (scans/scan_<время>)")
    ap.add_argument("--res", type=float, default=0.25, help="шаг сетки модели, мм")
    ap.add_argument("--above", type=float, default=0.3, help="объект - то, что выше стола больше чем на, мм")
    ap.add_argument("--min-part", type=float, default=0.75,
                    help="выступы контура тоньше 2 * N мм (ложные полоски вдоль пазов стола) убираются, мм")
    ap.add_argument("--out", default=None, help="куда писать model.stl и model.png (по умолчанию - папка скана)")
    ap.add_argument("--show", action="store_true", help="открыть окно с моделью (вращать мышью)")
    add_height_args(ap)  # параметры карты высот - те же, что в measure_height.py
    args = ap.parse_args()

    frames, meta = load_scan(args.scan_dir)
    g, gp = meta["guess"], meta.get("global_plane")
    t0 = time.time()
    sc = ScanResult(frames, meta["positions"], meta["ref_index"], meta["camera_matrix"], np.array(meta["dist_coeffs"]),
                    meta["model"], meta["V_cam"], (np.array(g["n"]), g["d"], g["kind"]),
                    None if gp is None else (np.array(gp["n"]), gp["d"]), args, log=lambda m: print("  " + m))
    top = build_model(sc, res=args.res, above=args.above, min_part=args.min_part)
    if top is None:
        print(f"объекта выше стола на {args.above:g} мм не найдено")
        return
    out = Path(args.out) if args.out else Path(args.scan_dir)
    out.mkdir(parents=True, exist_ok=True)
    stl = out / "model.stl"
    write_stl(stl, top["V"], top["F"])
    print(f"модель за {time.time() - t0:.1f} с")
    for line in describe(top, sc):
        print(line)
    print(f"STL: {stl} ({len(top['F'])} треугольников, замкнутое тело, мм)")
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")  # до импорта pyplot
    png = out / "model.png"
    render(top, png, title=Path(args.scan_dir).name)
    print(f"картинка: {png}")
    if args.show:
        render(top, show=True, title=Path(args.scan_dir).name)


if __name__ == "__main__":
    main()
