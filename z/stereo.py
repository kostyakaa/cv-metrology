"""Высота по кадрам с известными сдвигами камеры: станок двигает камеру чистым сдвигом в XY, поэтому
несколько кадров с разных точек - стереопара (многоракурсная) с базой, которую задаёт станок.

Оси: всё - в осях опорной камеры (кадр, снятый в исходной точке скана): x вправо по кадру, y вниз,
z по оси объектива. Кадры - без дисторсии с той же матрицей K (идеальная камера-обскура).
Плоскость n·X = d: n - единичная нормаль от камеры к столу, d - расстояние от камеры до плоскости.
Высота точки над плоскостью: h = d - n·X (к камере - плюс).
"""
import json
import os
import sys
import types
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.geometry import machine_motion  # noqa: E402


class SweepAborted(Exception):
    pass


# ---------- ход станка в осях камеры (из калибровки)
def find_frames_dir(calib, calib_path):
    """Папка кадров, по которым снята калибровка (у старых калибровок путь не записан - ищем по ходам)."""
    mode = calib.get("mode") or {}
    if mode.get("frames_dir") and (Path(mode["frames_dir"]) / "mode.json").exists():
        return Path(mode["frames_dir"])
    offsets = mode.get("machine_offsets")
    if not offsets:
        return None
    w, h = calib["image_size"]
    for d in sorted(Path(calib_path).parent.glob(f"frames_cam{mode.get('camera', '*')}_{w}x{h}_*"), reverse=True):
        try:
            m = json.loads((d / "mode.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("machine_offsets") == offsets:
            return d
    return None


def load_motion(calib, calib_path, log=print):
    """Ход станка в осях камеры. Если в калибровке его нет (снята до z/) - досчитывается по её же кадрам
    и дописывается в файл калибровки (только ключ machine_check.motion, остальное не меняется)."""
    mc = calib.get("machine_check") or {}
    if mc.get("motion"):
        return mc["motion"]
    frames = find_frames_dir(calib, calib_path)
    if frames is None:
        log("в калибровке нет хода станка в осях камеры и не найдены её кадры - нужна калибровка со станком "
            "(calibrate_distortion.py --grbl)")
        return None
    log(f"досчитываю ход станка в осях камеры по кадрам калибровки {frames.name} ...")
    sys.path.append(str(ROOT / "scripts"))
    import calibrate_distortion as cd
    from common.markers import make_charuco_board
    mode = json.loads((frames / "mode.json").read_text(encoding="utf-8"))
    board = make_charuco_board(**{k: v for k, v in mode["board"].items() if k != "name"})
    views, _, offsets = cd.views_from_dir(frames, types.SimpleNamespace(min_corners=8), board,
                                          cv2.aruco.CharucoDetector(board), mode.get("machine_offsets", {}))
    K, D = np.array(calib["camera_matrix"], float), np.array(calib["dist_coeffs"], float)
    res = machine_motion(views, offsets, K, D, calib["model"])
    if res is None:
        log("в кадрах калибровки нет сетки станком - ход в осях камеры не посчитать")
        return None
    path = Path(calib_path)
    st = path.stat()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("machine_check", {})["motion"] = res["motion"]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # калибровки выбираются по свежести - её не меняем
    calib.setdefault("machine_check", {})["motion"] = res["motion"]
    log(f"ход в осях камеры дописан в {Path(calib_path).name}")
    return res["motion"]


def motion_normal(V):
    n = np.cross(V[:, 0], V[:, 1])
    return n / np.linalg.norm(n) * (1 if n[2] > 0 else -1)


def under_camera(V, X):
    """Смещение станка (мм команды), при котором точка X (оси опорной камеры) окажется на оси объектива."""
    sol = np.linalg.solve(np.column_stack([V, [0.0, 0.0, 1.0]]), np.asarray(X, float))
    return sol[:2]


# ---------- кадры без дисторсии
class Rectifier:
    """Кадр -> кадр без дисторсии с той же K. valid - где в исправленном кадре есть данные."""

    def __init__(self, K, D, size, model="pinhole"):
        K = np.asarray(K, float)
        if model == "fisheye":
            self.m1, self.m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_32FC1)
        else:
            self.m1, self.m2 = cv2.initUndistortRectifyMap(K, D, None, K, size, cv2.CV_32FC1)
        ones = np.ones((size[1], size[0]), np.float32)
        v = cv2.remap(ones, self.m1, self.m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.valid = cv2.erode((v > 0.999).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def __call__(self, img):
        return cv2.remap(img, self.m1, self.m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def highpass(img, sigma):
    """Кадр без плавной составляющей: из него вычитается размытая копия (Гаусс sigma, px).
    Свет едет вместе с камерой, поэтому плавная засветка и блики в разных кадрах лежат по-разному. ZNCC
    убирает сдвиг и масштаб яркости в окне, а наклон (перепад поперёк окна) - нет: на слабой текстуре
    (чёрная маска платы) он перебивает настоящую текстуру и даёт ложные высоты. Линейный перепад размытие
    не меняет, поэтому он вычитается целиком, а мелкая текстура (по ней и идёт сопоставление) остаётся."""
    img = np.asarray(img, np.float32)
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def half(img, K):
    """Кадр и K вдвое меньше (INTER_AREA: пиксель i покрывает 2i, 2i+1 -> центр 2i+0.5)."""
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Ks = np.array(K, float).copy()
    Ks[0, 0] /= 2
    Ks[1, 1] /= 2
    Ks[0, 2] = (Ks[0, 2] - 0.5) / 2
    Ks[1, 2] = (Ks[1, 2] - 0.5) / 2
    return small, Ks


def plane_homography(K, O, n, d):
    """Пиксель опорного кадра -> пиксель кадра с центром камеры O для точек плоскости n·X = d."""
    return K @ (np.eye(3) - np.outer(O, n) / d) @ np.linalg.inv(K)


# ---------- плотная карта: перебор плоскостей, параллельных опорной
def plane_sweep(ref, others, Os, K, n, d0, heights, ref_valid=None, others_valid=None, win=9, min_std=2.0,
                min_score=0.5, min_views=2, tick=None, workers=None, strip=None):
    """Для каждого пикселя опорного кадра - высота h над плоскостью n·X = d0, при которой остальные кадры,
    перенесённые через плоскость n·X = d0 - h, лучше всего совпадают с опорным (средняя ZNCC в окне win).
    Возвращает (карта высот с NaN где не определилась, лучшая корреляция, маска достоверных).
    workers - потоков (по умолчанию min(ядер, 8)), strip - строк в полосе (по умолчанию кадр / workers).
    tick(i, n) вызывается из вызывающего потока; True - прервать (SweepAborted)."""
    hh, ww = ref.shape
    r = win // 2
    var_floor = np.float32(min_std ** 2)

    def box(a):
        return cv2.boxFilter(a, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REFLECT)

    # Яркость - от среднего кадра: ZNCC от сдвига яркости не зависит, а дисперсия как box(x²) - box(x)² во
    # float32 при яркостях ~200 теряет точность (вычитание близких ~4e4) - на однотонных местах это давало
    # "корреляцию" > 1, и такой мусор выигрывал перебор высот.
    ref = ref.astype(np.float32)
    ref = ref - np.float32(ref.mean())
    mr = box(ref)
    vr = np.maximum(box(ref * ref) - mr * mr, 0)
    texture = vr > var_floor
    if ref_valid is not None:
        texture &= cv2.erode(ref_valid.astype(np.uint8), np.ones((win, win), np.uint8)) > 0
    inv_sd = (1 / np.sqrt(np.maximum(vr, var_floor))).astype(np.float32)
    # Маска "окно целиком в данных" не зависит от уровня высоты - эрозия один раз на кадр, на уровне только
    # перенос (uint8, ближайший сосед; для переноса это то же, что box(перенесённая маска) == 1).
    kernel = np.ones((win, win), np.uint8)
    imgs, wins = [], []
    for img, m in zip(others, others_valid or [None] * len(others)):
        img = img.astype(np.float32)
        m = np.ones(img.shape, np.uint8) if m is None else m.astype(np.uint8)
        imgs.append(img - np.float32(img[m > 0].mean() if m.any() else img.mean()))
        wins.append(cv2.erode(m, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    Kinv = np.linalg.inv(K)
    heights = np.asarray(heights, float)
    nl = len(heights)
    # гомографии (выход -> вход) для каждого уровня и кадра
    Hs = [[K @ (np.eye(3) - np.outer(O, n) / (d0 - h)) @ Kinv for O in Os] for h in heights]
    lin = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    near = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP

    best = np.full((hh, ww), -2, np.float32)
    best_i = np.full((hh, ww), -1, np.int32)
    s_before = best.copy()
    s_after = best.copy()

    # Кадр режется на горизонтальные полосы, каждая считается целиком (все уровни, все кадры) в своём потоке.
    # Полоса - с запасом r строк сверху и снизу: окна её строк видят настоящих соседей, результат тот же, что по
    # целому кадру. Буферы полосы выделяются один раз и небольшие (кэш процессора), OpenCV и numpy на время
    # счёта отпускают GIL. Внутренний пул потоков OpenCV на это время выключен: на мелких вызовах он только
    # крутится вхолостую и мешает.
    # Больше 8 потоков не ускоряет (упирается в память и GIL); полос - по одной на поток.
    workers = max(1, int(workers or min(os.cpu_count() or 1, 8)))
    strip = max(int(strip or -(-hh // workers)), 2 * r + 1)
    bands = [(y0, min(y0 + strip, hh)) for y0 in range(0, hh, strip)]
    bands = [bd for bd in bands if texture[bd[0]:bd[1]].any()]  # без текстуры карта всё равно пустая
    progress = [0] * len(bands)
    stop = threading.Event()

    def run_band(k, y0, y1):
        a, b = max(0, y0 - r), min(hh, y1 + r)
        rows = b - a
        sz = (ww, rows)
        T = np.array([[1, 0, 0], [0, 1, a], [0, 0, 1]], float)  # строка полосы -> строка кадра
        Hb = [[H @ T for H in Hl] for Hl in Hs]
        ref_s, mr_s, isd_s = ref[a:b], mr[a:b], inv_sd[a:b]
        o, m_rows = y0 - a, y1 - y0
        bst = np.full((m_rows, ww), -2, np.float32)
        bi = np.full((m_rows, ww), -1, np.int32)
        sp, sb, sa = bst.copy(), bst.copy(), bst.copy()
        f32 = lambda: np.empty((rows, ww), np.float32)  # noqa: E731
        w, ok, ww2, rw, t1, t2, acc = f32(), np.empty((rows, ww), np.uint8), f32(), f32(), f32(), f32(), f32()
        cnt = np.empty((rows, ww), np.uint8)
        sc = np.empty((m_rows, ww), np.float32)
        for i in range(nl):
            if stop.is_set():
                return
            acc[:] = 0
            cnt[:] = 0
            for img, m_win, H in zip(imgs, wins, Hb[i]):
                cv2.warpPerspective(img, H, sz, dst=w, flags=lin)
                cv2.warpPerspective(m_win, H, sz, dst=ok, flags=near)
                np.multiply(w, w, out=ww2)
                np.multiply(ref_s, w, out=rw)
                mw, bww, brw = cv2.split(box(cv2.merge([w, ww2, rw])))  # box(w), box(w²), box(ref·w) - за проход
                # vw = max(box(w²) - mw², min_std²): однотонное окно против текстурного - ncc ~ 0
                np.multiply(mw, mw, out=t1)
                np.subtract(bww, t1, out=t1)
                np.maximum(t1, var_floor, out=t1)
                np.sqrt(t1, out=t1)
                np.multiply(mr_s, mw, out=t2)
                np.subtract(brw, t2, out=t2)
                np.multiply(t2, isd_s, out=t2)
                np.divide(t2, t1, out=t2)
                np.clip(t2, -1, 1, out=t2)
                cv2.add(acc, t2, dst=acc, mask=ok)
                cv2.add(cnt, 1, dst=cnt, mask=ok)
            a_s, c_s = acc[o:o + m_rows], cnt[o:o + m_rows]
            np.divide(a_s, np.maximum(c_s, 1), out=sc)
            sc[c_s < min_views] = -2
            if i > 0:
                m = bi == i - 1
                sa[m] = sc[m]
            better = sc > bst
            sb[better] = sp[better]
            bst[better] = sc[better]
            bi[better] = i
            sp, sc = sc, sp  # sc перезаписывается на следующем уровне - меняем буферы местами
            progress[k] = i + 1
        best[y0:y1], best_i[y0:y1], s_before[y0:y1], s_after[y0:y1] = bst, bi, sb, sa

    workers = min(workers, len(bands) or 1)
    cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(workers) as pool:
            futs = [pool.submit(run_band, k, y0, y1) for k, (y0, y1) in enumerate(bands)]
            try:
                shown = 0
                while True:
                    done, _ = wait(futs, timeout=0.05)
                    for f in done:
                        f.result()  # ошибка в потоке - сразу наружу
                    level = sum(progress) // len(bands) if bands else nl
                    if tick is not None:
                        for j in range(shown + 1, level + 1):
                            if tick(j, nl):
                                raise SweepAborted()
                    shown = max(shown, level)
                    if len(done) == len(futs):
                        break
            finally:
                stop.set()
    finally:
        cv2.setNumThreads(cv_threads)
    step = heights[1] - heights[0]
    den = s_before - 2 * best + s_after
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(den < -1e-6, 0.5 * (s_before - s_after) / den, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    hmap = (heights[0] + (best_i + delta) * step).astype(np.float32)
    valid = texture & (best > min_score) & (best_i > 0) & (best_i < len(heights) - 1)
    hmap[~valid] = np.nan
    return hmap, best, valid


def points_from_heights(hmap, K, n, d0):
    """3D точки (оси опорной камеры) пикселей карты высот над плоскостью n·X = d0."""
    hh, ww = hmap.shape
    u, v = np.meshgrid(np.arange(ww, dtype=np.float64), np.arange(hh, dtype=np.float64))
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    lam = (d0 - hmap.astype(np.float64)) / (rays @ np.asarray(n, float))
    return rays * lam[..., None]


def fit_plane(P):
    """МНК-плоскость n·X = d по точкам (N, 3); n - от камеры (n_z > 0)."""
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1] * (1 if Vt[-1][2] > 0 else -1)
    return n, float(n @ c)


def local_table(X, hmap, valid, near=3.0, tol=0.3, min_share=0.05):
    """Стол в кадре: самый населённый уровень высот в пределах +-near от опорной плоскости, плоскость по его
    точкам (итеративно, отбрасывая дальше tol). Возвращает (n, d, доля пикселей) или None."""
    hs = hmap[valid]
    if hs.size < 2000:
        return None
    bins = np.arange(-near, near + 0.05, 0.05)
    hist, _ = np.histogram(hs, bins)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    h_peak = bins[int(np.argmax(hist))] + 0.025
    pts = X[valid]
    inl = np.abs(hs - h_peak) < tol
    if inl.sum() < max(2000, min_share * hs.size):
        return None
    for _ in range(4):
        n, d = fit_plane(pts[inl])
        res = d - pts @ n
        inl = np.abs(res) < tol
        if inl.sum() < max(2000, min_share * hs.size):
            return None
    return n, d, float(inl.sum() / valid.size)


# ---------- точка: уточнение по полному разрешению
def _sample(R, q):
    """Билинейная выборка карты R в точках q (M, 2); вне карты - NaN."""
    n = len(q)
    pad = (-n) % 1024  # remap: не больше 32767 точек в строке
    mx = np.pad(q[:, 0].astype(np.float32), (0, pad), constant_values=-1e4).reshape(-1, 1024)
    my = np.pad(q[:, 1].astype(np.float32), (0, pad), constant_values=-1e4).reshape(-1, 1024)
    s = cv2.remap(R.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-9)
    s = s.ravel()[:n].astype(np.float64)
    s[s < -5] = np.nan
    return s


def _peak(hs, s):
    i = int(np.nanargmax(s))
    if 0 < i < len(s) - 1 and np.all(np.isfinite(s[i - 1:i + 2])):
        den = s[i - 1] - 2 * s[i] + s[i + 1]
        if den < 0:
            return hs[i] + 0.5 * (s[i - 1] - s[i + 1]) / den * (hs[1] - hs[0]), s[i], i
    return hs[i], s[i], i


def refine_point(ref, others, Os, K, n, d0, x, h_lo, h_hi, patch=31, step=0.01, min_std=3.0, ref_valid=None):
    """Высота точки x (пиксель опорного кадра) над плоскостью n·X = d0: шаблон вокруг x ищется в каждом кадре
    вдоль её эпиполярной линии (только высоты h_lo..h_hi), корреляции всех кадров складываются по высоте.
    Возвращает dict: h, score, X (3D), h_views (по каждому кадру отдельно), sigma; либо {'error': ...}."""
    r = patch // 2
    xi, yi = int(round(x[0])), int(round(x[1]))
    if not (r <= xi < ref.shape[1] - r and r <= yi < ref.shape[0] - r):
        return {"error": "слишком близко к краю кадра"}
    tpl = ref[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float32)
    if ref_valid is not None and not ref_valid[yi - r:yi + r + 1, xi - r:xi + r + 1].all():
        return {"error": "слишком близко к краю кадра"}
    if tpl.std() < min_std:
        return {"error": "в точке нет текстуры (однотонно) - высоту не по чему мерить"}
    ray = np.linalg.inv(K) @ np.array([xi, yi, 1.0])
    hs = np.arange(h_lo, h_hi + step / 2, step)
    X = ((d0 - hs) / (np.asarray(n) @ ray))[:, None] * ray[None, :]
    total = np.zeros(len(hs))
    count = np.zeros(len(hs))
    h_views, s_views = [], []
    for img, O in zip(others, Os):
        P = (K @ (X - np.asarray(O)).T).T
        p = P[:, :2] / P[:, 2:3]
        x0, y0 = np.floor(p.min(axis=0)).astype(int) - r - 2
        x1, y1 = np.ceil(p.max(axis=0)).astype(int) + r + 3
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        if x1 - x0 < patch + 2 or y1 - y0 < patch + 2:
            continue
        roi = img[y0:y1, x0:x1].astype(np.float32)
        R = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        s = _sample(R, p - np.array([x0 + r, y0 + r]))
        ok = np.isfinite(s)
        if ok.sum() < 3:
            continue
        total[ok] += s[ok]
        count[ok] += 1
        hv, sv, _ = _peak(hs, np.where(ok, s, np.nan))
        h_views.append(float(hv))
        s_views.append(float(sv))
    if len(h_views) < 2:
        return {"error": "точку видно меньше чем в 2 кадрах скана"}
    full = count == count.max()
    S = np.where(full, total / np.maximum(count, 1), np.nan)
    h, score, i = _peak(hs, S)
    edge = i in (0, len(hs) - 1)
    Xh = ((d0 - h) / (np.asarray(n) @ ray)) * ray
    hv = np.array(h_views)
    good = np.array(s_views) > 0.5
    sigma = float(np.std(hv[good], ddof=1) / np.sqrt(good.sum())) if good.sum() >= 3 else float("nan")
    return {"h": float(h), "score": float(score), "X": Xh, "h_views": h_views, "s_views": s_views,
            "sigma": sigma, "at_range_edge": bool(edge), "views": int(count.max())}


# ---------- метки: триангуляция углов
def triangulate(pts, Os, K):
    """Точка по её пикселям (без дисторсии) в кадрах с центрами камеры Os (оси опорной камеры): лучи ->
    середина, затем Гаусс-Ньютон по ошибке репроекции. Возвращает (X, rms ошибки репроекции, px)."""
    K = np.asarray(K, float)
    Kinv = np.linalg.inv(K)
    A, b = np.zeros((3, 3)), np.zeros(3)
    for x, O in zip(pts, Os):
        r = Kinv @ np.array([x[0], x[1], 1.0])
        r /= np.linalg.norm(r)
        P = np.eye(3) - np.outer(r, r)
        A += P
        b += P @ np.asarray(O, float)
    X = np.linalg.solve(A, b)
    fx, fy = K[0, 0], K[1, 1]
    for _ in range(6):
        J, e = [], []
        for x, O in zip(pts, Os):
            Y = X - np.asarray(O, float)
            u = (K @ Y)[:2] / Y[2]
            e.extend(u - np.asarray(x, float))
            J.append([fx / Y[2], 0, -fx * Y[0] / Y[2] ** 2])
            J.append([0, fy / Y[2], -fy * Y[1] / Y[2] ** 2])
        dX = np.linalg.lstsq(np.array(J), -np.array(e), rcond=None)[0]
        X = X + dX
        if np.linalg.norm(dX) < 1e-7:
            break
    e = []
    for x, O in zip(pts, Os):
        Y = X - np.asarray(O, float)
        e.append(np.linalg.norm((K @ Y)[:2] / Y[2] - np.asarray(x, float)))
    return X, float(np.sqrt(np.mean(np.square(e))))


# ---------- объекты на столе
def find_objects(h_table, valid, min_height=0.5, min_area_px=200, close_px=7, gap_px=4):
    """Связные области выше min_height над столом (части ближе gap_px - один объект: верх и видимая сбоку стенка
    разделены полосой без данных). Высоты - по достоверным пикселям после медианы 5x5 (одиночные выбросы)."""
    filled = np.where(valid, h_table, -100).astype(np.float32)
    hmed = cv2.medianBlur(filled, 5)
    good = valid & (hmed > -50)
    mask = ((hmed > min_height) & good).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1)))
    num, lab = cv2.connectedComponents(grouped, connectivity=8)
    objs = []
    for i in range(1, num):
        region = (lab == i) & (mask > 0)
        area = int(region.sum())
        if area < min_area_px:
            continue
        hs = hmed[region & good & (hmed > min_height)]
        if hs.size < 20:
            continue
        cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top = float(np.percentile(hs, 98))
        ys, xs = np.nonzero(region)
        objs.append({"area_px": area, "centroid_px": [float(xs.mean()), float(ys.mean())],
                     "bbox_px": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1],
                     "contour": max(cnts, key=cv2.contourArea),
                     "height_top": top, "height_median": float(np.median(hs)),
                     "height_max": float(hs.max()), "valid_px": int(hs.size),
                     "top_px": np.argwhere(region & good & (hmed >= top - 0.2))[:, ::-1]})
    return sorted(objs, key=lambda o: -o["height_top"])


def low_texture(img, win=9, min_std=2.0, valid=None, sat=250.0, raw=None):
    """Пиксели без текстуры (СКО яркости в окне win меньше min_std) - там перебор высот не по чему сравнивать.
    Пересвеченные (>= sat) не входят: однотонность там от блика, а не от поверхности.
    img - тот же кадр, по которому шло сопоставление (после highpass); пересвет тогда ищется по raw (исходному)."""
    img = img.astype(np.float32)
    m = cv2.boxFilter(img, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    v = cv2.boxFilter(img * img, -1, (win, win), borderType=cv2.BORDER_REFLECT) - m * m
    low = v < min_std ** 2
    raw = img if raw is None else np.asarray(raw, np.float32)
    bright = cv2.dilate((raw >= sat).astype(np.uint8), np.ones((win, win), np.uint8)) > 0
    low &= ~bright
    if valid is not None:
        low &= valid
    return low


def plane_points(K, n, d, shape):
    """Точки плоскости n·X = d на лучах всех пикселей кадра (H, W, 3)."""
    hh, ww = shape
    u, v = np.meshgrid(np.arange(ww, dtype=np.float64), np.arange(hh, dtype=np.float64))
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    return rays * (d / (rays @ np.asarray(n, float)))[..., None]


def ref_glare(views, valid, Os, K, X, thr=0.6, min_area=0, t_abs=12.0, t_rel=0.5, sigma=4.0):
    """Блик в опорном кадре: место, где опорный кадр заметно светлее той же точки в остальных кадрах.
    Блик стоит на месте в снимке (свет едет с камерой), а предметы сдвигаются: одна и та же точка в одних
    кадрах под бликом, в других - нет. "Чистый" вид точки - второй снизу по яркости среди всех кадров (блик
    только добавляет свет; второй, а не самый тёмный - запас на шум).
    views - кадры [опорный, остальные...] (исходные, не highpass), valid - где в кадрах есть данные (одна маска
    на все), Os - центры камер, X - (H, W, 3) точка каждого пикселя опорного (по найденной высоте, где она есть,
    иначе по плоскости стола). Превышение сглаживается (sigma) и сравнивается с t_abs + t_rel * яркость точки:
    разница освещённости между кадрами (виньетирование) так блика не даёт. Пятна меньше min_area пикселей
    отброшены: это блики на мелких деталях, а не засветка поверхности."""
    hh, ww = views[0].shape
    m1 = np.full((hh, ww), np.inf, np.float32)
    m2 = m1.copy()
    cnt = np.zeros((hh, ww), np.uint8)
    vm = np.asarray(valid, np.uint8)
    for img, O in zip(views, Os):
        P = (X - np.asarray(O, float)) @ np.asarray(K, float).T
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = (P[..., 0] / P[..., 2]).astype(np.float32)
            my = (P[..., 1] / P[..., 2]).astype(np.float32)
        w = cv2.remap(np.asarray(img, np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=0)
        ok = cv2.remap(vm, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        w = np.where(ok, w, np.inf).astype(np.float32)
        np.minimum(m2, np.maximum(m1, w), out=m2)  # два наименьших значения по кадрам
        np.minimum(m1, w, out=m1)
        cnt += ok
    clean = np.where(cnt >= 3, m2, np.nan)
    ref = np.asarray(views[0], np.float32)
    e = np.where(np.isfinite(clean), ref - clean, 0).astype(np.float32)
    e = cv2.GaussianBlur(cv2.medianBlur(e, 5), (0, 0), sigma)
    level = cv2.GaussianBlur(np.nan_to_num(clean, nan=0).astype(np.float32), (0, 0), sigma)
    glare = (e / (t_abs + t_rel * level) > thr).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(glare, connectivity=8)
    keep = np.zeros(num, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def _ransac_plane(P, tol, iters, rng):
    """Плоскость n·X = d по точкам P (N, 3), устойчиво к выбросам: (n, d, маска inlier) или None."""
    best = None
    sub = P if len(P) <= 4000 else P[rng.choice(len(P), 4000, replace=False)]
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        k = int((np.abs(sub @ n - n @ a) < tol).sum())
        if best is None or k > best[0]:
            best = (k, n, float(n @ a))
    if best is None:
        return None
    _, n, d = best
    inl = np.abs(P @ n - d) < tol
    for _ in range(3):  # уточнение МНК по inlier
        if inl.sum() < 3:
            return None
        n, d = fit_plane(P[inl])
        inl = np.abs(P @ n - d) < tol
    return n, d, inl


def fill_planar(X, valid, low, K, n_ref, min_area=3000, tol=0.25, min_inliers=300, min_share=0.3,
                max_tilt=10.0, iters=300, seed=0, margin_px=0):
    """Достройка однотонных областей по плоскости: однотонная связная область (без текстуры, без высоты)
    вместе со всем, что внутри неё (площадки, отверстия, компоненты), - кандидат на плоскую поверхность.
    По измеренным точкам внутри неё RANSAC ищет плоскость (компоненты над ней - выбросы); однотонные
    пиксели заполняются этой плоскостью, но только внутри выпуклой оболочки точек, легших на неё, плюс
    margin_px (край платы под бликом: крайние измеренные точки чуть не доходят до края). Плоскость не дальше
    max_tilt градусов от n_ref (стола), иначе это не "лежащая" поверхность.
    Возвращает (номер области у достроенных пикселей (0 - нет), X достроенного (NaN вне), список областей)."""
    hh, ww = valid.shape
    rng = np.random.default_rng(seed)
    filled = np.zeros((hh, ww), np.int16)
    Xf = np.full((hh, ww, 3), np.nan)
    Kinv = np.linalg.inv(K)
    cand = (low & ~valid).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    regions = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = np.zeros((hh, ww), np.uint8)
        cv2.drawContours(area, cnts, -1, 1, cv2.FILLED)  # область вместе с "дырками" (площадки, компоненты)
        pts_m = valid & (area > 0)
        if pts_m.sum() < min_inliers:
            continue
        ys, xs = np.nonzero(pts_m)
        P = X[ys, xs]
        res = _ransac_plane(P, tol, iters, rng)
        if res is None:
            continue
        n, d, inl = res
        if n @ n_ref < 0:
            n, d = -n, -d
        tilt = float(np.degrees(np.arccos(np.clip(abs(n @ n_ref), -1, 1))))
        if inl.sum() < min_inliers or inl.mean() < min_share or tilt > max_tilt:
            continue
        hull = cv2.convexHull(np.column_stack([xs[inl], ys[inl]]).astype(np.int32))
        hm = np.zeros((hh, ww), np.uint8)
        cv2.fillConvexPoly(hm, hull, 1)
        if margin_px > 0:
            hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)))
        f = (comp > 0) & (hm > 0)
        if not f.any():
            continue
        fy, fx = np.nonzero(f)
        rays = np.column_stack([fx, fy, np.ones(len(fx))]) @ Kinv.T
        Xf[fy, fx] = rays * (d / (rays @ n))[:, None]
        filled[f] = len(regions) + 1
        rms = float(np.sqrt(np.mean((P[inl] @ n - d) ** 2)))
        regions.append({"n": n, "d": float(d), "tilt_deg": tilt, "inliers": int(inl.sum()),
                        "inlier_share": float(inl.mean()), "rms_mm": rms, "filled_px": int(f.sum())})
    return filled, Xf, regions


def colorize(h, valid, base, lo, hi, filled=None):
    """Карта высот поверх серого кадра: TURBO от lo до hi, недостоверное - тёмно-серое."""
    g = base if base.ndim == 2 else cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    t = np.clip((np.nan_to_num(h, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    col = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out = (gray * 0.45).astype(np.uint8)
    if filled is not None:  # достроенное по плоскости - тот же цвет, но бледнее и в косую штриховку
        soft = cv2.addWeighted(col, 0.35, gray, 0.2, 0)
        yy, xx = np.indices(valid.shape)
        stripe = ((xx + yy) // 3) % 4 == 0
        soft[stripe] = (soft[stripe] * 0.5).astype(np.uint8)
        out[filled] = soft[filled]
    blend = cv2.addWeighted(col, 0.65, gray, 0.35, 0)
    out[valid] = blend[valid]
    return out
