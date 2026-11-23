"""
Высота объектов на столе и высота в точке объекта - той же камерой, без щупа и без оси Z.

Как: станок сдвигает камеру относительно стола чистым сдвигом на известные ходы, поэтому кадры
с нескольких точек - стереосъёмка с известной базой (ход станка в осях камеры - из калибровки со
станком). Чем выше точка, тем сильнее она смещается в кадре при том же ходе (параллакс).
Высота - от стола: от его поверхности вокруг объекта в тех же кадрах (ошибки позиционирования
станка при этом почти сокращаются), а если стола в кадре мало - от плоскости меток (обход t).

Ось Z станка не используется и не двигается. Фокус не трогать: калибровка верна только при нём.
Нужна текстура: однотонную гладкую поверхность (белый пластик, полированный металл) так не
измерить - на карте она пустая, клик по ней скажет "нет текстуры". Если вокруг однотонного места
есть измеренные точки одной плоскости (чёрная маска платы между площадками), оно достраивается по
ней - на карте бледным в штриховку, это не замер (--no-fill - не достраивать). Блики: свет едет
вместе с камерой, засветка в разных кадрах лежит по-разному - сопоставление идёт по кадрам без
плавной составляющей (--highpass), а место под бликом в опорном кадре, где не измерилось,
достраивается так же, как однотонное.

Порядок:
  1. Камера на метке: c - подключение, k - отладка осей и наведение на метку (привязка к раскладке
     меток; по ходу станка и калибровке - сверка направлений и высота камеры над меткой).
  2. t - один раз (и каждый раз, когда трогали стол/метки): обход меток с замером высоты у каждой ->
     плоскость стола и её наклон к ходу станка, calib/table_plane.json. Нужна раскладка меток
     (survey_markers.py r или здесь r).
  3. Камера над объектом (wasd, шаг 1/2/3): объект в середине кадра, вокруг виден стол. p - скан:
     По умолчанию 5x5 = 25 ракурсов, запрошенный шаг 16 мм (размах до 64x64 мм).
     Шаг автоматически уменьшается по границам и перекрытию на высоте --max-height.
     Подход к каждой точке с +, возврат в исходную. Фактический план виден до подтверждения.
     Окно "высоты": карта высот, объекты и их высота над столом. Кадры скана - в scans/.
  4. Клик по карте - высота в точке (уточнение на полном разрешении по всем кадрам скана).
     g - навести камеру на выбранную точку (с учётом её высоты) и проверить по кадру.
  q - выход. ESC - стоп хода.

Без станка - разбор сохранённого скана (карта, клики):
    python z/measure_height.py --from-scan scans/scan_<время>

Запуск:
    python z/measure_height.py --camera 0 --width 1920 --height 1080 --port COM4
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))
sys.path.append(str(HERE))
from common.camera import capture_mode, grab_fresh, open_camera, read_frame, verify_focus  # noqa: E402
from common.display import add_zoom_trackbar, draw_texts, get_zoom, show_scaled, text_px  # noqa: E402
from common.grbl import GrblError  # noqa: E402
from common.markers import detect  # noqa: E402
from pose_refinement import stereo_positions
from stereo import (Rectifier, SweepAborted, fill_planar, find_objects, low_texture, fit_plane, half, highpass,  # noqa: E402
                    load_motion, local_table, motion_normal, plane_homography, plane_points, plane_sweep,
                    points_from_heights, ref_glare, refine_point, under_camera, colorize)
from survey_markers import (JOG_KEYS, JOG_STEPS, LAYOUT_SETTINGS, Abort, Survey, WindowUI, add_survey_args,  # noqa: E402
                            angle_deg, find_calibration, fmt_move)

PLANE_FILE = ROOT / "calib" / "table_plane.json"
SCANS_DIR = ROOT / "scans"
GLARE_MIN_AREA = 100.0  # мм²: пятна блика меньше - блики на мелких деталях, а не засветка поверхности
SATURATED = 250.0       # яркость пересвета (кадр - среднее 8 снимков, потолок ~254)


# ---------- скан: обработка (общая для живого скана и разбора сохранённого)
class ScanResult:
    """Кадры скана без дисторсии, карта высот над столом, объекты; высота в точке по клику."""

    def __init__(self, frames, positions, ref_index, K, D, model, V, guess, global_plane, args, rect=None,
                 tick=None, log=print):
        self.args, self.K, self.V = args, np.asarray(K, float), np.asarray(V, float)
        self.positions = [np.asarray(p, float) for p in positions]
        order = [ref_index] + [i for i in range(len(frames)) if i != ref_index]
        self.c_ref = self.positions[ref_index]
        self.cs = [self.positions[i] for i in order]
        self.Os = [self.V @ (c - self.c_ref) for c in self.cs]
        size = (frames[0].shape[1], frames[0].shape[0])
        self.rect = rect or Rectifier(self.K, D, size, model)
        self.full = [self.rect(frames[i].astype(np.float32)) for i in order]
        self.valid_full = self.rect.valid
        halfs = [half(f, self.K) for f in self.full]
        self.Ks = halfs[0][1]
        vh = cv2.resize(self.valid_full.astype(np.uint8), (halfs[0][0].shape[1], halfs[0][0].shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
        self.guess = guess                  # (n, d, что это) - откуда перебирать высоты
        self.global_plane = global_plane    # (n, d) плоскости меток в осях опорной камеры или None
        n0, d0, _ = guess
        heights = sweep_levels(args, d0)
        t0 = time.time()
        # Подавляем плавные блики только для сопоставления. Исходные кадры нужны для отображения.
        self.hp = float(getattr(args, "highpass", 4.0) or 0)
        raw = [h[0] for h in halfs]
        self._match_full = None
        self.click_masks = None
        indices = list(range(1, len(halfs)))
        if args.map_views:
            # При ограничении числа кадров берём разные стороны и базы, а не начало змейки.
            selected, anchors = [], [np.zeros(3)]
            while indices and len(selected) < args.map_views:
                best = max(indices, key=lambda j: min(np.linalg.norm(self.Os[j] - a) for a in anchors))
                selected.append(best)
                anchors.append(self.Os[best])
                indices.remove(best)
            indices = selected
        if not 2 <= args.min_map_views <= len(indices):
            raise ValueError("недостаточно ракурсов для --min-map-views; увеличьте --map-views или уменьшите минимум")
        match_ref = highpass(raw[0], self.hp) if self.hp > 0 else raw[0]
        others = [highpass(raw[i], self.hp) if self.hp > 0 else raw[i] for i in indices]
        photo_valid = [vh & (im < SATURATED) for im in raw]
        self.hmap, self.score, self.valid = plane_sweep(
            match_ref, others, [self.Os[i] for i in indices], self.Ks, n0, d0, heights, ref_valid=photo_valid[0],
            others_valid=[photo_valid[i] for i in indices], win=args.win, min_std=args.min_texture, min_score=args.min_score,
            min_views=args.min_map_views, tick=tick,
            peak_margin=getattr(args,'peak_margin',.045), texture_ratio=getattr(args,'texture_ratio',.045))
        log(f"карта высот: {len(heights)} уровней x {len(others)} кадров за {time.time() - t0:.0f} с, "
            f"определилось {100 * self.valid.mean():.0f}% кадра")
        self.X = points_from_heights(self.hmap, self.Ks, n0, d0)
        self.notes = []
        table_mask = self.valid
        if getattr(args, 'table_roi', None):
            table_mask = np.zeros_like(self.valid)
            hh, ww = self.valid.shape
            for x, y, w, h in args.table_roi:
                if min(x, y) < 0 or min(w, h) <= 0 or x+w > ww or y+h > hh:
                    raise ValueError("--table-roi вне карты половинного разрешения")
                table_mask[y:y+h, x:x+w] = self.valid[y:y+h, x:x+w]
        loc = local_table(self.X, self.hmap, table_mask, near=args.table_near)
        if getattr(args, 'table_roi', None) and loc is None:
            self.notes.append("Недостаточно опоры в заданных ROI стола; используется исходная плоскость")
        self.global_offset = None
        if loc is not None:
            self.n_t, self.d_t, share = loc
            self.ref_kind = f"стол в кадре ({100 * share:.0f}% кадра)"
            if global_plane is not None:
                gn, gd = global_plane
                ht_g = gd - self.X @ gn
                on_table = self.valid & (np.abs((self.d_t - self.X @ self.n_t)) < 0.3)
                self.global_offset = float(np.median(ht_g[on_table]))
                self.notes.append(f"стол здесь на {self.global_offset:+.3f} мм от плоскости меток, наклон к ней "
                                  f"{angle_deg(self.n_t, gn):.3f}°")
        elif global_plane is not None:
            self.n_t, self.d_t = global_plane
            self.ref_kind = "плоскость меток (стола в кадре мало)"
        else:
            self.n_t, self.d_t = n0, d0
            self.ref_kind = "уровень метки при k (стола в кадре мало, плоскость меток не измерена - t)"
            self.notes.append("высоты приблизительные: нет ни стола в кадре, ни плоскости меток")
        self.ht = (self.d_t - self.X @ self.n_t).astype(np.float32)
        self.ht[~self.valid] = np.nan
        px_mm = self.Ks[0, 0] / self.d_t
        # Однотонные места (чёрная маска платы, гладкий пластик) перебор высот не меряет. Если вокруг них есть
        # измеренные точки одной плоскости (площадки, отверстия, маркировка) - достраиваем по ней. Это НЕ замер:
        # отдельная маска self.filled, объекты ищутся только по измеренному.
        # Под бликом в опорном кадре поверхности не видно (там ложная текстура засветки, а не однотонность) -
        # где там не измерилось, тоже достраиваем. Кроме пересвета: под ним может быть что угодно.
        self.filled = np.zeros_like(self.valid)
        self.glare = np.zeros_like(self.valid)
        self.planes = []
        if getattr(args, "fill_planes", True):
            low = low_texture(match_ref, args.win, args.min_texture, vh, sat=SATURATED, raw=raw[0])
            X_al = np.where(self.valid[..., None], self.X, plane_points(self.Ks, self.n_t, self.d_t, self.valid.shape))
            self.glare = ref_glare(raw, vh, self.Os, self.Ks, X_al, min_area=int(GLARE_MIN_AREA * px_mm ** 2))
            low |= self.glare & (raw[0] < SATURATED) & vh
            lab, Xf, self.planes = fill_planar(
                self.X, self.valid, low, self.Ks, self.n_t,
                min_area=max(500, int(getattr(args, "fill_min_area", 25.0) * px_mm ** 2)),
                tol=getattr(args, "fill_tol", 0.25), margin_px=int(round(getattr(args, "fill_margin", 3.0) * px_mm)))
            self.filled = lab > 0
            if self.glare.any():
                log(f"блик в опорном кадре: {100 * self.glare.mean():.0f}% кадра (где не измерилось - достраивается)")
            if self.filled.any():
                self.X = np.where(self.filled[..., None], Xf, self.X)
                self.ht[self.filled] = (self.d_t - Xf[self.filled] @ self.n_t).astype(np.float32)
                for k, pl in enumerate(self.planes, 1):
                    pl["height_mm"] = float(np.median(self.ht[lab == k]))  # над столом
                    pl["tilt_to_table_deg"] = pl.pop("tilt_deg")
                log(f"достроено по плоскости: {100 * self.filled.mean():.0f}% кадра ({len(self.planes)} обл.)")
        self.objects = find_objects(self.ht, self.valid, args.min_height, max(50, int(args.min_area * px_mm ** 2)))
        for o in self.objects:
            pts = o["top_px"]
            Xo = np.median(self.X[pts[:, 1], pts[:, 0]], axis=0) if len(pts) else self.X[int(o["centroid_px"][1]),
                                                                                           int(o["centroid_px"][0])]
            o["machine_xy"] = (self.c_ref + under_camera(self.V, Xo)).tolist()

    def table_height_at(self, x_full):
        """Высота по карте (половинное разрешение) в пикселе полного кадра или None."""
        u, v = int(x_full[0] // 2), int(x_full[1] // 2)
        if 0 <= v < self.ht.shape[0] and 0 <= u < self.ht.shape[1] and self.valid[v, u]:
            return float(self.ht[v, u])
        return None

    def match_full(self):
        """Кадры полного разрешения для замера по клику - с тем же фильтром, что и карта (σ вдвое больше:
        пиксели вдвое мельче). Считаются при первом клике."""
        if self._match_full is None:
            self._match_full = [highpass(f, 2 * self.hp) for f in self.full] if self.hp > 0 else self.full
        return self._match_full

    def measure(self, x_full):
        a = self.args
        h_map = self.table_height_at(x_full)
        mf = self.match_full()
        if self.click_masks is None:
            self.click_masks = [self.valid_full & (im < SATURATED) for im in self.full]
        limits = -a.below,min(a.max_height,self.d_t-1.)
        def measure_in(lo,hi):
            return refine_point(mf[0],mf[1:],self.Os[1:],self.K,self.n_t,self.d_t,x_full,lo,hi,
                                patch=a.patch,min_std=a.min_texture,ref_valid=self.click_masks[0],robust=True,
                                others_valid=self.click_masks[1:],
                                texture_ratio=getattr(a,'texture_ratio',.045))
        def reliable(res):
            return ('error' not in res and res['score'] > .6 and sum(res['inlier_views']) >= 3
                    and res['sigma'] < a.max_sigma and res['peak_margin'] >= .045 and not res['at_range_edge'])
        window = limits if h_map is None else (max(limits[0],h_map-a.click_range),min(limits[1],h_map+a.click_range))
        if window[1]-window[0] < .03:
            window = limits
        res = measure_in(*window)
        # A weak dense-map seed should not confine a click to the wrong height band.
        if window != limits and not reliable(res):
            alternative = measure_in(*limits)
            if reliable(alternative):
                res = alternative
        if "error" in res:
            return res
        res["h_map"] = h_map
        res["machine_xy"] = (self.c_ref + under_camera(self.V, res["X"])).tolist()
        good = np.array(res["inlier_views"],dtype=bool)
        hv = np.array(res["h_views"])[good]
        res["spread"] = float(hv.max() - hv.min()) if hv.size >= 2 else float("nan")
        res["reliable"] = bool(reliable(res))
        res["x"] = [float(x_full[0]), float(x_full[1])]
        return res

    def render(self, sel=None):
        a = self.args
        hi = max(a.min_height + 0.5, max([o["height_top"] for o in self.objects], default=1.0) * 1.05)
        small = colorize(self.ht, self.valid, self.full[0][::2, ::2][:self.ht.shape[0], :self.ht.shape[1]], -0.3, hi,
                         filled=self.filled)
        img = cv2.resize(small, (self.full[0].shape[1], self.full[0].shape[0]), interpolation=cv2.INTER_NEAREST)
        size = text_px(img)
        texts = []
        for i, o in enumerate(self.objects, 1):
            cv2.drawContours(img, [o["contour"] * 2], -1, (255, 255, 255), 2, cv2.LINE_AA)
            x, y = (int(v * 2) for v in o["centroid_px"])
            texts.append((f"#{i}: {o['height_top']:.2f} мм", (x - 2 * size, y - size // 2), size, (255, 255, 255),
                          (40, 40, 40)))
        if sel is not None and "x" in sel:
            x, y = (int(round(v)) for v in sel["x"])
            cv2.drawMarker(img, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 3 * size, 2, cv2.LINE_AA)
            r = a.patch // 2
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r), (255, 0, 255), 1)
            if "h" in sel:
                txt = f"{sel['h']:.3f} мм" + ("" if sel["reliable"] else " (?)")
                texts.append((txt, (x + size, y - size), size, (255, 255, 255), (120, 0, 120)))
        head = [f"высоты над столом: от -0.3 до {hi:.1f} мм (синий -> красный), серое - не определилось",
                f"отсчёт: {self.ref_kind}"] + self.notes
        if self.filled.any():
            head.append("бледное в штриховку - достроено по плоскости (однотонно или под бликом, не измерено)")
        for i, line in enumerate(head):
            texts.append((line, (8, 6 + i * (size + 8)), size, (255, 255, 255), (0, 0, 0)))
        return draw_texts(img, texts)

    def report(self, log=print):
        log(f"отсчёт высоты: {self.ref_kind}")
        for n in self.notes:
            log(n)
        if not self.objects:
            log(f"объектов выше {self.args.min_height:g} мм не найдено")
        for i, o in enumerate(self.objects, 1):
            log(f"объект #{i}: верх {o['height_top']:.3f} мм (медиана {o['height_median']:.3f}, макс {o['height_max']:.3f}), "
                f"центр верха под камерой при X{o['machine_xy'][0]:.3f} Y{o['machine_xy'][1]:.3f}")

    def save_analysis(self, folder):
        (folder / "result.json").write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False),
                                            encoding="utf-8")
        np.savez_compressed(folder / "heights.npz", height_mm=self.ht, measured=self.valid,
                            filled=self.filled, glare=self.glare, score=self.score,
                            K=self.Ks, table_normal=self.n_t, table_distance=self.d_t)
        if not cv2.imwrite(str(folder / "heights.png"), self.render()):
            raise OSError("Не удалось сохранить heights.png")

    def to_json(self):
        return {"ref_kind": self.ref_kind, "table_plane_ref_cam": {"n": np.asarray(self.n_t).tolist(), "d": float(self.d_t)},
                "global_offset_mm": self.global_offset, "notes": self.notes,
                "objects": [{k: v for k, v in o.items() if k not in ("contour", "top_px")} for o in self.objects],
                "filled_planes": [{k: (np.asarray(v).tolist() if k == "n" else v) for k, v in p.items()}
                                  for p in self.planes],
                "measured_fraction": float(self.valid.mean()), "filled_fraction": float(self.filled.mean()),
                "unknown_fraction": float((~(self.valid | self.filled)).mean()),
                "highpass_px": self.hp, "glare_share": float(self.glare.mean())}


def save_scan(folder, frames, positions, ref_index, meta):
    folder.mkdir(parents=True, exist_ok=False)
    for i, f in enumerate(frames):  # 16 бит: усреднение кадров не теряется
        path = folder / f"frame_{i:02d}.png"
        if not cv2.imwrite(str(path), np.clip(np.round(f * 256), 0, 65535).astype(np.uint16)):
            raise OSError(f"Не удалось записать кадр {path}")
    meta = dict(meta, positions=[np.asarray(p).tolist() for p in positions], ref_index=ref_index)
    (folder / "scan.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_scan(folder):
    folder = Path(folder)
    meta = json.loads((folder / "scan.json").read_text(encoding="utf-8"))
    positions = np.asarray(meta["positions"], float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all()
            or not isinstance(meta["ref_index"], int) or not 0 <= meta["ref_index"] < len(positions)):
        raise ValueError(f"Неверные позиции или индекс опорного кадра в {folder / 'scan.json'}")
    frames = []
    for i in range(len(positions)):
        path = folder / f"frame_{i:02d}.png"
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None or frame.ndim != 2 or frame.dtype != np.uint16:
            raise ValueError(f"Ожидался читаемый 16-битный серый кадр {path}")
        if frames and frame.shape != frames[0].shape:
            raise ValueError(f"Размер кадра {path} не совпадает с первым кадром")
        frames.append(frame.astype(np.float32) / 256)
    return frames, meta


def sweep_levels(args, distance):
    step, below, above = args.sweep_step, args.below, args.max_height
    if (not np.isfinite([step, below, above, distance]).all() or step <= 0
            or below < 0 or above < 0 or above + step / 2 >= distance
            or (below + above) / step < 1.5):
        raise ValueError("Проверьте --sweep-step, --below и --max-height: нужны три уровня перед камерой")
    if not (0 <= getattr(args,'peak_margin',.045) <= 2 and 0 <= getattr(args,'texture_ratio',.045) <= .5):
        raise ValueError("Пороги --peak-margin и --texture-ratio должны лежать в [0,2] и [0,0.5]")
    return np.arange(-below, above + step / 2, step)


class MapWindow:
    """Окно карты высот: клик -> координаты пикселя кадра (с учётом масштаба окна)."""

    def __init__(self, name="высоты"):
        self.name, self.click = name, None
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        add_zoom_trackbar(name, default=50)
        cv2.setMouseCallback(name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            z = get_zoom(self.name) / 100
            self.click = (x / z, y / z)

    def show(self, img):
        show_scaled(self.name, img, get_zoom(self.name))

    def take_click(self):
        c, self.click = self.click, None
        return c


# ---------- станок + камера
class HeightTool(Survey):
    def __init__(self, cap, args, ui, calib, calib_path, grbl_factory=None, frame_path=None, plane_path=None,
                 scans_dir=None, map_window=None):
        super().__init__(cap, args, ui, grbl_factory=grbl_factory, frame_path=frame_path or args.frame_file, calib=calib)
        self.profile_id = hashlib.sha256(json.dumps(calib, sort_keys=True).encode()).hexdigest()
        if self.K is None:
            raise SystemExit("для высоты нужна калибровка камеры (calibrate_distortion.py со станком)")
        motion = load_motion(calib, calib_path, log=self.say)
        if motion is None:
            raise SystemExit("в калибровке нет хода станка в осях камеры - нужна калибровка со станком (--grbl)")
        self.calib_name = Path(calib_path).name
        self.D_model = calib["model"]
        self.V_calib = np.array(motion["V_cam"], float)
        self.V = self.V_calib.copy()
        self.n_m = motion_normal(self.V)
        self.rect = Rectifier(self.K, self.D, self.size, self.calib_model)
        self.plane_path = Path(plane_path) if plane_path else args.plane_file
        self.scans_dir = Path(scans_dir) if scans_dir else SCANS_DIR
        self.map = map_window
        self.table_plane = None
        self.depth_pass = False
        self.marker_3d = {}
        self.z_marker = None     # расстояние камера -> плоскость метки при k (по нормали к ходу), мм
        self.scan = None
        self.sel = None
        self.load_table_plane()
        self.say(f"ход станка в осях камеры: из {self.calib_name}; ось камеры к плоскости хода "
                 f"{motion.get('camera_tilt_deg', float('nan')):.2f}°")

    # ---------- оси: сверка с калибровкой
    def calibrate_axes(self):
        super().calibrate_axes()
        self.check_motion()
        self.center_and_record()

    def check_motion(self):
        """Ход станка в кадре (k) против хода из калибровки: та же ли сторона по каждой оси, высота камеры."""
        f = np.diag([self.K[0, 0], self.K[1, 1]])
        V = self.V_calib.copy()
        flips = []
        for i, name in enumerate("XY"):
            ang = angle_deg(self.A[:, i], -f @ V[:2, i])
            if ang > 160:
                V[:, i] *= -1
                flips.append(name)
            elif ang > 20:
                raise Abort(f"ось {name}: в кадре ход повёрнут на {ang:.0f}° относительно калибровки - камеру "
                            f"сдвигали/поворачивали на кронштейне? Нужна новая калибровка со станком")
        if flips:
            self.say(f"ось {', '.join(flips)} едет в обратную сторону, чем при калибровке ($3 меняли после неё) - учтено")
        self.V, self.n_m = V, motion_normal(V)
        P = -f @ V[:2]
        Z = float(np.sum(P * P) / np.sum(P * self.A))
        self.z_marker = Z * self.n_m[2]
        self.say(f"камера над меткой ~{self.z_marker:.1f} мм (по ходу станка; точно - по скану)")

    # ---------- плоскость стола по меткам
    def load_table_plane(self):
        if not self.plane_path.exists():
            return
        try:
            self.table_plane = json.loads(self.plane_path.read_text(encoding="utf-8"))
            self.say(f"плоскость стола из {self.plane_path.name} ({self.table_plane.get('created')})")
        except (OSError, ValueError) as e:
            self.say(f"не читается {self.plane_path.name}: {e}")

    def table_plane_here(self, c):
        """Плоскость меток в осях камеры при положении станка c: ((n, d), None) или (None, почему нет)."""
        tp = self.table_plane
        if tp is None:
            return None, "плоскость меток не измерена (t)"
        if tp.get("head_z_mm") != self.args.head_z_mm:
            return None, "высота головки изменилась — повторите t"
        if tp.get("profile_id") and tp["profile_id"] != self.profile_id:
            return None, "содержимое профиля изменилось — повторите t"
        if tp.get("calib") != self.calib_name:
            return None, f"плоскость меток измерена с другой калибровкой ({tp.get('calib')}) - повторите t"
        lay = {int(k): np.array(v, float) for k, v in tp["markers_rel_home_machine_mm"].items()}
        if not self.layout or any(k not in self.layout or np.linalg.norm(self.layout[k] - v) > 0.5 for k, v in lay.items()):
            return None, "раскладка меток другая, чем при замере плоскости - повторите t"
        if {int(k): v for k, v in tp.get("grbl_settings", {}).items()} != self.layout_settings:
            return None, "с замера плоскости менялись $3/$100/$101 - повторите t"
        exp = self.expected()
        home = int(tp["home_id"])
        if home not in exp:
            return None, "нет привязки к меткам в этой сессии (k на любой метке)"
        n, d = np.array(tp["plane"]["n"], float), float(tp["plane"]["d"])
        return (n, d - n @ (self.V @ (np.asarray(c, float) - exp[home]))), None

    def record(self, marker_id, pos, save=True):
        super().record(marker_id, pos, save)
        if self.depth_pass and marker_id not in self.marker_3d:
            self.marker_3d[marker_id] = self.measure_marker_3d(marker_id)

    def measure_marker_3d(self, marker_id):
        """Высота метки: кадры с сетки точек как у скана (--grid x --grid, шаг --baseline), только внутрь стола;
        метка целиком - шаблон, её сдвиг между кадрами - корреляцией (как точки скана). Углы для этого не годятся:
        уточнение углов подгоняет прямые к краям, а бочка их гнёт - смещение растёт к краю кадра (до ~1 px)."""
        c_ref = self.position()
        exp = self.expected()
        mid = np.mean(list(exp.values()), axis=0)
        sgn = np.where(mid - c_ref >= 0, 1.0, -1.0)
        m0 = self.require_marker(marker_id)
        side_px = float(np.mean(np.linalg.norm(np.roll(m0["corners"], -1, axis=0) - m0["corners"], axis=1)))
        patch = int(1.25 * side_px) // 2 * 2 + 1  # метка + немного белого поля
        half_px = patch / 2 + 20
        b = self.args.marker_baseline
        while True:  # метка должна остаться в кадре во всех точках
            steps = np.arange(self.args.marker_grid) * b
            offs = [sgn * np.array([sx, sy]) for sy in steps for sx in steps]
            pred = [m0["center"] + self.A @ o for o in offs]
            if all(half_px < p[0] < self.size[0] - half_px and half_px < p[1] < self.size[1] - half_px for p in pred):
                break
            b *= 0.8
            if b < 2:
                raise Abort(f"ID{marker_id}: для замера высоты метки нет места в кадре")
        frames, cs = [self.rect(self.capture())], [c_ref]
        for o in offs[1:]:
            self.move_approach(c_ref + o - self.position())
            frames.append(self.rect(self.capture()))
            cs.append(self.position())
        self.move_approach(c_ref - self.position())
        Os = [self.V @ (c - c_ref) for c in cs]
        d_guess = self.z_marker if self.z_marker else float(self.n_m[2] * self.K[0, 0] * self.args.marker_size / side_px)
        res = refine_point(frames[0], frames[1:], Os[1:], self.K, self.n_m, d_guess, m0["center"], -4, 4,
                           patch=patch, ref_valid=self.rect.valid)
        if "error" in res:
            raise Abort(f"ID{marker_id}: высота метки не измерилась ({res['error']})")
        if (res['at_range_edge'] or not np.isfinite(res['sigma']) or res['sigma'] > .15
                or res['score'] < .60 or res['peak_margin'] < .045 or sum(res['inlier_views']) < 3):
            raise Abort(f"ID{marker_id}: слабый замер метки, повторите t (score={res['score']:.2f}, "
                        f"sigma={res['sigma']:.3f}, margin={res['peak_margin']:.3f})")
        X = res["X"]
        dist = float(self.n_m @ X)
        side_mm = side_px * X[2] / self.K[0, 0]
        self.say(f"ID{marker_id}: до метки {dist:.3f} ±{res['sigma']:.3f} мм (корреляция {res['score']:.2f}), сторона метки "
                 f"{side_mm:.3f} мм (печать {self.args.marker_size:g}), {len(offs)} точек шагом {b:.1f} мм")
        return {"W": X + self.V @ c_ref, "dist": dist, "sigma": res["sigma"], "score": res["score"],
                "side_mm": side_mm}

    def measure_table_plane(self):
        """t: обход меток (как r) с замером высоты стола у каждой -> плоскость меток."""
        self.require_grbl()
        if (self.args.marker_grid < 2 or not np.isfinite(self.args.marker_baseline)
                or self.args.marker_baseline <= 0):
            raise Abort("--marker-grid должен быть >= 2, --marker-baseline положительным")
        if self.A is None:
            raise Abort("сначала отладка осей (k)")
        if len(self.expected()) < 4:
            raise Abort("нужна раскладка меток: обход r (здесь или в survey_markers.py)")
        if not self.confirm(f"t: обход всех меток, у каждой {self.args.marker_grid ** 2 - 1} доп. точек внутрь стола "
                            f"шагом {self.args.marker_baseline:g} мм"):
            raise Abort("отменено")
        self.marker_3d = {}
        self.depth_pass = True
        try:
            self.run_survey()
        finally:
            self.depth_pass = False
        return self.save_table_plane()

    def save_table_plane(self):
        ids = sorted(self.marker_3d)
        if len(ids) < 3:
            raise Abort("высота измерена меньше чем у 3 меток - плоскость не построить")
        home = self.home_id()
        c_home = self.markers[home]
        P = {k: self.marker_3d[k]["W"] - self.V @ c_home for k in ids}
        n, d = fit_plane(np.array([P[k] for k in ids]))
        resid = {k: float(d - n @ P[k]) for k in ids}
        slope = -(n @ self.V) * 100  # изменение расстояния камера-стол на 100 мм хода X / Y
        data = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "calib": self.calib_name, "home_id": home,
            "markers_rel_home_machine_mm": {str(k): np.asarray(v).tolist() for k, v in self.layout.items()},
            "grbl_settings": {str(k): v for k, v in self.layout_settings.items()},
            "V_cam": self.V.tolist(),
            "points_rel_home_cam_mm": {str(k): P[k].tolist() for k in ids},
            "plane": {"n": n.tolist(), "d": float(d)},
            "head_z_mm": self.args.head_z_mm, "profile_id": self.profile_id,
            "camera_to_table_at_home_mm": float(d),
            "tilt_to_motion_deg": angle_deg(n, self.n_m),
            "rise_per_100mm_machine": {"X": float(-slope[0]), "Y": float(-slope[1])},
            "marker_out_of_plane_mm": {str(k): v for k, v in resid.items()},
            "marker_side_mm": {str(k): float(self.marker_3d[k]["side_mm"]) for k in ids},
            "marker_sigma_mm": {str(k): self.marker_3d[k]["sigma"] for k in ids},
            "note": "оси - камеры при наведении на домашнюю метку; плоскость n·W = d, W = V_cam·(c - c_home) + X_cam",
        }
        self.plane_path.parent.mkdir(parents=True, exist_ok=True)
        self.plane_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.table_plane = data
        self.say(f"--- плоскость стола по {len(ids)} меткам: {self.plane_path.name} ---")
        print(f"камера над столом у ID{home}: {d:.3f} мм; стол к плоскости хода станка: "
              f"{data['tilt_to_motion_deg']:.3f}°")
        print(f"стол поднимается на {data['rise_per_100mm_machine']['X']:+.3f} мм на 100 мм хода X+ и на "
              f"{data['rise_per_100mm_machine']['Y']:+.3f} мм на 100 мм хода Y+ (к камере - плюс)")
        print("метки над плоскостью (неплоскость стола/наклейки): " +
              "  ".join(f"ID{k}: {v:+.3f}" for k, v in resid.items()) + " мм")
        print("сторона метки по 3D: " + "  ".join(f"ID{k}: {data['marker_side_mm'][str(k)]:.3f}" for k in ids) +
              f" мм (печать {self.args.marker_size:g} - так проверяется масштаб высот)")
        return data

    # ---------- скан
    def cover_area(self, positions, center, lo, hi, step, n, distance):
        """Add reachable stops where a sampled ROI has too few complete stereo windows."""
        dims = np.asarray(self.args.area_mm, float)
        margin = self.args.area_margin
        if not np.isfinite([*dims, margin]).all() or min(dims) <= 0 or margin < 0:
            raise Abort("Размеры области должны быть положительными, запас — неотрицательным")
        half_size = dims / 2 + margin
        # No point planning a huge/inaccessible array before checking the travel envelope.
        if np.any(half_size > hi-lo + distance * max(self.size)/min(self.K[0,0], self.K[1,1])):
            raise Abort("Область выходит за доступное поле")
        axes = [np.linspace(-a, a, max(2, int(np.ceil(2*a/2))+1)) for a in half_size]
        xy = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 2)
        ray_base = xy @ self.V.T
        worlds = []
        for height in (-self.args.below, self.args.max_height):
            P = ray_base.copy()
            P[:, 2] += (distance-height-ray_base @ n)/n[2]
            worlds.append(P)
        worlds = np.concatenate(worlds)
        # Read the usable region of a full-size patch, including its full border.
        k = 2*self.args.win + 1
        mask = cv2.erode(self.rect.valid.astype(np.uint8), np.ones((k,k), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        def visible(p):
            proj = (worlds - self.V @ (p-center)) @ self.K.T
            uv = np.rint(proj[:, :2]/proj[:, 2:]).astype(int)
            x, y = uv.T
            ok = (proj[:, 2] > 0) & (x>=0) & (x<mask.shape[1]) & (y>=0) & (y<mask.shape[0])
            ok[ok] = mask[y[ok], x[ok]] != 0
            return ok
        votes = np.sum([visible(p) for p in positions], axis=0)
        need = self.args.min_map_views + 1
        if votes.min() >= need:
            return positions
        limits = np.ceil(half_size/step).astype(int) + 1
        candidates = []
        masks = []
        for y in range(-limits[1], limits[1]+1):
            for x in range(-limits[0], limits[0]+1):
                p = center + np.array([x, y])*step
                if np.any(p-self.backlash_comp < lo) or np.any(p > hi):
                    continue
                if any(np.linalg.norm(p-q) < .01 for q in positions):
                    continue
                candidates.append(p)
                masks.append(visible(p))
        while votes.min() < need and len(positions) < 49 and masks:
            gains = [np.count_nonzero(m & (votes < need)) for m in masks]
            idx = int(np.argmax(gains))
            if not gains[idx]:
                break
            positions.append(candidates.pop(idx))
            votes += masks.pop(idx)
        if votes.min() < need:
            raise Abort("Для области не хватает ракурсов в пределах хода и 49 остановок. Разделите участок")
        return positions

    def plan_scan(self, c0):
        g, b = self.args.grid, self.args.baseline
        if self.args.pose_tag_height is not None and not np.isfinite(self.args.pose_tag_height):
            raise Abort("Некорректная высота опорных меток")
        for x,y,w,h in self.args.table_roi or []:
            if min(x,y) < 0 or min(w,h) <= 0 or x+w > self.size[0]//2 or y+h > self.size[1]//2:
                raise Abort("--table-roi вне карты половинного разрешения")
        if self.args.area_mm and (not np.isfinite([self.args.agree_mm, self.args.cell_mm]).all()
                                  or min(self.args.agree_mm, self.args.cell_mm) <= 0
                                  or not 2 <= self.args.references <= g*g or g*g > 49):
            raise Abort("Проверьте число опорных кадров, шаг ячейки и порог согласия")
        if g < 3 or g % 2 == 0 or not np.isfinite(b) or b <= 0:
            raise Abort("--grid: нечётное число >= 3; --baseline: положительный конечный шаг")
        if not 0 < self.args.min_overlap <= 1:
            raise Abort("--min-overlap должен быть в (0, 1]")
        if not 2 <= self.args.min_map_views <= g * g - 1:
            raise Abort("--min-map-views: от 2 до числа дополнительных ракурсов")
        if self.args.map_views and not self.args.min_map_views <= self.args.map_views <= g * g - 1:
            raise Abort("--map-views должен быть 0 или от --min-map-views до grid²−1")
        if self.args.scan_frames < 1 or not np.isfinite(self.args.scan_settle) or self.args.scan_settle < 0:
            raise Abort("Нужен хотя бы один кадр на точку и конечная неотрицательная пауза")
        k = np.arange(g) - (g - 1) // 2
        exp = self.expected()
        if len(exp) < 4:
            raise Abort("для расширенного скана нужна привязка к четырём меткам (k, затем r)")
        P = np.array(list(exp.values()))
        lo = P.min(axis=0) - self.args.scan_margin
        hi = P.max(axis=0) + self.args.scan_margin
        bc = self.backlash_comp
        if np.any(c0 - bc < lo) or np.any(c0 > hi):
            raise Abort("исходная точка и подход с выборкой люфта должны быть внутри границ скана")
        guess, _ = self.plane_guess(c0)
        n, d, _ = guess
        if not np.isfinite(d) or self.args.max_height >= d - 5:
            raise Abort("--max-height должен быть ниже расстояния до стола минимум на 5 мм")
        try:
            sweep_levels(self.args, d)
        except ValueError as e:
            raise Abort(str(e)) from e
        # Проверка видимой площади на малой маске, на столе и на максимальной высоте.
        mask = cv2.resize(self.rect.valid.astype(np.uint8), (192, 108), interpolation=cv2.INTER_NEAREST)
        S = np.diag([192 / self.size[0], 108 / self.size[1], 1.0])
        Ksmall = S @ self.K
        for _ in range(40):
            axes = []
            for i in range(2):
                candidates = [(k + shift) * b for shift in range(-(g // 2), g // 2 + 1)]
                candidates = [o for o in candidates if c0[i] + o.min() - bc >= lo[i]
                              and c0[i] + o.max() <= hi[i]]
                if not candidates:
                    break
                axes.append(min(candidates, key=lambda o: abs(o.mean())))
            if len(axes) == 2:
                positions = [c0 + [ox, oy] for j, oy in enumerate(axes[1])
                             for ox in (axes[0] if j % 2 == 0 else axes[0][::-1])]
                overlap = 1.0
                for p in positions:
                    for height in (-self.args.below, self.args.max_height):
                        H = plane_homography(Ksmall, self.V @ (p - c0), n, d - height)
                        warped = cv2.warpPerspective(mask, H, (192, 108),
                                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                        overlap = min(overlap, float(np.sum((mask > 0) & (warped > 0)) / max(mask.sum(), 1)))
                if overlap >= self.args.min_overlap:
                    if self.args.area_mm:
                        positions = self.cover_area(positions, c0, lo, hi, b, n, d)
                    for p in positions + [c0]:
                        self.check_target(p)
                        self.check_target(p - bc)  # запас на любой подход с выборкой люфта
                    self.scan_plan = {"actual_step_mm": b, "min_pair_overlap": overlap,
                                      "span_mm": np.ptp(positions, axis=0).tolist()}
                    if self.args.area_mm:
                        self.scan_plan.update(area_mm=self.args.area_mm, area_margin=self.args.area_margin,
                                              initial_grid_overlap=overlap)
                        # The initial pair-overlap guarantee applies to the seed grid only.
                        self.scan_plan['min_pair_overlap'] = min(
                            float(np.count_nonzero(mask & cv2.warpPerspective(
                                mask, plane_homography(Ksmall, self.V @ (p-c0), n, d-height), (192,108),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)) / mask.sum())
                            for p in positions for height in (-self.args.below, self.args.max_height))
                    return positions
            b *= 0.9
            if b < 2:
                break
        raise Abort("не удалось разместить сетку с достаточным перекрытием; переместите камеру ближе к центру")

    def capture(self):
        """Усреднённый кадр (серый, float) после остановки: буфер вычитывается, станок успокаивается."""
        t_end = time.monotonic() + self.args.scan_settle
        while time.monotonic() < t_end:
            self.cap.grab()
        acc, frame = None, None
        times = []
        for i in range(self.args.scan_frames):
            frame = grab_fresh(self.cap, flush=self.args.flush) if i == 0 else read_frame(self.cap)
            times.append(time.monotonic())
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            acc = g if acc is None else acc + g
        if self.tick(frame):
            raise Abort("скан отменён (ESC)")
        self.last_capture = dict(received_s=times, median_interval_s=(
            float(np.median(np.diff(times))) if len(times) > 1 else None))
        return acc / self.args.scan_frames

    def scan_here(self):
        self.require_grbl()
        if self.A is None:
            raise Abort("сначала отладка осей (k) на метке")
        c0 = self.position()
        positions = self.plan_scan(c0)
        span = np.ptp(np.array(positions), axis=0)
        if not self.confirm(f"скан: {len(positions)} кадров, сетка {self.args.grid}x{self.args.grid} шаг "
                            f"{self.scan_plan['actual_step_mm']:.2f} мм (запрошено {self.args.baseline:g}; "
                            f"перекрытие >= {100 * self.scan_plan['min_pair_overlap']:.0f}%; "
                            f"охват X {span[0]:.0f} Y {span[1]:.0f} мм) от X{c0[0]:.2f} "
                            f"Y{c0[1]:.2f}, подход с +, затем назад"):
            raise Abort("скан отменён")
        bc = self.backlash_comp
        self.move(-bc, -bc, feed=self.args.slow_feed, check=False)  # к исходной - с минуса, как ко всем точкам
        self.move(bc, bc, feed=self.args.slow_feed, check=False)
        frames, shot_pos, capture_log = [], [], []
        for p in positions:
            step = p - self.position()
            if np.max(np.abs(step)) > 1e-6:
                self.move_approach(step)
            frames.append(self.capture())
            shot_pos.append(self.position())
            status = self.grbl.reported_position
            capture_log.append(dict(target=p.tolist(), status_frame=self.grbl.reported_frame,
                                    status_xyz=status, **self.last_capture))
            if not status or len(status) < 3 or not np.isfinite(status[:3]).all():
                raise Abort("При XY-сканировании отсутствует фактическая координата XYZ")
            if len(capture_log) > 1:
                first_status = capture_log[0]
                if (first_status['status_frame'] != self.grbl.reported_frame
                        or abs(first_status['status_xyz'][2] - status[2]) > .025):
                    raise Abort("При XY-сканировании изменились Z или система координат")
            if np.linalg.norm(shot_pos[-1]-p) > .05:
                raise Abort("Положение GRBL после остановки отличается от цели более чем на 0,05 мм")
        self.move_approach(c0 - self.position())
        ref_index = int(np.argmin([np.linalg.norm(p - c0) for p in shot_pos]))
        guess, global_plane = self.plane_guess(shot_pos[ref_index])
        meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "calib": self.calib_name,
                "camera_matrix": self.K.tolist(), "dist_coeffs": self.D.tolist(), "model": self.calib_model,
                "V_cam": self.V.tolist(), "guess": {"n": guess[0].tolist(), "d": guess[1], "kind": guess[2]},
                "global_plane": None if global_plane is None else {"n": global_plane[0].tolist(), "d": global_plane[1]},
                "grid": self.args.grid, "baseline": self.scan_plan['actual_step_mm'],
                "profile_id": self.profile_id, "head_z_mm": self.args.head_z_mm,
                "camera_mode": getattr(self.args, "camera_mode", None),
                "captures": capture_log, "scan_settle": self.args.scan_settle,
                "focus": self.focus, "opencv": cv2.__version__, "numpy": np.__version__,
                "requested_baseline": self.args.baseline, "scan_plan": self.scan_plan,
                "height_settings": height_settings(self.args)}
        folder = self.scans_dir / time.strftime("scan_%Y%m%d-%H%M%S")
        meta.update(positions=np.asarray(shot_pos).tolist(), ref_index=ref_index)
        # Preserve raw captures before any experimental image processing.
        save_scan(folder, frames, shot_pos, ref_index, meta)
        stereo_xy = stereo_positions(frames, meta, self.args)
        if "pose_fit" in meta:
            tmp = folder / "scan.tmp.json"
            tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(folder / "scan.json")
        self.say(f"кадры скана сохранены: {folder}")
        self.say("считаю карту высот... (ESC - отмена)")
        try:
            self.scan = ScanResult(frames, stereo_xy, ref_index, self.K, self.D, self.calib_model, self.V, guess,
                                   global_plane, self.args, rect=self.rect, tick=self.sweep_tick, log=self.say)
        except SweepAborted:
            raise Abort("расчёт карты отменён")
        self.sel = None
        self.scan_folder = folder
        self.scan.report(self.say)
        self.scan.save_analysis(folder)
        if self.args.area_mm:
            from area_height import reconstruct
            try:
                reconstruct(frames, dict(meta, positions=np.asarray(shot_pos).tolist(), ref_index=ref_index),
                            self.args, folder / "area", first_scan=self.scan, tick=self.sweep_tick)
            except SweepAborted:
                raise Abort("Расчёт области отменён. Исходники сохранены")
        self.show_map()
        return self.scan

    def plane_guess(self, c_ref):
        gp, why = self.table_plane_here(c_ref)
        if gp is not None:
            return (gp[0], gp[1], "плоскость меток"), gp
        self.say(f"плоскость меток: {why}")
        if self.z_marker is None:
            raise Abort("нет оценки высоты камеры (k на метке)")
        return (self.n_m, self.z_marker, "уровень метки при k"), None

    def sweep_tick(self, i, n):
        if i % 8 and i != n:
            return False
        self.messages[-1:] = [f"считаю карту высот: {100 * i // n}% (ESC - отмена)"]
        return self.tick()

    def show_map(self):
        if self.map is not None and self.scan is not None:
            self.map.show(self.scan.render(self.sel))

    def handle_click(self, x):
        if self.scan is None:
            return
        res = self.scan.measure(x)
        record = dict(xy_px=list(map(float, x)))
        for key in ('h', 'h_map', 'sigma', 'spread', 'score', 'reliable', 'views',
                    'peak_margin', 'at_range_edge', 'machine_xy', 'inlier_views', 'error'):
            if key in res:
                val = res[key]
                record[key] = None if isinstance(val, float) and not np.isfinite(val) else val
        with (self.scan_folder / 'clicks.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if "error" in res:
            self.sel = {"x": list(x)}
            self.say(f"точка: {res['error']}")
        else:
            self.sel = res
            self.say(f"точка: {res['h']:.3f} ±{res['sigma']:.3f} мм над столом (по {res['views'] + 1} кадрам, корреляция "
                     f"{res['score']:.2f})" +
                     (f". Камеру на точку (g): X{res['machine_xy'][0]:.3f} Y{res['machine_xy'][1]:.3f}"
                      if res['reliable'] else ". Ненадёжный замер, переезд недоступен"))
        self.show_map()

    def go_selected(self):
        self.require_grbl()
        if not self.sel or "machine_xy" not in self.sel:
            raise Abort("сначала клик по карте высот (точка с измеренной высотой)")
        if not self.sel.get("reliable"):
            raise Abort("выберите точку с надёжным замером высоты")
        target = np.array(self.sel["machine_xy"], float)
        delta = target - self.position()
        self.check_target(target)
        if not self.confirm(f"навести камеру на точку (высота {self.sel['h']:.2f} мм): {fmt_move(delta)} мм"):
            raise Abort("отменено")
        bc = self.backlash_comp
        self.move(*(delta - bc))  # подход с минуса, как при скане
        self.move(bc, bc, feed=self.args.slow_feed, check=False)
        self.check_selected()

    def check_selected(self):
        """Где выбранная точка в кадре после наведения (по шаблону из опорного кадра скана)."""
        sc, x = self.scan, self.sel["x"]
        r = 40
        xi, yi = int(round(x[0])), int(round(x[1]))
        tpl = sc.full[0][yi - r:yi + r + 1, xi - r:xi + r + 1]
        img = self.rect(self.capture())
        cx, cy = int(round(self.K[0, 2])), int(round(self.K[1, 2]))
        w = 200
        roi = img[max(cy - w, 0):cy + w, max(cx - w, 0):cx + w]
        if tpl.shape != (2 * r + 1, 2 * r + 1) or tpl.std() < 3:
            self.say("наведено; проверить по кадру нечем (нет текстуры у точки)")
            return
        R = cv2.matchTemplate(roi.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(R)
        found = np.array([loc[0] + max(cx - w, 0) + r, loc[1] + max(cy - w, 0) + r], float)
        off = found - self.K[:2, 2]
        z = float(np.asarray(self.sel["X"]) @ np.array([0, 0, 1.0]))
        mm = off * z / np.array([self.K[0, 0], self.K[1, 1]])
        self.say(f"наведено: точка в {np.linalg.norm(off):.1f} px от цели = {np.linalg.norm(mm):.3f} мм на её "
                 f"высоте (корреляция {score:.2f})")

    def focus_on_marker(self):
        raise Abort("фокус здесь не меняется: калибровка (и высоты) верны только при фокусе, с которым она снята")


def add_height_args(ap):
    ap.add_argument("--head-z-mm", type=float, help="Измеренное положение головки, мм в выбранной системе")
    ap.add_argument("--frame-file", type=Path, default=ROOT / "calib/table_frame.json")
    ap.add_argument("--plane-file", type=Path, default=PLANE_FILE)
    ap.add_argument("--area-mm", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"),
                    help="Прямоугольник в командных XY вокруг текущей позиции; захват с запасом и многоопорная карта")
    ap.add_argument("--area-margin", type=float, default=5.)
    ap.add_argument("--cell-mm", type=float, default=.5)
    ap.add_argument("--references", type=int, default=5)
    ap.add_argument("--agree-mm", type=float, default=.35, help="Порог согласия опорных карт, мм")
    ap.add_argument("--grid", type=int, default=5, help="скан: нечётное число кадров по оси (5 -> 25 кадров)")
    ap.add_argument("--baseline", type=float, default=16.0,
                    help="скан: шаг между кадрами, мм (больше - точнее высота, но меньше общая зона кадров)")
    ap.add_argument("--scan-margin", type=float, default=5.0,
                    help="скан: насколько камере можно выходить за прямоугольник меток, мм")
    ap.add_argument("--scan-frames", type=int, default=8, help="скан: кадров на точку (усредняются)")
    ap.add_argument("--scan-settle", type=float, default=0.6, help="скан: пауза после остановки, с (стол качается)")
    ap.add_argument("--max-height", type=float, default=30.0, help="самый высокий ожидаемый объект, мм")
    ap.add_argument("--below", type=float, default=3.0, help="насколько ниже стола искать (пазы, ямки), мм")
    ap.add_argument("--sweep-step", type=float, default=0.25, help="шаг перебора высот для карты, мм")
    ap.add_argument("--map-views", type=int, default=0, help="кадров для карты (0 - все)")
    ap.add_argument("--min-map-views", type=int, default=4, help="минимум дополнительных ракурсов для пикселя карты")
    ap.add_argument("--min-overlap", type=float, default=0.35, help="минимальная доля перекрытия каждого кадра с опорным; шаг уменьшается автоматически")
    ap.add_argument("--marker-grid", type=int, default=3, help="t: сетка замера меток, отдельно от скана объектов")
    ap.add_argument("--marker-baseline", type=float, default=12.0, help="t: шаг замера меток, мм")
    ap.add_argument("--win", type=int, default=9, help="окно сравнения для карты, px (половинное разрешение)")
    ap.add_argument("--min-texture", type=float, default=2.0, help="минимальная текстура (СКО яркости в окне)")
    ap.add_argument("--min-score", type=float, default=0.5, help="минимальная корреляция кадров для карты")
    ap.add_argument("--peak-margin", type=float, default=.045,
                    help="минимальный отрыв от альтернативной высоты по корреляции (0 отключает проверку)")
    ap.add_argument("--texture-ratio", type=float, default=.045,
                    help="минимальная доля градиента поперёк основной текстуры (0 отключает проверку)")
    ap.add_argument("--highpass", type=float, default=4.0,
                    help="сопоставление по кадрам без плавной засветки: из кадра вычитается его размытая копия "
                         "(σ, px половинного разрешения); 0 - выкл")
    ap.add_argument("--no-fill", dest="fill_planes", action="store_false",
                    help="не достраивать однотонные области и места под бликом (чёрная маска платы) по плоскости")
    ap.add_argument("--fill-tol", type=float, default=0.25, help="достройка: точки плоскости в пределах +-мм")
    ap.add_argument("--fill-min-area", type=float, default=25.0,
                    help="достройка: однотонная область меньше стольки мм^2 не достраивается")
    ap.add_argument("--fill-margin", type=float, default=3.0,
                    help="достройка: насколько выходить за крайние измеренные точки плоскости, мм")
    ap.add_argument("--no-pose-refinement", dest="pose_tag_height", action="store_const", const=None,
                    help="Использовать исходные позиции GRBL при пересчёте")
    ap.add_argument("--pose-tag-height", type=float, help="Высота вспомогательных меток ID4–49 над плоскостью guess, мм")
    ap.add_argument("--table-roi", nargs=4, type=int, action="append", metavar=("X","Y","W","H"),
                    help="Открытая поверхность стола, координаты исправленной карты половинного разрешения; повторяемая опция")
    ap.add_argument("--table-near", type=float, default=3.0, help="стол ищется в пределах +-мм от ожидаемого")
    ap.add_argument("--min-height", type=float, default=0.5, help="объект - то, что выше стола на, мм")
    ap.add_argument("--min-area", type=float, default=4.0, help="минимальная площадь объекта, мм^2")
    ap.add_argument("--patch", type=int, default=31, help="клик: размер шаблона, px полного кадра")
    ap.add_argument("--click-range", type=float, default=1.0, help="клик: +-мм вокруг высоты по карте")
    ap.add_argument("--max-sigma", type=float, default=0.05,
                    help="клик: допустимая погрешность (СКО среднего по кадрам), мм")
    ap.add_argument("--analysis-out", type=Path, help="Сохранить пересчёт --from-scan в новом каталоге без GUI")
    ap.add_argument("--from-scan", type=str, default=None, help="разобрать сохранённый скан (без камеры и станка)")
    return ap


def height_settings(args):
    """Settings needed to reproduce the same Z data from the same frames."""
    keys = ('max_height','below','sweep_step','map_views','min_map_views','win','min_texture','min_score',
            'peak_margin','texture_ratio','highpass','fill_planes','fill_tol','fill_min_area','fill_margin',
            'references','cell_mm','agree_mm','table_near','pose_tag_height','table_roi','min_height','min_area','patch','click_range','max_sigma')
    return {key:getattr(args,key) for key in keys if hasattr(args,key)}


def replay_args(parser, meta, argv=None):
    defaults = height_settings(add_height_args(argparse.ArgumentParser()).parse_args([]))
    saved = meta.get("height_settings", {})
    restored = {k: saved[k] for k in defaults if k in saved}
    if '--table-roi' in (sys.argv[1:] if argv is None else argv):
        restored.pop('table_roi', None)
    parser.set_defaults(**restored)
    return parser.parse_args(argv)


def run_offline(args):
    frames, meta = load_scan(args.from_scan)
    g = meta["guess"]
    gp = meta.get("global_plane")
    scan = ScanResult(frames, stereo_positions(frames, meta, args), meta["ref_index"], meta["camera_matrix"],
                      np.array(meta["dist_coeffs"]), meta["model"], meta["V_cam"],
                      (np.array(g["n"]), g["d"], g["kind"]),
                      None if gp is None else (np.array(gp["n"]), gp["d"]), args)
    scan.report()
    if args.analysis_out:
        folder = Path(args.analysis_out)
        folder.mkdir(parents=True, exist_ok=False)
        scan.save_analysis(folder)
        settings = dict(height_settings=height_settings(args), pose_fit=meta.get("pose_fit") if args.pose_tag_height is not None else None,
                        source=str(Path(args.from_scan).resolve()), opencv=cv2.__version__, numpy=np.__version__)
        (folder / 'replay.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        return
    win = MapWindow()
    sel = None
    win.show(scan.render())
    print("клик по карте - высота в точке; q/ESC - выход")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27) or cv2.getWindowProperty(win.name, cv2.WND_PROP_VISIBLE) < 1:
            break
        c = win.take_click()
        if c is not None:
            sel = scan.measure(c)
            if "error" in sel:
                print("точка:", sel["error"])
                sel = {"x": list(c)}
            else:
                print(f"точка ({c[0]:.0f}, {c[1]:.0f}): {sel['h']:.3f} ±{sel['sigma']:.3f} мм (корреляция {sel['score']:.2f}"
                      f"{'' if sel['reliable'] else ', НЕНАДЁЖНО'}), по кадрам "
                      f"{np.round(sel['h_views'], 3).tolist()}")
        win.show(scan.render(sel))
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_survey_args(ap)
    add_height_args(ap)
    args = ap.parse_args()
    if args.from_scan:
        meta = json.loads((Path(args.from_scan) / "scan.json").read_text(encoding="utf-8"))
        return run_offline(replay_args(ap, meta))

    cap = open_camera(args.camera, args.width, args.height, fourcc=args.fourcc or None)
    frame = read_frame(cap)
    size = (frame.shape[1], frame.shape[0])
    calib, calib_path = find_calibration(args.camera, size, args.calib)
    if calib is None:
        cap.release()
        raise SystemExit(f"нет достоверной калибровки камеры {args.camera} для {size[0]}x{size[1]} - запустите с тем же "
                         f"--width/--height, что при калибровке")
    try:
        actual = verify_focus(cap, calib["mode"].get("focus"))
    except ValueError as exc:
        cap.release()
        raise SystemExit(str(exc)) from exc
    args.camera_mode = capture_mode(cap, size, args.fourcc)
    args.verified_focus = actual
    tool = HeightTool(cap, args, WindowUI(f"measure_height [{args.camera}]"), calib, calib_path, map_window=MapWindow())
    tool.focus = actual
    tool.say(f"калибровка: {calib_path.name} ({calib['model']}, RMS {calib['rms']:.2f} px)")
    if tool.table_plane is None:
        tool.say("плоскость меток ещё не измерена - после k сделайте t (или высоты только от стола в кадре)")
    print(__doc__)

    actions = {ord('c'): tool.connect, ord('u'): tool.unlock, ord('f'): tool.focus_on_marker,
               ord('k'): tool.calibrate_axes, ord('i'): tool.invert_axes, ord('m'): tool.center_and_record,
               ord('r'): tool.run_survey, ord('t'): tool.measure_table_plane, ord('p'): tool.scan_here,
               ord('g'): tool.go_selected, ord('h'): tool.go_home}
    try:
        while True:
            frame = read_frame(cap)
            found = detect(tool.detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            tool.seen_marker |= bool(found)
            key = tool.ui.show(tool.render(frame, found))
            try:
                click = tool.map.take_click()
                if click is not None:
                    tool.handle_click(click)
                if key == ord('q') or tool.ui.closed:
                    break
                elif key in JOG_STEPS:
                    tool.jog_step = JOG_STEPS[key]
                elif key in JOG_KEYS:
                    tool.jog_key(key)
                elif key in actions:
                    actions[key]()
            except (Abort, GrblError) as e:
                tool.say(f"СТОП: {e}")
    except KeyboardInterrupt:
        print("Прервано (Ctrl+C)")
    finally:
        if tool.grbl is not None:
            tool.grbl.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
