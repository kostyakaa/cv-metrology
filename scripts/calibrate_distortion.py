"""
Калибровка камеры по доске ChArUco: матрица камеры (fx, fy, cx, cy),
дисторсия (обычная pinhole-модель или fisheye) и live-просмотр undistort.

Доска: печатная ChArUco из scripts/make_chessboard.py, выбирается --board:
10mm (обзорная камера), 5mm / 3mm / 2mm (головная камера). Нужно, чтобы модуль
кода в кадре был >= ~5 px: при 640x480 и ~13.5 px/мм это доска 5mm. Каждый
угол опознаётся по кодам, поэтому доску не обязательно видеть целиком.

Режим камеры: калибровка верна только для того режима (разрешение + кодек +
фокус), в котором снимали (см. scripts/compare_modes.py). Режим и фокус пишутся
в файл калибровки, при загрузке размер кадра сверяется.

=== Как снимать головную камеру (станок + наклоны) ===
  1. Доска 5mm наклеена на тонкую ровную жёсткую пластину и ЗАКРЕПЛЕНА на столе
     (скотч/упоры - стол на дрыгостоле ездит с ускорением). Камера над её центром.
  2. Запуск с портом станка:
       python scripts/calibrate_distortion.py --camera 0 --board 5mm --width 640 --height 480 --grbl COM4
  3. g - авто-съёмка станком: сетка --grid x --grid точек в пределах +-range мм
     (по умолчанию +-25) от текущего положения, к каждой точке подход с одной
     стороны (выборка люфта), остановка, усреднение углов по нескольким кадрам.
     Станок поедет, только если доска уже видна в кадре; за +-range не выйдет.
     ESC - мгновенная остановка.
  4. Наклоны вручную: подложить под край доски 3-5 мм (монета, линейка) -> c;
     по 2-3 кадра с подкладкой с каждой из 4 сторон + пара кадров с доской,
     повёрнутой на 30-45 градусов. Без наклонов фокусное не определяется (кадры
     только со сдвигом для этого вырождены). Сильнее 10-15 градусов не нужно -
     у камеры глубина резкости 1-2 мм.
  5. q - расчёт. Хорошо: RMS < 0.3 px (до 0.5 приемлемо). Выбросы - удалить их PNG
     из папки кадров и пересчитать: --from-dir <папка> (можно и с --fisheye).
  6. Сверка со станком (по кадрам сетки): где была камера относительно доски vs
     заказанные ходы -> сколько станок реально проходит по X и Y (проверка
     $100/$101 по доске, независимо от размера меток), перпендикулярность осей
     станка, поворот доски, невязка позиционирования, высота камеры над доской.

Без станка (--grbl не задан) работает как раньше: c - кадр, доску двигать рукой.

Запуск:
    python scripts/calibrate_distortion.py --camera 0 --board 5mm --width 640 --height 480 --grbl COM4
    python scripts/calibrate_distortion.py --camera 1 --board 10mm --width 1920 --height 1080
    python scripts/calibrate_distortion.py --from-dir calib/frames_cam0_640x480_<время> --fisheye
    python scripts/calibrate_distortion.py --camera 0 --undistort-only --calib calib/calib_cam0_640x480_pinhole.json

Кадры сохраняются без потерь в calib/frames_cam<N>_<WxH>_<время>/ (вместе с ходами станка).

Клавиши: g - авто-съёмка станком, c - кадр вручную, q - закончить и посчитать,
ESC - стоп хода / выход без калибровки, zoom % - масштаб окна.

Отчёт: fx, fy (фокусное в px), главная точка, FOV, дисторсия, ошибка по каждому
кадру. С --pixel-size-um (IMX179 - 1.4 мкм) - фокусное в мм; в режимах с
биннингом эффективный пиксель крупнее (scale из compare_modes.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.camera import load_saved_focus, open_camera, read_frame, set_focus  # noqa: E402
from common.display import add_zoom_trackbar, draw_texts, get_zoom, show_scaled, text_px  # noqa: E402
from common.grbl import Grbl, GrblAborted, GrblError  # noqa: E402
from common.geometry import machine_motion  # noqa: E402
from common.markers import CHARUCO_BOARDS, make_charuco_board  # noqa: E402

COVERAGE_GRID = (8, 6)
# в OpenCV 4 флаги fisheye лежат в cv2.fisheye, в OpenCV 5 - в cv2
FISHEYE = {name: getattr(cv2.fisheye, name, None) or getattr(cv2, name)
           for name in ("CALIB_RECOMPUTE_EXTRINSIC", "CALIB_CHECK_COND", "CALIB_FIX_SKEW")}


def detect_board(detector, board, gray, min_corners):
    """Возвращает (точки доски в мм (N, 3), точки кадра (N, 2), углы, id) или None."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < min_corners or board.checkCharucoCornersCollinear(ids):
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32), corners, ids


def average_detections(dets, board, min_corners):
    """Углы, найденные во всех кадрах неподвижной сцены, усреднённые - меньше шума на кадр."""
    common = set.intersection(*(set(d[3].ravel().tolist()) for d in dets))
    ids = np.array(sorted(common), np.int32)
    if len(ids) < min_corners or board.checkCharucoCornersCollinear(ids):
        return None
    pts = []
    for d in dets:
        by_id = dict(zip(d[3].ravel().tolist(), d[2].reshape(-1, 2)))
        pts.append([by_id[i] for i in ids])
    avg = np.mean(pts, axis=0).astype(np.float32)
    obj, img = board.matchImagePoints(avg.reshape(-1, 1, 2), ids.reshape(-1, 1))
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def covered_cells(points, size):
    gx, gy = COVERAGE_GRID
    w, h = size
    return {(min(int(x / w * gx), gx - 1), min(int(y / h * gy), gy - 1)) for x, y in points}


class Capture:
    def __init__(self, cap, args, board, detector):
        self.cap, self.args, self.board, self.detector = cap, args, board, detector
        self.win = "calibrate - захват доски"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        add_zoom_trackbar(self.win)
        self.views, self.offsets, self.names, self.all_pts = [], [], [], []
        self.frames_dir, self.size, self.grbl, self.grbl_settings = None, None, None, {}

    # ---------- окно
    def preview(self, frame):
        h, w = frame.shape[:2]
        self.size = (w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.args.detect_width / w)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
        return gray, detect_board(self.detector, self.board, small, self.args.min_corners), scale

    def render(self, frame, det, scale, lines):
        vis = frame.copy()
        w, h = self.size
        gx, gy = COVERAGE_GRID
        cells = covered_cells(self.all_pts, self.size)
        for cx in range(gx):
            for cy in range(gy):
                if (cx, cy) not in cells:  # красные клетки - здесь ещё нет ни одного угла
                    cv2.rectangle(vis, (int(cx * w / gx) + 2, int(cy * h / gy) + 2),
                                  (int((cx + 1) * w / gx) - 2, int((cy + 1) * h / gy) - 2), (0, 0, 200), 1)
        for x, y in self.all_pts:
            cv2.circle(vis, (int(x), int(y)), 2, (255, 255, 0), -1)
        if det:  # OpenCV 5 отдаёт углы (N, 2), а рисование ждёт (N, 1, 2) float32
            corners = (det[2].reshape(-1, 1, 2) / scale).astype(np.float32)
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, det[3].reshape(-1, 1), (0, 255, 0))
        size = text_px(vis)
        n_auto = sum(o is not None for o in self.offsets)
        head = (f"кадров: {len(self.views)} (станком {n_auto}, вручную {len(self.views) - n_auto})   доска "
                f"{self.args.board}: {f'{len(det[0])} углов' if det else 'нет'}   покрытие {len(cells)}/{gx * gy}")
        texts = [(head, (8, 6), size, (255, 255, 255), (0, 90, 0) if det else (0, 0, 150))]
        texts += [(line, (8, 6 + (i + 1) * (size + 8)), size, (255, 255, 255), (60, 40, 0)) for i, line in enumerate(lines)]
        return draw_texts(vis, texts)

    def show(self, vis):
        show_scaled(self.win, vis, get_zoom(self.win))
        return cv2.waitKey(1) & 0xFF

    def tick(self, msg):
        frame = read_frame(self.cap)
        _, det, scale = self.preview(frame)
        return self.show(self.render(frame, det, scale, [msg, "ESC - стоп"])) == 27

    def confirm(self, msg):
        print(msg + "   [y] да / [n] нет")
        while True:
            frame = read_frame(self.cap)
            _, det, scale = self.preview(frame)
            key = self.show(self.render(frame, det, scale, [msg, "y - да, n - нет"]))
            if key == ord('y'):
                return True
            if key in (ord('n'), 27):
                return False

    # ---------- кадры
    def add_view(self, frame, obj, img, offset):
        if self.frames_dir is None:
            w, h = self.size
            self.frames_dir = ROOT / "calib" / f"frames_cam{self.args.camera}_{w}x{h}_{time.strftime('%Y%m%d-%H%M%S')}"
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        name = f"view_{len(self.views) + 1:02d}.png"
        cv2.imwrite(str(self.frames_dir / name), frame)
        self.views.append((obj, img))
        self.offsets.append(offset)
        self.names.append(name)
        self.all_pts.extend(img.tolist())
        gx, gy = COVERAGE_GRID
        where = f", станок X{offset[0]:+.2f} Y{offset[1]:+.2f}" if offset is not None else ""
        print(f"Кадр #{len(self.views)}: {len(obj)} углов{where}, покрытие "
              f"{len(covered_cells(self.all_pts, self.size))}/{gx * gy}")

    def capture_manual(self, frame, gray):
        full = detect_board(self.detector, self.board, gray, self.args.min_corners)
        if full is None:
            print(f"На полном разрешении доска не найдена (нужно >= {self.args.min_corners} углов не на одной линии).")
            return
        self.add_view(frame, full[0], full[1], None)

    def drain(self, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:  # вычитываем буфер: в замер не попадут кадры, снятые на ходу
            self.cap.grab()

    def auto_grid(self):
        """Сетка точек +-range мм от текущего положения: подход с одной стороны, остановка, усреднение кадров."""
        a = self.args
        if not a.grbl:
            print("Станок не задан - запустите с --grbl COM4")
            return
        frame = read_frame(self.cap)
        if detect_board(self.detector, self.board, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), a.min_corners) is None:
            print("Доска не видна в кадре - станок не двигаю. Поставьте камеру над доской.")
            return
        R, ap = a.range, a.approach
        xs = np.linspace(-R + ap, R, a.grid)  # подход с минуса на ap не выводит за -R
        plan = [np.array([x, y]) for j, y in enumerate(xs) for x in (xs if j % 2 == 0 else xs[::-1])]
        if not self.confirm(f"авто-съёмка: {len(plan)} точек в пределах +-{R:g} мм по X и Y от текущего "
                            f"положения (F{a.feed:.0f}), в конце возврат сюда"):
            return
        if self.grbl is None:
            self.grbl = Grbl(a.grbl, a.baud)
            print(f"{self.grbl.banner}; состояние {self.grbl.state}")
            try:  # сверка хода по доске верна только при этих $100/$101 - запомнить
                st = self.grbl.settings()
                self.grbl_settings = {str(n): st.get(n) for n in (3, 100, 101)}
            except GrblError:
                pass
        cur = np.zeros(2)
        try:
            self.grbl.require_idle()
            for k, p in enumerate(plan, 1):
                assert np.all(np.abs(p) <= R + 1e-9) and np.all(p - ap >= -R - 1e-9)
                for target in (p - ap, p):  # к каждой точке подход с минуса по обеим осям - люфт выбран
                    d = target - cur
                    self.grbl.jog(d[0], d[1], a.feed, tick=lambda: self.tick(
                        f"сетка {k}/{len(plan)}: X{p[0]:+.1f} Y{p[1]:+.1f} мм от старта"))
                    cur = target
                self.drain(a.settle)
                frames = [read_frame(self.cap) for _ in range(a.view_frames)]
                dets = [detect_board(self.detector, self.board, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), a.min_corners)
                        for f in frames]
                avg = None if any(d is None for d in dets) else average_detections(dets, self.board, a.min_corners)
                if avg is None:
                    print(f"  точка {k}: доска не найдена устойчиво - пропуск")
                    continue
                self.add_view(frames[0], *avg, offset=p.tolist())
            self.grbl.jog(-cur[0], -cur[1], a.feed, tick=lambda: self.tick("возврат в центр"))
            cur = np.zeros(2)
            print("Сетка снята, станок вернулся в стартовую точку. Теперь наклоны вручную (c), потом q.")
        except GrblAborted:
            print(f"Остановлено (ESC). Станок стоит в X{cur[0]:+.2f} Y{cur[1]:+.2f} от старта - назад не везу.")
        except GrblError as e:
            print(f"GRBL: {e} - съёмка сеткой прервана")

    def run(self):
        print("g - авто-съёмка станком, c - кадр вручную (наклоны), q - закончить и посчитать, ESC - выход")
        while True:
            frame = read_frame(self.cap)
            gray, det, scale = self.preview(frame)
            key = self.show(self.render(frame, det, scale, []))
            if key == 27:
                return False
            if key == ord('c'):
                self.capture_manual(frame, gray)
            elif key == ord('g'):
                self.auto_grid()
            elif key == ord('q'):
                return True

    def close(self):
        cv2.destroyWindow(self.win)
        if self.grbl is not None:
            self.grbl.close()


def views_from_dir(frames_dir, args, board, detector, offsets_by_name):
    views, offsets, size = [], [], None
    for path in sorted(Path(frames_dir).glob("*.png")):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if size is not None and (gray.shape[1], gray.shape[0]) != size:
            print(f"  {path.name}: другой размер кадра {gray.shape[1]}x{gray.shape[0]} - пропущен")
            continue
        size = (gray.shape[1], gray.shape[0])
        det = detect_board(detector, board, gray, args.min_corners)
        off = offsets_by_name.get(path.name)
        print(f"  {path.name}: {f'{len(det[0])} углов' if det else 'доска НЕ найдена'}"
              + (f", станок X{off[0]:+.2f} Y{off[1]:+.2f}" if off else ""))
        if det:
            views.append(det[:2])
            offsets.append(off)
    return views, size, offsets


def per_view_errors(obj, img, rvecs, tvecs, project):
    errs = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj = project(o, r, t).reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum((proj - i.reshape(-1, 2)) ** 2, axis=1)))))
    return errs


def calibrate_pinhole(views, image_size, rational=False, K0=None):
    obj = [o for o, _ in views]
    img = [i.reshape(-1, 1, 2) for _, i in views]
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    if K0 is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    D0 = np.zeros(8 if rational else 5)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, K0, D0 if K0 is not None else None, flags=flags)
    errs = per_view_errors(obj, img, rvecs, tvecs, lambda o, r, t: cv2.projectPoints(o, r, t, K, D)[0])
    return rms, K, D, errs


def reliability(views, image_size, K, D, model, rms):
    """Достоверна ли калибровка. Главная проверка: решение заново из фокусного в 2 раза меньше/больше -
    если фокусное 'уплывает', кадры его не определяют (мало наклонов) и параметры - случайные числа."""
    notes = []
    w = image_size[0]
    fov = np.degrees(2 * np.arctan(w / (2 * K[0, 0])))
    if not 10 < fov < 179:
        notes.append(f"поле зрения по fx {fov:.1f}° - неправдоподобно")
    if model.startswith("pinhole") and np.max(np.abs(np.ravel(D)[[0, 1, 4]])) > 10:
        notes.append("коэффициенты дисторсии огромные - модель не сошлась")
    if rms > 1.0:
        notes.append(f"RMS {rms:.2f} px - велико (доска неплоская, блики, смаз или модель не та)")
    if model.startswith("pinhole"):
        fxs = []
        for f in (0.5, 2.0):
            K0 = np.array([[K[0, 0] * f, 0, w / 2], [0, K[1, 1] * f, image_size[1] / 2], [0, 0, 1.0]])
            try:
                fxs.append(calibrate_pinhole(views, image_size, model == "pinhole_rational", K0)[1][0, 0])
            except cv2.error:
                fxs.append(np.inf)
        spread = max(abs(f - K[0, 0]) for f in fxs) / K[0, 0]
        if spread > 0.02:
            notes.append(f"фокусное не определяется кадрами: из другого начального значения выходит "
                         f"{', '.join(f'{f:.0f}' for f in fxs)} вместо {K[0, 0]:.0f} px - нужны кадры с наклоном "
                         f"доски 10-25° в разные стороны")
    return not notes, notes


def calibrate_fisheye(views, image_size):
    obj = [o.reshape(1, -1, 3).astype(np.float64) for o, _ in views]
    img = [i.reshape(1, -1, 2).astype(np.float64) for _, i in views]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    base = FISHEYE["CALIB_RECOMPUTE_EXTRINSIC"] | FISHEYE["CALIB_FIX_SKEW"]
    n = len(obj)
    for flags in (base | FISHEYE["CALIB_CHECK_COND"], base):
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj, img, image_size, np.zeros((3, 3)), np.zeros((4, 1)),
                [np.zeros((1, 1, 3)) for _ in range(n)], [np.zeros((1, 1, 3)) for _ in range(n)], flags, criteria)
            break
        except cv2.error:
            if not flags & FISHEYE["CALIB_CHECK_COND"]:
                raise
            print("fisheye: проверка обусловленности не прошла (есть плохой кадр) - пересчёт без неё")
    errs = per_view_errors(obj, img, rvecs, tvecs, lambda o, r, t: cv2.fisheye.projectPoints(o, r, t, K, D)[0])
    return rms, K, D, errs


def machine_check(views, offsets, K, D, model):
    """Сверка со станком по кадрам сетки (common/geometry.machine_motion): реальный ход по осям, их
    перпендикулярность, поворот доски, невязка позиционирования, высота камеры и ход станка в осях камеры
    (для замера высоты, z/)."""
    res = machine_motion(views, offsets, K, D, model)
    if res is None:
        return None
    sx, sy, mo = res["real_per_cmd_x"], res["real_per_cmd_y"], res["motion"]
    print("--- сверка со станком (кадры сетки, доска как эталон длины) ---")
    print(f"Кадров: {res['views']}, камера над доской на ~{res['camera_height_mm']:.1f} мм")
    for name, s, key in (("X", sx, 100), ("Y", sy, 101)):
        print(f"Ход {name}: станок проходит {100 * s:.2f}% заказанного"
              + ("" if abs(s - 1) < 0.005 else f"  -> ${key} правильное = текущее / {s:.4f}"))
    print(f"Угол между осями X и Y станка: {res['axes_angle_deg']:.3f}° (90 = перпендикулярны)")
    print(f"Доска повёрнута относительно оси X станка на {res['board_rotation_deg']:+.2f}°")
    print(f"Невязка позиционирования: {res['residual_rms_mm']:.4f} мм (люфт/микрошаг станка + шум калибровки)")
    print(f"Ось камеры отклонена от перпендикуляра к плоскости хода на {mo['camera_tilt_deg']:.2f}°, доска к "
          f"плоскости хода - {mo['board_tilt_to_motion_deg']:.2f}°")
    print("    (размер клетки доски сверьте штангенциркулем: печать неточна - и эти % тоже)")
    return res


def print_camera_report(K, D, image_size, rms, errs, pixel_size_um=None):
    w, h = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print("--- Параметры камеры ---")
    print(f"Режим (кадр):            {w}x{h} px")
    print(f"Фокусное расстояние:     fx={fx:.2f} px   fy={fy:.2f} px   (fy/fx={fy / fx:.4f})")
    print(f"Главная точка:           cx={cx:.2f} px   cy={cy:.2f} px   (центр кадра {w / 2:.1f}, {h / 2:.1f})")
    print(f"Поле зрения (FOV):       {np.degrees(2 * np.arctan(w / (2 * fx))):.1f}° x "
          f"{np.degrees(2 * np.arctan(h / (2 * fy))):.1f}°")
    print(f"Коэффициенты дисторсии:  {np.ravel(D).round(5).tolist()}")
    if pixel_size_um:
        print(f"Фокусное расстояние:     {fx * pixel_size_um / 1000:.2f} мм (при пикселе {pixel_size_um} мкм)")
    print(f"RMS репроекции:          {rms:.4f} px" + ("   <- высоковато, см. кадры ниже" if rms > 0.5 else ""))
    med = float(np.median(errs))
    for i, e in enumerate(errs, 1):
        mark = "   <- выброс, лучше переснять" if e > 2 * med and e > 0.3 else ""
        print(f"    кадр {i:2d}: {e:.3f} px{mark}")


def save_calib(path, K, D, image_size, model, rms, errs, mode, check=None, reliable=True, notes=()):
    data = {"model": model, "image_size": list(image_size), "camera_matrix": K.tolist(),
            "dist_coeffs": D.tolist(), "rms": float(rms), "per_view_rms": errs, "mode": mode,
            "machine_check": check, "reliable": reliable, "reliability_notes": list(notes)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def live_undistort(cap, K, D, image_size, model):
    frame = read_frame(cap)
    if (frame.shape[1], frame.shape[0]) != tuple(image_size):
        print(f"ОШИБКА: калибровка снята в режиме {image_size[0]}x{image_size[1]}, а камера сейчас отдаёт "
              f"{frame.shape[1]}x{frame.shape[0]}. Другой режим может кропить/растягивать кадр - калибровка к "
              f"нему не подходит. Запустите с --width {image_size[0]} --height {image_size[1]} (и тем же --fourcc) "
              f"или перекалибруйте в нужном режиме.")
        return
    if model == "fisheye":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)
    else:  # pinhole и pinhole_rational - одна функция, число коэффициентов берётся из D
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, image_size, 1, image_size)
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, image_size, cv2.CV_16SC2)

    win = "undistort (слева оригинал, справа исправленный)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    add_zoom_trackbar(win, default=50)
    print("Просмотр undistort: q/ESC - выход. Прямые края справа должны стать прямыми и у краёв кадра.")
    while True:
        frame = read_frame(cap)
        show_scaled(win, np.hstack([frame, cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)]), get_zoom(win))
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break
    cv2.destroyWindow(win)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fourcc", type=str, default="", help="'' - сырой поток как есть; 'MJPG' - сжатый")
    ap.add_argument("--focus", type=int, default=None, help="выключить автофокус и выставить фокус (0..1023)")
    ap.add_argument("--board", choices=sorted(CHARUCO_BOARDS), default="10mm",
                    help="какая напечатанная доска ChArUco (см. make_chessboard.py)")
    ap.add_argument("--min-corners", type=int, default=8, help="минимум углов доски в кадре для захвата")
    ap.add_argument("--detect-width", type=int, default=960, help="ширина кадра для детекции в превью (скорость)")
    ap.add_argument("--grbl", type=str, default=None, help="порт GRBL (например COM4) - авто-съёмка станком (g)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--range", type=float, default=25.0, help="авто-съёмка: +-мм по X и Y от стартовой точки")
    ap.add_argument("--grid", type=int, default=5, help="авто-съёмка: точек по каждой оси (5 -> 25 кадров)")
    ap.add_argument("--approach", type=float, default=0.3, help="авто-съёмка: подход к точке с минуса, мм (люфт)")
    ap.add_argument("--feed", type=float, default=600, help="авто-съёмка: подача, мм/мин")
    ap.add_argument("--settle", type=float, default=0.4, help="авто-съёмка: пауза после остановки, с")
    ap.add_argument("--view-frames", type=int, default=3, help="авто-съёмка: кадров на точку (углы усредняются)")
    ap.add_argument("--fisheye", action="store_true", help="модель fisheye вместо pinhole (объективы >120°)")
    ap.add_argument("--rational", action="store_true",
                    help="pinhole с 8 коэффициентами дисторсии - для широкоугольных объективов с сильной бочкой")
    ap.add_argument("--calib", type=str, default=None, help="файл калибровки (куда сохранить / что загрузить)")
    ap.add_argument("--from-dir", type=str, default=None, help="посчитать по сохранённым кадрам вместо съёмки")
    ap.add_argument("--undistort-only", action="store_true", help="не калибровать, только live undistort по --calib")
    ap.add_argument("--pixel-size-um", type=float, default=None, help="размер пикселя сенсора, мкм (для f в мм)")
    args = ap.parse_args()

    model = "fisheye" if args.fisheye else ("pinhole_rational" if args.rational else "pinhole")

    if args.undistort_only:
        if not args.calib:
            print("Для --undistort-only нужен --calib <файл калибровки>")
            return
        data = json.loads(Path(args.calib).read_text(encoding="utf-8"))
        K, D, size = np.array(data["camera_matrix"]), np.array(data["dist_coeffs"]), tuple(data["image_size"])
        mode = data.get("mode", {})
        print(f"Калибровка: {data['model']}, кадр {size[0]}x{size[1]}, RMS={data['rms']:.4f} px, режим {mode}")
        cap = open_camera(args.camera, args.width or size[0], args.height or size[1],
                          fourcc=args.fourcc or mode.get("fourcc") or None)
        focus = args.focus if args.focus is not None else (mode.get("focus") if mode.get("autofocus") == 0 else None)
        if focus is not None:
            set_focus(cap, focus)
            print(f"Фокус {focus} (как при калибровке)")
        live_undistort(cap, K, D, size, data["model"])
        cap.release()
        cv2.destroyAllWindows()
        return

    board_params = {"name": args.board, **CHARUCO_BOARDS[args.board]}
    cap = None
    if args.from_dir:
        meta_path = Path(args.from_dir) / "mode.json"
        mode = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        board_params = mode.get("board", board_params)
        board = make_charuco_board(**{k: v for k, v in board_params.items() if k != "name"})
        print(f"Кадры из {args.from_dir} (доска {board_params}):")
        views, size, offsets = views_from_dir(args.from_dir, args, board, cv2.aruco.CharucoDetector(board),
                                              mode.get("machine_offsets", {}))
        camera_index = mode.get("camera", args.camera)
        mode["frames_dir"] = str(Path(args.from_dir).resolve())
    else:
        board = make_charuco_board(**CHARUCO_BOARDS[args.board])
        cap = open_camera(args.camera, args.width, args.height, fourcc=args.fourcc or None)
        focus = args.focus if args.focus is not None else load_saved_focus(args.camera)
        if args.focus is not None:
            set_focus(cap, args.focus)
        if focus is None:
            print("ВНИМАНИЕ: фокус не зафиксирован - автофокус меняет масштаб и калибровка поплывёт. "
                  "Сначала autofocus.py / detect_markers.py (f) или передайте --focus N.")
        capture = Capture(cap, args, board, cv2.aruco.CharucoDetector(board))
        try:
            finished = capture.run()
        finally:
            capture.close()
        views, offsets, size, frames_dir = capture.views, capture.offsets, capture.size, capture.frames_dir
        mode = {"camera": args.camera, "image_size": list(size) if size else None, "fourcc": args.fourcc,
                "autofocus": 0 if focus is not None else 1, "focus": focus, "board": board_params,
                "grbl_settings": capture.grbl_settings,
                "machine_offsets": {n: o for n, o in zip(capture.names, offsets) if o is not None}}
        if frames_dir is not None:
            (frames_dir / "mode.json").write_text(json.dumps(mode, indent=2), encoding="utf-8")
            mode["frames_dir"] = str(frames_dir)
            print(f"Кадры сохранены в {frames_dir}")
        camera_index = args.camera
        if not finished:
            print("Калибровка отменена.")
            cap.release()
            cv2.destroyAllWindows()
            return

    if len(views) < 8:
        print(f"Слишком мало кадров ({len(views)}), нужно хотя бы 8, лучше 25+ (сетка станком + наклоны).")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        return
    if all(o is not None for o in offsets):
        print("ВНИМАНИЕ: все кадры сняты только сдвигом без наклона - фокусное расстояние так не определяется "
              "надёжно. Добавьте 8-12 кадров с наклоном доски (c).")

    print(f"Калибровка по {len(views)} кадрам, модель {model} ...")
    try:
        if args.fisheye:
            rms, K, D, errs = calibrate_fisheye(views, size)
        else:
            rms, K, D, errs = calibrate_pinhole(views, size, rational=args.rational)
    except cv2.error as e:
        print("Калибровка не сошлась:", e)
        print("Нужно больше кадров, разные наклоны доски и углы у краёв кадра.")
        if cap is not None:
            cap.release()
        return

    print_camera_report(K, D, size, rms, errs, args.pixel_size_um)
    reliable, notes = reliability(views, size, K, D, model, rms)
    check = machine_check(views, offsets, K, D, model) if reliable else None
    suffix = "" if reliable else "_unreliable"
    calib_path = Path(args.calib) if args.calib else \
        ROOT / "calib" / f"calib_cam{camera_index}_{size[0]}x{size[1]}_{model}{suffix}.json"
    save_calib(calib_path, K, D, size, model, rms, errs, mode, check, reliable, notes)
    if reliable:
        print("Проверка достоверности: пройдена (фокусное устойчиво к начальному приближению)")
    else:
        print("!!! КАЛИБРОВКА НЕДОСТОВЕРНА - не использовать:")
        for n in notes:
            print(f"    - {n}")
    print(f"Калибровка сохранена: {calib_path}")

    if cap is not None:
        live_undistort(cap, K, D, size, model)
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
