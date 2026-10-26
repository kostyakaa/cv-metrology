"""
Привязка 4 меток углов стола к координатам GRBL-станка: точное наведение камеры
на каждую метку, рамка стола ("виртуальная плоскость") и домашняя точка.

Калибровка камеры (calibrate_distortion.py) подхватывается сама по камере и
режиму (--width/--height): углы меток исправляются от дисторсии, цель наведения -
главная точка объектива (там дисторсии нет), реальные мм считаются по ходу
станка, измеренному по доске при калибровке. Режим и фокус - те же, что при
калибровке.

Безопасность: автоматические ходы едут только туда, где станок уже был, или по
направлению, которое вы сами указали (и не дальше --span + половина окна поиска).
Перед каждым переездом - пробный ход 2 мм: если картинка сдвинулась не так, как
предсказывает отладка осей, длинный ход не начинается. ESC - мгновенная остановка.

Порядок работы:
  1. Камера на любой угловой метке, f - автофокус (если фокус ещё не тот).
  2. c - подключение (--port COM4); k - отладка осей (+2 +2 -2 -2 мм по X и Y:
     направления, масштаб, люфт). i - инвертировать ось в GRBL ($3), если нужно.
  3. r - обход по направлениям:
       - наведение на метку под камерой;
       - "куда следующая метка?" - w/a/s/d (как ручной jog: +Y/-X/-Y/+X станка);
         станок едет на --span-x / --span-y в эту сторону, ищет в окне
         --search-window и наводится на ту метку, которую найдёт (ID любой);
       - второе направление (поперёк первого) - так же;
       - четвёртая метка - достраиванием прямоугольника, сам;
       - возврат на первую (ошибка замыкания) и рамка стола в calib/table_frame.json.
     Если раскладка уже известна (прошлая сессия или показанные метки) - r объезжает
     все метки сам, без вопросов.
  4. m - навестись на метку в кадре и записать её (можно показать метки вручную:
     jog 3 + wasd к каждой и m). h - к домашней метке (ID0) и наведение;
     z - ноль G54 там (G10 L20 P1 X0 Y0).

Следующие сессии: раскладка из calib/table_frame.json подхватывается сама (если
$3/$100/$101 не менялись) - достаточно c, k, r.

Ручной jog: a/d - X-/X+, w/s - Y+/Y-, 1/2/3 - шаг 0.1/1/10 мм (без ограничений -
следите за краями). q - выход.

Запуск (режим как при калибровке):
    python scripts/survey_markers.py --camera 0 --width 1920 --height 1080 --port COM4 --span-x 283 --span-y 111
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
from common.camera import grab_fresh, load_saved_focus, open_camera, read_frame, save_focus  # noqa: E402
from common.display import MarkerHold, add_zoom_trackbar, draw_marker, draw_texts, get_zoom, show_scaled, text_px  # noqa: E402
from common.focus import FocusAborted, autofocus, center_roi, roi_around  # noqa: E402
from common.geometry import undistort_points  # noqa: E402
from common.grbl import Grbl, GrblAborted, GrblError  # noqa: E402
from common.markers import CORNER_LABELS, detect, detect_precise, make_detector  # noqa: E402

JOG_KEYS = {ord('a'): (-1, 0), ord('d'): (1, 0), ord('w'): (0, 1), ord('s'): (0, -1)}
JOG_STEPS = {ord('1'): 0.1, ord('2'): 1.0, ord('3'): 10.0}
LAYOUT_SETTINGS = (3, 100, 101)  # при их изменении сохранённая раскладка недействительна


class Abort(Exception):
    pass


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1))))


def marker_scale(corners, size_mm):
    """px/мм по средней длине сторон метки."""
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return float(sides.mean() / size_mm)


def fmt_move(v):
    return "  ".join(f"{ax} {val:+.2f}" for ax, val in zip("XY", v) if abs(val) >= 0.005) or "0"


def axis_deviation(v):
    """Отклонение направления от ближайшей оси станка, град (-45..45)."""
    a = np.degrees(np.arctan2(v[1], v[0])) % 90
    return float(a if a <= 45 else a - 90)


def find_calibration(camera, size, path=None):
    """Самая свежая достоверная калибровка этой камеры для этого размера кадра (или явно указанная)."""
    if path:
        cands = [Path(path)]
    else:
        cands = sorted((ROOT / "calib").glob(f"calib_cam{camera}_{size[0]}x{size[1]}_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if tuple(data.get("image_size", ())) != tuple(size):
            if path:
                print(f"{p.name}: калибровка для кадра {data.get('image_size')}, а камера отдаёт {list(size)} - не подходит")
            continue
        if "unreliable" in p.name or not data.get("reliable", True):
            continue
        return data, p
    return None, None


class WindowUI:
    def __init__(self, name):
        self.name = name
        self.closed = False
        self.shown = False
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        add_zoom_trackbar(name)

    def show(self, img):
        """Код клавиши; закрытие окна крестиком = ESC (отмена хода) и выход."""
        if self.closed or (self.shown and cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1):
            self.closed = True
            return 27
        show_scaled(self.name, img, get_zoom(self.name))
        self.shown = True
        return cv2.waitKey(1) & 0xFF


class Survey:
    def __init__(self, cap, args, ui, grbl_factory=None, frame_path=None, calib=None):
        self.cap, self.args, self.ui = cap, args, ui
        self.grbl_factory = grbl_factory or (lambda: Grbl(args.port, args.baud))
        self.frame_path = Path(frame_path) if frame_path else ROOT / "calib" / "table_frame.json"
        self.detector = make_detector()                       # живое окно
        self.detector_precise = make_detector(precise=True)   # замеры после остановки
        self.hold = MarkerHold()
        self.grbl = None
        self.A = None              # px (без дисторсии) сдвига картинки на 1 мм хода станка; столбцы - X, Y
        self.axis_mismatch = {}    # ось -> угол расхождения с осями меток, град
        self.move_ratio = np.ones(2)  # реальных мм на 1 мм команды по размеру метки (справочно)
        self.backlash = [0.0, 0.0]
        self.backlash_comp = args.backlash if args.backlash > 0 else 0.1
        self.scale_marker = None   # px/мм по размеру метки
        self.markers = {}          # id -> координаты станка (эта сессия), где центр метки точно в цели
        self.layout = {}           # id -> положение относительно дома из прошлой сессии
        self.layout_settings = {}
        self.visited = []          # где станок побывал - виртуальные концевики
        self.authorized = []       # концы ходов по направлениям, которые подтвердил человек
        self.closure = None
        self.last_centered = None
        self.last_dir = np.zeros(2)  # знак последнего хода по осям (0 - неизвестно): с какой стороны выбран люфт
        self.seen_marker = False
        self.jog_step = 1.0
        self.messages = []
        self.direction_source = None  # для тестов: список направлений вместо вопроса
        self.focus = load_saved_focus(args.camera)
        frame = read_frame(cap)
        self.size = (frame.shape[1], frame.shape[0])
        self.target = np.array([(self.size[0] - 1) / 2, (self.size[1] - 1) / 2])
        self._last_status = 0.0

        self.K = self.D = self.calib_model = self.calib_ratio = None
        self.calib_settings = {}
        if calib is not None:
            self.K, self.D = np.array(calib["camera_matrix"], float), np.array(calib["dist_coeffs"], float)
            self.calib_model = calib["model"]
            self.target = np.array([self.K[0, 2], self.K[1, 2]])  # главная точка - там дисторсии нет
            mc = calib.get("machine_check") or {}
            if mc.get("real_per_cmd_x"):
                self.calib_ratio = np.array([mc["real_per_cmd_x"], mc["real_per_cmd_y"]], float)
            mode = calib.get("mode") or {}
            self.calib_settings = {int(k): v for k, v in (mode.get("grbl_settings") or {}).items()}
            if mode.get("focus") is not None and self.focus is not None and int(mode["focus"]) != int(self.focus):
                self.say(f"ВНИМАНИЕ: калибровка снята при фокусе {mode['focus']}, сейчас {self.focus} - масштаб и "
                         f"дисторсия другие. Верните фокус или перекалибруйте")
        self.load_layout()

    # ---------- вывод
    def say(self, msg):
        print(msg)
        self.messages = (self.messages + [msg])[-3:]

    def undistort(self, pts):
        """Точки кадра -> те же точки без дисторсии (в пикселях той же камеры). Без калибровки - как есть."""
        if self.K is None:
            return np.asarray(pts, np.float64).reshape(-1, 2)
        return undistort_points(pts, self.K, self.D, self.calib_model)

    def render(self, frame, found):
        vis = frame.copy()
        size = text_px(vis)
        th = max(1, vis.shape[1] // 640)
        tx, ty = (int(round(v)) for v in self.target)
        cv2.drawMarker(vis, (tx, ty), (255, 120, 0), cv2.MARKER_CROSS, size * 3, th, cv2.LINE_AA)
        texts = []
        for marker_id, c in found.items():
            x, y = draw_marker(vis, c, (0, 220, 0))
            label = f"ID{marker_id} {CORNER_LABELS.get(marker_id, '')}"
            if marker_id in self.markers:
                label += " (записана)"
            if self.A is not None:
                center = self.undistort(c).mean(axis=0)
                label += f"   до центра: {fmt_move(np.linalg.solve(self.A, self.target - center))} мм"
            texts.append((label, (x, max(0, y - size - 10)), size, (255, 255, 255), (0, 110, 0)))
        for marker_id, (c, ago) in self.hold.update(found).items():
            x, y = draw_marker(vis, c, (0, 150, 255), lost=True)
            texts.append((f"ID{marker_id}: нет в кадре {ago:.1f} с", (x, max(0, y - size - 10)), size,
                          (255, 255, 255), (0, 100, 200)))
        g = "не подключён"
        if self.grbl is not None:
            if time.time() - self._last_status > 0.3:
                try:
                    self.grbl.status()
                except GrblError:
                    pass
                self._last_status = time.time()
            p = self.grbl.mpos or self.grbl.wpos
            g = f"{self.grbl.state}  X{p[0]:.3f} Y{p[1]:.3f}" if p else str(self.grbl.state)
        known = sorted(self.expected())
        head = [f"GRBL: {g}   jog {self.jog_step:g} мм   фокус: {self.focus if self.focus is not None else 'авто!'}"
                f"   калибровка: {'да' if self.K is not None else 'нет'}",
                f"оси: {'откалиброваны' if self.A is not None else 'нет (k)'}   записаны: "
                f"{', '.join(f'ID{k}' for k in sorted(self.markers)) or '-'}   известно где: "
                f"{', '.join(f'ID{k}' for k in known) or '-'}"]
        for i, line in enumerate(head):
            texts.append((line, (8, 6 + i * (size + 8)), size, (220, 255, 220), (0, 0, 0)))
        small = max(12, int(size * 0.85))
        for i, line in enumerate(reversed(self.messages)):
            texts.append((line, (8, vis.shape[0] - (i + 1) * (small + 8) - 4), small, (255, 255, 255), (60, 40, 0)))
        return draw_texts(vis, texts)

    def tick(self, frame=None):
        """Показать кадр во время операции; True = нажат ESC (отмена)."""
        frame = read_frame(self.cap) if frame is None else frame
        found = detect(self.detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return self.ui.show(self.render(frame, found)) == 27

    def wait_key(self):
        frame = read_frame(self.cap)
        return self.ui.show(self.render(frame, detect(self.detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))

    def confirm(self, msg):
        self.say(msg + "   [y] да / [n] нет")
        if not self.args.confirm:
            return True
        while True:
            key = self.wait_key()
            if key == ord('y'):
                return True
            if key in (ord('n'), 27):
                self.say("отменено")
                return False

    def ask_direction(self, prompt):
        self.say(prompt + "   w/a/s/d = +Y/-X/-Y/+X станка (как ручной jog), ESC - отмена")
        if self.direction_source is not None:
            return np.array(self.direction_source.pop(0), float)
        while True:
            key = self.wait_key()
            if key in JOG_KEYS:
                return np.array(JOG_KEYS[key], float)
            if key == 27:
                raise Abort("обход отменён")

    # ---------- зрение
    def measure_all(self, frames=None):
        """Углы всех меток без дисторсии (медиана по кадрам, точное уточнение углов) после остановки станка."""
        t_end = time.time() + self.args.settle
        while time.time() < t_end:  # не спим, а вычитываем: в буфере не останется кадров, снятых на ходу
            self.cap.grab()
        acc = {}
        n = frames or self.args.frames
        for _ in range(n):
            frame = grab_fresh(self.cap, flush=self.args.flush)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for marker_id, c in detect_precise(self.detector, self.detector_precise, gray).items():
                acc.setdefault(marker_id, []).append(self.undistort(c))
        return {i: np.median(v, axis=0) for i, v in acc.items() if len(v) * 2 >= n}

    def measure(self, marker_id=None, attempts=3):
        """Метка (ближайшая к цели, если id не задан). После хода стол может ещё качаться - если метки нет,
        замер повторяется (до attempts раз), а не сразу 'пропала'."""
        for _ in range(attempts):
            found = self.measure_all()
            mid = marker_id
            if mid is None and found:
                mid = min(found, key=lambda i: np.linalg.norm(found[i].mean(axis=0) - self.target))
            if mid in found:
                c = found[mid]
                return {"id": mid, "corners": c, "center": c.mean(axis=0)}
        return None

    def require_marker(self, marker_id=None):
        m = self.measure(marker_id)
        if m is None:
            raise Abort("метка не видна - станок не двигаю" if marker_id is None else
                        f"метка ID{marker_id} не видна - станок не двигаю")
        return m

    # ---------- масштаб и где можно ездить
    def real_ratio(self, direction):
        """Реальных мм на 1 мм команды вдоль направления: по калибровке (доска), иначе по размеру метки."""
        d = unit(direction)
        if self.calib_ratio is not None:
            return float(np.linalg.norm(self.calib_ratio * d))
        return float(np.linalg.norm(self.A @ d) / self.scale_marker)

    def position(self):
        return np.array(self.grbl.position(), float)

    def expected(self):
        """Где ждать метки: записанные в этой сессии + из сохранённой раскладки (через любую записанную)."""
        exp = {k: np.asarray(v, float) for k, v in self.markers.items()}
        anchors = [k for k in self.markers if k in self.layout]
        if anchors:
            origin = np.asarray(self.markers[anchors[0]]) - self.layout[anchors[0]]
            for k, rel in self.layout.items():
                exp.setdefault(k, origin + rel)
        return exp

    def envelope(self):
        """Виртуальные концевики: где станок был, известные метки и подтверждённые ходы, + --envelope мм."""
        pts = np.array(self.visited + list(self.expected().values()) + self.authorized, float)
        return pts.min(axis=0) - self.args.envelope, pts.max(axis=0) + self.args.envelope

    def check_target(self, target):
        lo, hi = self.envelope()
        if np.any(target < lo - 1e-6) or np.any(target > hi + 1e-6):
            raise Abort(f"ход в X{target[0]:.2f} Y{target[1]:.2f} выходит за зону, где станок уже был "
                        f"(X {lo[0]:.1f}..{hi[0]:.1f}, Y {lo[1]:.1f}..{hi[1]:.1f}) - не еду")

    # ---------- движение
    def require_grbl(self):
        if self.grbl is None:
            raise Abort("станок не подключён (c)")

    def move(self, dx, dy, feed=None, manual=False, check=True):
        """Относительный ход. Автоматические ходы - внутри виртуальных концевиков (кроме наведения на метку,
        которую видно в кадре: оно ограничено --max-servo и проверкой расхождения)."""
        self.require_grbl()
        d = np.array([dx, dy], float)
        if np.max(np.abs(d)) > self.args.max_move:
            raise Abort(f"ход {fmt_move(d)} мм длиннее --max-move {self.args.max_move} - запрещено")
        if check and not manual:
            self.check_target(self.position() + d)
        self.last_centered = None
        moving = np.abs(d) > 1e-9
        try:
            self.grbl.jog(dx, dy, feed or self.args.feed, tick=self.tick)
            self.last_dir[moving] = np.sign(d[moving])
        except GrblAborted:
            self.last_dir[moving] = 0
            raise Abort("ход отменён (ESC), станок остановлен")
        finally:
            self.visited.append(self.position())

    def move_approach(self, step):
        """Ход наведения с выборкой люфта: последний подход по каждой оси всегда в +. Если до этого ось ехала
        в минус (или неизвестно куда), даже маленький ход в + люфт не выберет - тогда тоже перебег и подход."""
        b = self.backlash_comp
        pre, post = np.array(step, float), np.zeros(2)
        for i in range(2):
            if step[i] < -1e-6 or self.last_dir[i] <= 0:
                pre[i] -= b
                post[i] = b
        if np.any(np.abs(pre) > 1e-9):
            self.move(*pre, feed=self.args.slow_feed, check=False)
        if np.any(post):
            self.move(*post, feed=self.args.slow_feed, check=False)

    # ---------- раскладка меток между сессиями
    def load_layout(self):
        if not self.frame_path.exists():
            return
        try:
            data = json.loads(self.frame_path.read_text(encoding="utf-8"))
            self.layout = {int(k): np.array(v, float) for k, v in data["frame"]["markers_rel_home_machine_mm"].items()}
            self.layout_settings = {int(k): v for k, v in data.get("grbl_settings", {}).items()}
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.say(f"не читается {self.frame_path.name}: {e} - раскладка не загружена")
            self.layout = {}
            return
        self.say(f"раскладка меток из прошлой сессии: ID {sorted(self.layout)} (проверится по $3/$100/$101 при c)")

    def check_settings(self):
        """Раскладка и поправка хода из калибровки верны только при тех же $3/$100/$101."""
        try:
            cur = self.grbl.settings()
        except GrblError as e:
            cur = None
            self.say(f"не прочитать настройки GRBL ({e})")
        if self.layout:
            changed = [n for n in LAYOUT_SETTINGS if cur is None or self.layout_settings.get(n) != cur.get(n)]
            if changed:
                self.layout = {}
                self.say(f"с прошлой сессии изменились {', '.join(f'${n}' for n in changed)} - раскладка недействительна")
        if self.calib_ratio is not None and self.calib_settings:
            changed = [n for n in (100, 101) if cur is None or self.calib_settings.get(n) != cur.get(n)]
            if changed:
                self.calib_ratio = None
                self.say(f"после калибровки изменились {', '.join(f'${n}' for n in changed)} - её поправку хода не "
                         f"использую (мм по размеру метки)")

    # ---------- операции
    def connect(self):
        if self.grbl is not None:
            self.say("уже подключён")
            return
        if not self.seen_marker:
            raise Abort("сначала метка в кадре - потом подключение к станку")
        self.say(f"подключение к GRBL {self.args.port} @ {self.args.baud}...")
        self.grbl = self.grbl_factory()
        self.say(f"{self.grbl.banner}; состояние {self.grbl.state}")
        self.visited, self.authorized = [self.position()], []
        self.last_dir = np.zeros(2)
        self.markers, self.closure = {}, None
        self.check_settings()
        if self.grbl.state == "Alarm":
            self.say("станок в ALARM - разблокировка клавишей u ($X), если уверены что всё в порядке")

    def unlock(self):
        self.require_grbl()
        if self.confirm("разблокировать станок ($X)?"):
            self.grbl.unlock()
            self.say(f"состояние: {self.grbl.status()}")

    def jog_key(self, key):
        self.require_grbl()
        d = np.array(JOG_KEYS[key]) * self.jog_step
        self.move(*d, feed=self.args.slow_feed if self.jog_step < 5 else self.args.feed, manual=True)

    def focus_on_marker(self):
        frame = read_frame(self.cap)
        found = detect(self.detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if found:
            marker_id = min(found, key=lambda i: np.linalg.norm(found[i].mean(axis=0) - self.target))
            roi = roi_around(found[marker_id], frame)
            self.say(f"автофокус по метке ID{marker_id}...")
        else:
            roi = center_roi(frame)
            self.say("метки нет - автофокус по центру кадра...")
        try:
            f, s, _ = autofocus(self.cap, roi, samples=2, settle=0.15,
                                on_step=lambda fr, fv, sv: self.tick(fr))
        except FocusAborted:
            raise Abort("автофокус отменён (фокус мог остаться не на месте - повторите f)")
        self.focus = f
        self.say(f"фокус {f} (резкость {s:.0f}) сохранён в {save_focus(self.args.camera, f).name}")
        if self.A is not None:
            self.A = None
            self.say("фокус меняет масштаб картинки - калибровку осей (k) нужно повторить")
        if self.K is not None:
            self.say("калибровка камеры снята при другом фокусе - по-хорошему её нужно переснять")

    def calibrate_axes(self):
        """По каждой оси ходы +d +d -d -d с замером метки после каждого. Масштаб - только по ходам без
        смены направления (люфт в него не попадает), люфт - по недоходу на развороте."""
        self.require_grbl()
        m0 = self.require_marker()
        mid = m0["id"]
        s_mark = marker_scale(m0["corners"], self.args.marker_size)
        d = min(self.args.probe, 0.12 * min(self.size) / s_mark)  # уход до 2d - метка остаётся в кадре
        if not self.confirm(f"пробные ходы по метке ID{mid}: X +{d:.2f} +{d:.2f} -{d:.2f} -{d:.2f}, затем так же "
                            f"по Y (F{self.args.slow_feed})"):
            raise Abort("отладка осей отменена")

        cols, backlash = [], []
        for axis, name in ((0, "X"), (1, "Y")):
            pts = [m0["center"] if axis == 0 else self.require_marker(mid)["center"]]
            for sign in (1, 1, -1, -1):
                self.move(*(np.eye(2)[axis] * sign * d), feed=self.args.slow_feed)
                m = self.measure(mid)
                if m is None:
                    raise Abort(f"после хода {name}{'+' if sign > 0 else '-'} метка ID{mid} пропала из кадра - "
                                f"дальше не двигаю")
                pts.append(m["center"])
            p0, pa, pb, pc, pd = pts
            col = ((pb - pa) - (pd - pc)) / (2 * d)
            cols.append(col)
            backlash.append(max(0.0, float(d - np.linalg.norm(pc - pb) / np.linalg.norm(col))))
        A = np.column_stack(cols)
        if abs(np.linalg.det(A)) < 1e-6:
            raise Abort("ходы X и Y сдвигают картинку в одну сторону - оси не различаются, проверьте станок")
        self.A, self.backlash = A, backlash
        self.scale_marker = float(s_mark)
        self.move_ratio = np.linalg.norm(A, axis=0) / s_mark
        # подход к метке всегда с одной стороны - даже если люфт в замере вышел мал (он шумит у порога)
        self.backlash_comp = self.args.backlash if self.args.backlash > 0 else max(0.1, round(2 * max(backlash) + 0.05, 2))
        self.report_axes(m0["corners"])

    def report_axes(self, corners):
        A, s_mark = self.A, self.scale_marker
        sx, sy = np.linalg.norm(A, axis=0)
        ux = unit(corners[1] - corners[0])      # +X по метке в кадре (верхняя сторона метки)
        udown = unit(corners[3] - corners[0])   # -Y по метке в кадре (верх метки = +Y)
        self.say("--- отладка осей ---")
        print(f"Ход X +1 мм: картинка сдвигается на ({A[0, 0]:+.2f}, {A[1, 0]:+.2f}) px = {sx:.2f} px/мм")
        print(f"Ход Y +1 мм: картинка сдвигается на ({A[0, 1]:+.2f}, {A[1, 1]:+.2f}) px = {sy:.2f} px/мм")
        print(f"По размеру метки {self.args.marker_size:g} мм: {s_mark:.2f} px/мм"
              + ("  (углы без дисторсии - по калибровке)" if self.K is not None else ""))
        for name, col, expect in (("X", A[:, 0], -ux), ("Y", A[:, 1], udown)):
            ang = angle_deg(col, expect)
            self.axis_mismatch[name] = ang
            if ang < 10:
                verdict = "совпадает с осью меток"
            elif ang > 170:
                verdict = ("противоположна оси меток (на обход не влияет; чтобы +X/+Y станка смотрели как метки - "
                           "клавиша i, $3)")
            else:
                verdict = f"повёрнута относительно меток на {ang:.1f}° (на обход не влияет)"
            self.say(f"ось {name} станка: {verdict}")
        print(f"Угол между осями X и Y станка в кадре: {angle_deg(A[:, 0], A[:, 1]):.2f}° (90 = перпендикулярны)")
        words = {(1, 0): "вправо", (-1, 0): "влево", (0, 1): "вниз", (0, -1): "вверх"}
        for name, col in (("X", A[:, 0]), ("Y", A[:, 1])):
            cam = -col  # камера относительно стола движется против сдвига картинки
            i = int(np.argmax(np.abs(cam)))
            where = words[tuple(int(np.sign(cam[j])) if j == i else 0 for j in range(2))]
            print(f"Направление {name}+: камера смещается по своему кадру {where} (метки в окне уезжают в обратную "
                  f"сторону) - сверьте: при {'d' if name == 'X' else 'w'} голова/стол должны ехать так же")
        if self.calib_ratio is not None:
            self.say(f"ход по калибровке (доска): X {100 * self.calib_ratio[0]:.2f}%, Y {100 * self.calib_ratio[1]:.2f}% "
                     f"заказанного - им пересчитываются мм")
            print(f"    по размеру метки (справочно): X {100 * self.move_ratio[0]:.1f}%, Y {100 * self.move_ratio[1]:.1f}%")
        else:
            try:
                steps = self.grbl.settings()
            except GrblError:
                steps = {}
            for name, ratio, key in (("X", self.move_ratio[0], 100), ("Y", self.move_ratio[1], 101)):
                cur = steps.get(key)
                fix = "" if abs(ratio - 1) < 0.01 else (f" - ${key}: {cur:g} -> {cur / ratio:.3f}?" if cur else "")
                self.say(f"ход {name} по размеру метки: {100 * ratio:.1f}% заказанного{fix} (оценка грубая - "
                         f"точнее калибровка с доской и станком)")
        for name, b in zip("XY", self.backlash):
            print(f"Люфт {name}: {b:.3f} мм (недоход на развороте)")
        self.say(f"выборка люфта: подход к метке всегда с + (перебег {self.backlash_comp:.2f} мм)")

    def invert_axes(self):
        """Инвертировать в GRBL ($3) оси, которые отладка осей нашла противоположными меткам."""
        self.require_grbl()
        if not self.axis_mismatch:
            raise Abort("сначала отладка осей (k) - по ней видно, какие оси инвертировать")
        flip = [k for k, v in self.axis_mismatch.items() if v > 150]
        rotated = [k for k, v in self.axis_mismatch.items() if 30 < v <= 150]
        if rotated:
            raise Abort(f"оси {rotated} повёрнуты относительно меток, а не противоположны - инверсией не "
                        f"исправить, проверьте наклейку меток")
        if not flip:
            self.say("по последней отладке осей инвертировать нечего")
            return
        old = int(self.grbl.settings().get(3, 0))
        new = old ^ sum({"X": 1, "Y": 2}[k] for k in flip)
        if not self.confirm(f"инвертировать {' и '.join(flip)} в GRBL: $3={old} -> $3={new}. Меняет направление "
                            f"этих осей для ВСЕХ программ на станке, записанные метки сбросятся"):
            raise Abort("инверсия отменена")
        self.grbl.cmd(f"$3={new}")
        self.A, self.axis_mismatch, self.markers, self.closure, self.layout = None, {}, {}, None, {}
        self.say(f"записано $3={new}. Повторите k и обход")

    def center_on(self, marker_id=None):
        """Наведение: центр метки -> цель (главная точка). Возвращает (id, координаты станка точного совпадения).

        Ходы в сотки мм станок отрабатывает неточно (микрошаг под нагрузкой, люфт, залипание стола), а камера
        меряет остаток точнее. Поэтому координата метки = позиция станка + остаток по камере: если очередной ход
        уже не уменьшает ошибку, остаток до --accept принимается и учитывается поправкой."""
        self.require_grbl()
        if self.A is None:
            raise Abort("сначала отладка осей (k)")
        Ainv = np.linalg.inv(self.A)
        if np.any(self.last_dir <= 0):  # люфт выбран не с той стороны - замер дал бы смещение на весь люфт
            self.move_approach(np.zeros(2))
        m = self.require_marker(marker_id)
        marker_id = m["id"]
        prev = None
        for i in range(self.args.max_iter + 1):
            mv = Ainv @ (self.target - m["center"])
            n = float(np.linalg.norm(mv))
            # у предела точности станка ошибка может и подрасти; настоящее расхождение растёт в разы и выше 0.2 мм
            if prev is not None and n > 1.5 * prev and n > max(4 * self.args.accept, 0.2):
                raise Abort(f"наведение на ID{marker_id} расходится ({prev:.3f} -> {n:.3f} мм): станок едет не так, "
                            f"как в отладке осей (сменили $3, провод, муфта?) - стоп, повторите k")
            stalled = prev is not None and n > 0.7 * prev
            if n <= self.args.tol or (n <= self.args.accept and (stalled or i == self.args.max_iter)):
                pos = self.position() + mv
                self.last_centered = marker_id
                note = "" if n <= self.args.tol else ", точнее станок не доводит - учтён поправкой"
                self.say(f"ID{marker_id} наведена: X{pos[0]:.3f} Y{pos[1]:.3f} (остаток {n:.3f} мм{note})")
                return marker_id, pos
            if i == self.args.max_iter:
                break
            prev = n
            if n > self.args.max_servo:
                mv = mv / n * self.args.max_servo
            self.move_approach(mv)
            m = self.require_marker(marker_id)
        raise Abort(f"наведение на ID{marker_id} не сошлось: остаток {n:.3f} мм больше --accept {self.args.accept} мм")

    def record(self, marker_id, pos, save=True):
        old = self.markers.get(marker_id)
        self.markers[marker_id] = np.asarray(pos, float)
        note = "" if old is None else f" (от прошлого замера {fmt_move(pos - old)} = {np.linalg.norm(pos - old):.3f} мм)"
        self.say(f"ID{marker_id} записана{note}; записаны: {', '.join(f'ID{k}' for k in sorted(self.markers))}")
        if save and len(self.markers) >= 4:
            self.save_frame()

    def center_and_record(self):
        self.record(*self.center_on())

    def verify_direction(self, marker_id, direction):
        """Пробный ход в сторону цели, пока видна текущая метка: картинка должна сдвинуться так, как предсказывает
        отладка осей. Ловит ось, которая поехала не туда (сменили $3, перепутаны провода, сорвало муфту)."""
        p0 = self.require_marker(marker_id)["center"]
        step = unit(direction) * self.args.probe
        self.move(*step, feed=self.args.slow_feed)
        m = self.measure(marker_id)
        if m is None:
            raise Abort(f"после пробного хода {fmt_move(step)} мм метка ID{marker_id} пропала - дальше не еду")
        got, expect = m["center"] - p0, self.A @ step
        ang, ratio = angle_deg(got, expect), np.linalg.norm(got) / np.linalg.norm(expect)
        if ang > 15 or not 0.7 < ratio < 1.3:
            raise Abort(f"станок поехал не так, как ожидалось: картинка сдвинулась на ({got[0]:+.1f}, {got[1]:+.1f}) px "
                        f"вместо ({expect[0]:+.1f}, {expect[1]:+.1f}) - длинный ход отменён. Повторите k")
        return step

    def travel_to_point(self, target, what):
        """По прямой в известную точку (внутри зоны, где станок был) с пробой направления."""
        here = self.require_marker()  # ехать только от видимой метки: по ней проверяется направление
        delta = np.asarray(target, float) - self.position()
        dist = float(np.linalg.norm(delta))
        if dist <= 3 * self.args.probe:
            return
        self.check_target(np.asarray(target, float))
        if not self.confirm(f"ход к {what}: {fmt_move(delta)} мм по прямой (F{self.args.feed:.0f}), сначала проба "
                            f"{self.args.probe:g} мм"):
            raise Abort("отменено")
        delta = delta - self.verify_direction(here["id"], delta)
        self.move(*delta)

    def travel_to(self, marker_id):
        """К известной метке (записанной или из раскладки), затем наведение."""
        self.require_grbl()
        if self.A is None:
            raise Abort("сначала отладка осей (k)")
        exp = self.expected()
        if marker_id not in exp:
            raise Abort(f"где ID{marker_id}, ещё неизвестно - обход r или подведите камеру (wasd) и m")
        self.travel_to_point(exp[marker_id], f"ID{marker_id}")
        if self.measure(marker_id) is None:
            raise Abort(f"ID{marker_id} нет там, где она была записана - метку сдвинули или станок потерял шаги. "
                        f"Подведите камеру вручную и m")
        return self.center_on(marker_id)

    def explore_leg(self, here_id, direction):
        """Ход по указанному человеком направлению на --span, поиск новой метки, наведение на неё."""
        span = self.args.span_x if abs(direction[0]) > 0.5 else self.args.span_y
        if span is None:
            raise Abort("для обхода по направлениям нужны --span-x и --span-y (расстояния между центрами меток, мм)")
        ratio = self.real_ratio(direction)
        span_c, win_c = span / ratio, self.args.search_window / ratio  # реальные мм -> мм команды
        v = self.A @ direction
        fov = min(self.size[0] / abs(v[0]) if abs(v[0]) > 1e-6 else np.inf,
                  self.size[1] / abs(v[1]) if abs(v[1]) > 1e-6 else np.inf)
        step = max(1.0, 0.5 * (fov - self.args.marker_size / ratio))
        jump, limit = max(0.0, span_c - win_c / 2), span_c + win_c / 2
        if not self.confirm(f"от ID{here_id}: {fmt_move(direction * jump)} мм (~{jump * ratio:.0f} мм реально), затем "
                            f"поиск шагами {step:.1f} мм до {limit:.0f} - новая зона, по вашему направлению"):
            raise Abort("обход отменён")
        self.authorized.append(self.position() + direction * limit)
        known = set(self.markers)
        moved = self.verify_direction(here_id, direction)
        self.move(*(direction * jump - moved))
        travelled = jump
        while True:
            found = self.measure_all()
            new = [k for k in found if k not in known]
            if new:
                return self.center_on(min(new, key=lambda k: np.linalg.norm(found[k].mean(axis=0) - self.target)))
            if travelled + step > limit + 1e-6:
                raise Abort(f"новой метки нет до {travelled:.0f} мм в этом направлении - стоп. Проверьте направление "
                            f"и --span")
            self.move(*(direction * step), feed=self.args.slow_feed)
            travelled += step

    def find_new_marker(self, what):
        found = self.measure_all()
        new = [k for k in found if k not in self.markers]
        if not new:
            raise Abort(f"в {what} новой метки нет - прямоугольник не сходится? Подведите камеру вручную и m")
        return self.center_on(min(new, key=lambda k: np.linalg.norm(found[k].mean(axis=0) - self.target)))

    def run_survey(self):
        """Обход: по известной раскладке - сам; иначе по направлениям, которые укажет человек."""
        self.require_grbl()
        if self.A is None:
            raise Abort("сначала отладка осей (k)")
        start, pos = self.center_on()
        self.record(start, pos, save=False)
        first = self.markers[start]
        exp = self.expected()
        if len(exp) >= 4:
            ids = list(exp)
            c = np.mean([exp[k] for k in ids], axis=0)
            ring = sorted(ids, key=lambda k: np.arctan2(*(exp[k] - c)[::-1]))
            i0 = ring.index(start)
            for k in ring[i0 + 1:] + ring[:i0]:
                self.record(*self.travel_to(k), save=False)
        else:
            d1 = self.ask_direction(f"куда от ID{start} ближайшая соседняя метка?")
            m1, p1 = self.explore_leg(start, d1)
            self.record(m1, p1, save=False)
            d2 = self.ask_direction(f"куда от ID{m1} следующая метка (поперёк первого хода)?")
            if abs(np.dot(d1, d2)) > 0.1:
                raise Abort("второе направление должно быть поперёк первого")
            m2, p2 = self.explore_leg(m1, d2)
            self.record(m2, p2, save=False)
            corner4 = first + (p2 - p1)  # достраиваем прямоугольник
            self.travel_to_point(corner4, "4-го угла (по прямоугольнику)")
            self.record(*self.find_new_marker("4-м углу"), save=False)
        _, back = self.travel_to(start)
        self.closure = back - first
        err = float(np.linalg.norm(self.closure))
        self.say(f"ошибка замыкания (возврат на ID{start}): {fmt_move(self.closure)} = {err:.3f} мм"
                 + ("  <- больше 0.05: пропуск шагов/люфт/проскальзывание?" if err > 0.05 else ""))
        return self.save_frame()

    def home_id(self):
        return 0 if 0 in self.markers else min(self.markers)

    def table_frame(self):
        P = {k: np.asarray(v, float) for k, v in self.markers.items()}
        home = self.home_id()
        c = np.mean(list(P.values()), axis=0)
        ring = sorted(P, key=lambda k: np.arctan2(*(P[k] - c)[::-1]))  # обход по контуру, без диагоналей
        n = len(ring)
        edges = [(ring[i], ring[(i + 1) % n]) for i in range(n)]
        diagonals = [(ring[0], ring[2]), (ring[1], ring[3])] if n == 4 else []

        def real(v):  # мм команды -> реальные мм
            v = np.asarray(v, float)
            d = np.linalg.norm(v)
            return float(d * self.real_ratio(v / d)) if d > 0 else 0.0

        return {
            "home_id": home, "home_machine_session": P[home].tolist(),
            "markers_rel_home_machine_mm": {str(k): (P[k] - P[home]).tolist() for k in P},
            "ring_order": ring,
            "edges_mm": {f"{a}-{b}": float(np.linalg.norm(P[b] - P[a])) for a, b in edges},
            "edges_real_mm": {f"{a}-{b}": real(P[b] - P[a]) for a, b in edges},
            "edges_angle_to_machine_axis_deg": {f"{a}-{b}": axis_deviation(P[b] - P[a]) for a, b in edges},
            "diagonals_mm": {f"{a}-{b}": float(np.linalg.norm(P[b] - P[a])) for a, b in diagonals},
            "diagonals_real_mm": {f"{a}-{b}": real(P[b] - P[a]) for a, b in diagonals},
            "corner_angles_deg": {str(ring[i]): angle_deg(P[ring[i - 1]] - P[ring[i]], P[ring[(i + 1) % n]] - P[ring[i]])
                                  for i in range(n)},
        }

    def save_frame(self):
        frame = self.table_frame()
        try:
            st = self.grbl.settings()
            settings = {str(n): st.get(n) for n in LAYOUT_SETTINGS}
        except GrblError:
            settings = {}
        data = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera": self.args.camera, "frame_size": list(self.size), "focus": self.focus,
            "target_px": self.target.tolist(), "grbl_settings": settings, "calibrated_camera": self.K is not None,
            "A_px_per_mm": self.A.tolist(), "scale_marker_px_per_mm": self.scale_marker,
            "move_ratio_by_marker": self.move_ratio.tolist(),
            "move_ratio_by_calibration": None if self.calib_ratio is None else self.calib_ratio.tolist(),
            "backlash_mm": self.backlash, "backlash_comp_mm": self.backlash_comp,
            "markers_machine_session": {str(k): np.asarray(v).tolist() for k, v in self.markers.items()},
            "closure_mm": None if self.closure is None else self.closure.tolist(),
            "frame": frame,
            "note": "markers_machine_session - координаты станка ЭТОЙ сессии (без концевиков ноль сбрасывается "
                    "при переподключении); переносима раскладка markers_rel_home_machine_mm (при тех же $3/$100/$101).",
        }
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.layout = {int(k): np.array(v, float) for k, v in frame["markers_rel_home_machine_mm"].items()}
        self.layout_settings = {int(k): v for k, v in settings.items()}

        self.say(f"--- рамка стола: {len(self.markers)} метки, дом ID{frame['home_id']} ---")
        for k in sorted(self.markers):
            rel = frame["markers_rel_home_machine_mm"][str(k)]
            print(f"ID{k}: относительно ID{frame['home_id']}  X{rel[0]:+9.3f} Y{rel[1]:+9.3f} мм команды")
        for key in frame["edges_mm"]:
            print(f"сторона {key}: {frame['edges_mm'][key]:8.3f} мм команды = {frame['edges_real_mm'][key]:8.2f} мм "
                  f"реально, отклонение от оси станка {frame['edges_angle_to_machine_axis_deg'][key]:+.3f}°")
        if frame["diagonals_mm"]:
            print("диагонали: " + "  ".join(f"{k}: {v:.3f} мм команды ({frame['diagonals_real_mm'][k]:.2f} реально)"
                                           for k, v in frame["diagonals_mm"].items()))
        print("углы: " + "  ".join(f"ID{k}: {v:.3f}°" for k, v in frame["corner_angles_deg"].items()))
        print("    <- реальные стороны сверьте с линейкой (расстояния между центрами меток)")
        self.say(f"рамка сохранена: {self.frame_path}")
        return data

    def go_home(self):
        if not self.markers and not self.layout:
            raise Abort("рамки ещё нет - сначала обход r")
        home = 0 if 0 in self.expected() else min(self.expected())
        self.record(*self.travel_to(home))

    def set_zero(self):
        self.require_grbl()
        if not self.markers or self.last_centered != self.home_id():
            raise Abort("сначала навестись на домашнюю метку (h)")
        if self.confirm(f"записать ноль рабочей СК G54 здесь, на ID{self.last_centered} (G10 L20 P1 X0 Y0)?"):
            before = self.position()
            self.grbl.set_work_zero_here()
            shift = self.position() - before  # не 0, если GRBL шлёт рабочие координаты ($10=0)
            self.markers = {k: np.asarray(v) + shift for k, v in self.markers.items()}
            self.visited = [np.asarray(v) + shift for v in self.visited]
            self.authorized = [np.asarray(v) + shift for v in self.authorized]
            self.say(f"G54 X0 Y0 = центр метки ID{self.last_centered} в цели камеры")


def add_survey_args(ap):
    """Аргументы камеры, станка и наведения (общие с z/measure_height.py)."""
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fourcc", type=str, default="", help="'' - сырой поток как есть; 'MJPG' - сжатый")
    ap.add_argument("--calib", type=str, default=None,
                    help="файл калибровки камеры (по умолчанию - свежий достоверный calib/calib_cam<N>_<WxH>_*.json)")
    ap.add_argument("--no-calib", action="store_true", help="не использовать калибровку камеры")
    ap.add_argument("--port", type=str, default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--span-x", type=float, default=None, help="расстояние между центрами меток по X, мм линейкой")
    ap.add_argument("--span-y", type=float, default=None, help="расстояние между центрами меток по Y, мм линейкой")
    ap.add_argument("--search-window", type=float, default=30.0, help="окно поиска вокруг --span, мм")
    ap.add_argument("--marker-size", type=float, default=15.0, help="сторона метки (чёрная часть), мм")
    ap.add_argument("--probe", type=float, default=2.0, help="пробный ход (отладка осей, проба направления), мм")
    ap.add_argument("--envelope", type=float, default=30.0,
                    help="виртуальные концевики: насколько автоматическим ходам можно выходить за зону, где станок был, мм")
    ap.add_argument("--feed", type=float, default=1500, help="подача переездов, мм/мин")
    ap.add_argument("--slow-feed", type=float, default=300, help="подача наведения и проб, мм/мин")
    ap.add_argument("--max-move", type=float, default=400, help="максимальная длина одного хода, мм")
    ap.add_argument("--max-servo", type=float, default=30, help="максимальный шаг наведения, мм")
    ap.add_argument("--tol", type=float, default=0.02, help="цель наведения: ближе станок не доводит, мм")
    ap.add_argument("--accept", type=float, default=0.05,
                    help="остаток, который принимается (с поправкой по камере), если станок дальше не доводит, мм")
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--backlash", type=float, default=0.0,
                    help="перебег для выборки люфта (подход всегда с +), мм; 0 - по замеру люфта в k")
    ap.add_argument("--settle", type=float, default=0.3, help="пауза после остановки перед замером, с")
    ap.add_argument("--frames", type=int, default=5, help="кадров на замер метки (медиана)")
    ap.add_argument("--flush", type=int, default=1, help="кадров пропустить между кадрами замера")
    ap.add_argument("--no-confirm", dest="confirm", action="store_false",
                    help="не спрашивать подтверждение ходов (только когда всё уже отлажено!)")
    return ap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_survey_args(ap)
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, fourcc=args.fourcc or None)
    frame = read_frame(cap)
    size = (frame.shape[1], frame.shape[0])
    calib, calib_path = (None, None) if args.no_calib else find_calibration(args.camera, size, args.calib)
    survey = Survey(cap, args, WindowUI(f"survey_markers [{args.camera}]"), calib=calib)
    if calib is not None:
        survey.say(f"калибровка камеры: {calib_path.name} ({calib['model']}, RMS {calib['rms']:.2f} px) - углы меток "
                   f"без дисторсии, цель - главная точка ({survey.target[0]:.1f}, {survey.target[1]:.1f})")
    else:
        survey.say(f"калибровки камеры для {size[0]}x{size[1]} нет - замеры с дисторсией, цель - центр кадра")
    if survey.focus is None:
        survey.say("фокус не сохранён - наведите камеру на метку и нажмите f")
    print(__doc__)

    actions = {ord('c'): survey.connect, ord('u'): survey.unlock, ord('f'): survey.focus_on_marker,
               ord('k'): survey.calibrate_axes, ord('i'): survey.invert_axes, ord('m'): survey.center_and_record,
               ord('r'): survey.run_survey, ord('h'): survey.go_home, ord('z'): survey.set_zero}
    try:
        while True:
            frame = read_frame(cap)
            found = detect(survey.detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            survey.seen_marker |= bool(found)
            key = survey.ui.show(survey.render(frame, found))
            try:
                if key == ord('q') or survey.ui.closed:
                    break
                elif key in JOG_STEPS:
                    survey.jog_step = JOG_STEPS[key]
                elif key in JOG_KEYS:
                    survey.jog_key(key)
                elif key in actions:
                    actions[key]()
            except (Abort, GrblError) as e:
                survey.say(f"СТОП: {e}")
    except KeyboardInterrupt:
        print("Прервано (Ctrl+C)")
    finally:
        if survey.grbl is not None:
            survey.grbl.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
