"""Единственное место, где код различает Windows и Linux. Остальной код платформу не проверяет, а зовёт отсюда:

  CAMERA_BACKEND                      бэкенд OpenCV: DSHOW (Windows) / V4L2 (Linux)
  resolve_device(n)                   номер камеры проекта -> номер устройства (Linux: calib/camera_map.json)
  camera_devices()                    камеры системы [{'index', 'name'}] (пусто - не определить, перебирайте номера)
  camera_formats(n, raw_device)       режимы камеры [(fourcc, ширина, высота, макс. fps)]
  control_ranges(n)                   диапазоны регулировок камеры {имя PROPS: (мин, макс)} или None (взять таблицу
                                      по умолчанию - Windows)
  toggle_auto_exposure(cur)           значение CAP_PROP_AUTO_EXPOSURE для переключения авто/ручная
  camera_open_hint()                  подсказка к ошибке "камера не открылась"
  default_serial_port(), serial_open_hint(err, port)   порт станка и подсказка к ошибке открытия
  font_files()                        шрифты с кириллицей для подписей

Windows - DirectShow (формат/список через pygrabber), Linux - V4L2 (v4l2-ctl из v4l-utils). Другие ОС: камера - по
умолчанию OpenCV, остальное как на Linux без V4L2-помощников."""
import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
CAMERA_BACKEND = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_ANY
CAMERA_MAP_FILE = ROOT / "calib" / "camera_map.json"

# имя регулировки проекта (scripts/camera_params.py) -> имя регулировки V4L2
_V4L2_CONTROLS = {
    "Focus": "focus_absolute", "Autofocus": "focus_automatic_continuous", "Exposure": "exposure_time_absolute",
    "AutoExposure": "auto_exposure", "Brightness": "brightness", "Contrast": "contrast", "Saturation": "saturation",
    "Gain": "gain", "WB_Temp": "white_balance_temperature", "AutoWB": "white_balance_automatic",
}


# ---------- камеры: какая под каким номером
def _video_nodes():
    """Linux: узлы, отдающие видео, [{'index', 'name', 'path'}]. Узлы метаданных (у камеры их обычно два: видео и
    метаданные) пропускаются."""
    out = []
    for d in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        try:
            if (d / "index").read_text().strip() != "0":
                continue
            name = (d / "name").read_text().strip()
        except (OSError, ValueError):
            continue
        n = int(d.name[5:])
        out.append({"index": n, "name": name, "path": f"/dev/video{n}"})
    return out


def load_camera_map():
    """calib/camera_map.json (Linux): номер камеры проекта -> часть имени устройства, например {"0": "UC60"}.
    Нужен, потому что калибровки и фокус привязаны к номеру камеры (calib_cam0_*, camera_params_0.json), а номера
    /dev/videoN зависят от порядка подключения."""
    try:
        return {str(k): str(v) for k, v in json.loads(CAMERA_MAP_FILE.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {}


def resolve_device(index):
    """Номер камеры проекта -> номер устройства для OpenCV. Windows и номера вне camera_map - как есть."""
    if not IS_LINUX:
        return index
    want = load_camera_map().get(str(index))
    if want is None:
        return index
    for d in _video_nodes():
        if want.lower() in d["name"].lower():
            return d["index"]
    raise RuntimeError(f"Камера {index}: в {CAMERA_MAP_FILE.name} указано '{want}', но такого устройства нет. "
                       f"Сейчас видно: {', '.join(d['path'] + ' ' + d['name'] for d in _video_nodes()) or 'ничего'}")


def camera_devices():
    """Камеры системы: [{'index', 'name'}] (Linux - /dev/video<index>; Windows - порядок DirectShow, нужен pygrabber).
    Пусто - не определить (на Windows без pygrabber), тогда номера перебирают по очереди."""
    if IS_LINUX:
        return [{"index": d["index"], "name": d["name"]} for d in _video_nodes()]
    if IS_WINDOWS:
        try:
            from pygrabber.dshow_graph import FilterGraph
            return [{"index": i, "name": n} for i, n in enumerate(FilterGraph().get_input_devices())]
        except ImportError:
            pass
    return []


def camera_project_ids(name):
    """Номера камер проекта (camera_map.json), которым соответствует устройство с этим именем (только Linux)."""
    if not IS_LINUX:
        return []
    return [k for k, v in load_camera_map().items() if v.lower() in name.lower()]


def camera_open_hint():
    """Что добавить к ошибке "не удалось открыть камеру"."""
    if not IS_LINUX:
        return ""
    cams = _video_nodes()
    hint = (". Камеры: " + ", ".join(f"{d['index']} = {d['path']} ({d['name']})" for d in cams) if cams
            else ". Камер не видно (проверьте подключение и группу video: id)")
    return hint + "; свяжите номер камеры с устройством в calib/camera_map.json или передайте --camera <N>"


# ---------- режимы и регулировки камеры
def _v4l2_ctl(dev, *args):
    if shutil.which("v4l2-ctl") is None:
        raise RuntimeError("нужен v4l2-ctl (пакет v4l-utils)")
    return subprocess.run(["v4l2-ctl", "-d", f"/dev/video{dev}", *args], capture_output=True, text=True,
                          timeout=10).stdout


def camera_formats(index, raw_device=False):
    """Режимы камеры: [(fourcc, ширина, высота, макс. fps)]. index - номер камеры проекта (как --camera);
    raw_device=True - номер устройства системы без camera_map. RuntimeError - не получить (нет v4l2-ctl/pygrabber)."""
    if IS_WINDOWS:
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError as e:
            raise RuntimeError("нужен pygrabber (pip install pygrabber)") from e
        g = FilterGraph()
        g.add_video_input_device(index)
        try:
            return [(f["media_type_str"].upper(), f["width"], f["height"], max(f["min_framerate"], f["max_framerate"]))
                    for f in g.get_input_device().get_formats()]
        finally:
            g.remove_filters()
    dev = index if raw_device else resolve_device(index)
    fmt, size, res = None, None, {}
    for line in _v4l2_ctl(dev, "--list-formats-ext").splitlines():
        m = re.match(r"\s*\[\d+\]: '(\w{4})'", line)
        if m:
            fmt = m.group(1)
            continue
        m = re.match(r"\s*Size: Discrete (\d+)x(\d+)", line)
        if m:
            size = (int(m.group(1)), int(m.group(2)))
            res.setdefault((fmt, *size), 0.0)
            continue
        m = re.search(r"\(([\d.]+) fps\)", line)
        if m and fmt and size:
            res[(fmt, *size)] = max(res[(fmt, *size)], float(m.group(1)))
    return [(f, w, h, fps) for (f, w, h), fps in res.items()]


def control_ranges(index):
    """Диапазоны регулировок камеры {имя из scripts/camera_params.PROPS: (мин, макс)}: что камера не умеет - нет в
    словаре. None - таблица по умолчанию (Windows, где диапазоны не спросить, или нет v4l2-ctl)."""
    if not IS_LINUX:
        return None
    try:
        text = _v4l2_ctl(resolve_device(index), "--list-ctrls")
    except RuntimeError:
        return None
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\w+) 0x\w+ \((\w+)\)\s*:\s*(.*)", line)
        if not m:
            continue
        name, kind, rest = m.groups()
        lo, hi = re.search(r"min=(-?\d+)", rest), re.search(r"max=(-?\d+)", rest)
        if kind == "bool":
            found[name] = (0, 1)
        elif lo and hi:
            found[name] = (int(lo.group(1)), int(hi.group(1)))
    return {k: found[v] for k, v in _V4L2_CONTROLS.items() if v in found} or None


def toggle_auto_exposure(cur):
    """Значение CAP_PROP_AUTO_EXPOSURE, переключающее авто/ручную экспозицию (cur - текущее)."""
    if IS_LINUX:  # V4L2: 1 - ручная, 3 - автоматическая
        return 1.0 if cur == 3 else 3.0
    return 0.0 if cur else 1.0


# ---------- станок
def serial_ports():
    return sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*")) if not IS_WINDOWS else []


def default_serial_port():
    """Порт станка по умолчанию: COM4 на Windows; на Linux - первый /dev/ttyUSB* или /dev/ttyACM* (Arduino/CH340)."""
    if IS_WINDOWS:
        return "COM4"
    return (serial_ports() or ["/dev/ttyUSB0"])[0]


def serial_open_hint(err, port):
    """Что добавить к ошибке открытия порта."""
    if IS_WINDOWS:
        return ""
    text = str(err)
    if "ermission" in text:
        return " (нет прав на порт: добавьте себя в группу uucp/dialout и перелогиньтесь)"
    if "No such file" in text or "could not open" in text:
        return f" (порты сейчас: {', '.join(serial_ports()) or 'нет'})"
    return ""


# ---------- шрифты
def font_files():
    """Шрифты с кириллицей по убыванию предпочтения (первый найденный используется)."""
    yield from ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf")
    yield from ("DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
    try:  # DejaVu Sans лежит внутри matplotlib - есть везде, где он установлен
        import matplotlib
        yield str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    except ImportError:
        pass
