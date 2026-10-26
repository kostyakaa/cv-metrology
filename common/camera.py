"""Общие утилиты для работы с камерой через OpenCV (Windows, backend DSHOW)."""
import json
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def fourcc_to_str(value):
    v = int(value)
    chars = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return chars.strip().strip("\x00") or "?"


def params_path(index):
    return ROOT / "calib" / f"camera_params_{index}.json"


def load_saved_focus(index):
    """Фокус из calib/camera_params_<index>.json, если он там зафиксирован (Autofocus=0)."""
    path = params_path(index)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("Autofocus", 1) == 0 and data.get("Focus") is not None:
        return int(round(data["Focus"]))
    return None


def save_focus(index, focus):
    """Дописывает фокус в calib/camera_params_<index>.json (остальные параметры не трогает)."""
    path = params_path(index)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update({"Focus": float(focus), "Autofocus": 0})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def set_focus(cap, focus, settle=0.3):
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, focus)
    time.sleep(settle)


def open_camera(index, width=None, height=None, fourcc=None, backend=cv2.CAP_DSHOW, verbose=True, saved_focus=True):
    """fourcc=None по умолчанию - НЕ трогаем формат, камера отдаёт тот поток,
    что выставлен по умолчанию драйвером (почти всегда сырой YUY2/NV12, этого
    и хотим для точной детекции по пикселям - без JPEG-артефактов). Передайте
    fourcc="MJPG" явно только если сырой поток не укладывается в полосу USB
    на нужном разрешении/fps - это сознательный обмен точности на скорость.
    Точный список форматов конкретной камеры - scripts/list_formats.py
    (cv2.get(CAP_PROP_FOURCC) под DSHOW часто врёт, поэтому не полагаемся на него).

    saved_focus=True: драйвер при каждом открытии камеры сбрасывает фокус/автофокус,
    поэтому сохранённый фокус (autofocus.py, detect_markers.py, survey_markers.py)
    выставляется заново при каждом открытии."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {index}")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # не на всех backend'ах поддерживается, но не вредит

    focus = load_saved_focus(index) if saved_focus else None
    if focus is not None:
        cap.read()  # часть драйверов принимает настройки фокуса только после запуска потока
        set_focus(cap, focus)

    if verbose:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        reported = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        print(f"Камера {index}: {w}x{h} @ {fps:.1f} fps, cv2 сообщает формат={reported} "
              f"(под DSHOW это поле не всегда точное - точный список: scripts/list_formats.py)")
        if focus is not None:
            print(f"Камера {index}: фокус {focus} из {params_path(index).name} (автофокус выключен)")

    return cap


def read_frame(cap):
    """Быстрое чтение для живого просмотра - НЕ сбрасывает буфер. Используйте
    это в циклах предпросмотра, иначе fps проседает в разы без всякой пользы."""
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать кадр с камеры")
    return frame


def grab_fresh(cap, flush=4):
    """Сбросить старые кадры из буфера драйвера и вернуть свежий кадр.
    Нужно только сразу после физического перемещения станка/камеры (когда в
    буфере может лежать кадр со старой позиции) - НЕ используйте в цикле
    живого просмотра, это просто выбрасывает нормальные кадры и роняет fps."""
    for _ in range(flush):
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать кадр с камеры")
    return frame


def sharpness(gray):
    """Дисперсия лапласиана - метрика резкости, растёт при попадании в фокус."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()
