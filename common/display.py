"""Показ кадра в исходном разрешении камеры с ползунком масштаба, подписи (с кириллицей), отрисовка меток."""
import time

import cv2
import numpy as np

ZOOM_MIN, ZOOM_MAX = 25, 300

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # без Pillow подписи рисуются cv2.putText (кириллица превратится в '?')
    Image = None

_FONT_FILES = ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf")
_fonts = {}


def add_zoom_trackbar(win, default=100):
    cv2.createTrackbar("zoom %", win, default - ZOOM_MIN, ZOOM_MAX - ZOOM_MIN, lambda v: None)


def get_zoom(win):
    return ZOOM_MIN + cv2.getTrackbarPos("zoom %", win)


def show_scaled(win, img, zoom_percent):
    w = max(int(img.shape[1] * zoom_percent / 100), 1)
    h = max(int(img.shape[0] * zoom_percent / 100), 1)
    interp = cv2.INTER_NEAREST if zoom_percent >= 100 else cv2.INTER_AREA
    disp = cv2.resize(img, (w, h), interpolation=interp)
    cv2.resizeWindow(win, w, h)
    cv2.imshow(win, disp)


def text_style(img):
    """Масштаб шрифта и толщина под разрешение кадра, чтобы подписи читались и на 8 МП."""
    scale = max(0.5, img.shape[1] / 1600)
    return scale, max(1, int(round(scale * 2)))


def text_px(img):
    """Высота шрифта подписей в пикселях под разрешение кадра."""
    return max(14, int(img.shape[1] / 45))


def _font(size):
    if size not in _fonts:
        for path in _FONT_FILES:
            try:
                _fonts[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def draw_texts(img, items):
    """items: [(текст, (x, y) левый верх, размер px, цвет BGR, фон BGR или None)] - рисует на img на месте.
    Все подписи за один проход через Pillow (кириллица, читаемо на любом фоне)."""
    if not items:
        return img
    if Image is None:
        for text, (x, y), size, color, _bg in items:
            cv2.putText(img, text, (int(x), int(y + size)), cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 1, cv2.LINE_AA)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, (x, y), size, color, bg in items:
        font = _font(int(size))
        x, y = int(x), int(y)
        if bg is not None:
            l, t, r, b = draw.textbbox((x, y), text, font=font)
            draw.rectangle((l - 3, t - 2, r + 3, b + 2), fill=bg[::-1])
        draw.text((x, y), text, font=font, fill=color[::-1])
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def draw_marker(img, corners, color, lost=False):
    """Контур метки (субпиксельно, сглаженный), точка на её левом верхнем углу, крест в центре.
    Возвращает точку для подписи над меткой."""
    th = max(1, int(round(img.shape[1] / 640)))
    pts = np.round(np.asarray(corners) * 16).astype(np.int32)
    if lost:  # пунктир - метки в этом кадре нет, показано последнее положение
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            for t0 in np.arange(0, 1, 0.1):
                p0 = (a + (b - a) * t0).astype(np.int32)
                p1 = (a + (b - a) * (t0 + 0.05)).astype(np.int32)
                cv2.line(img, tuple(p0), tuple(p1), color, th + 1, cv2.LINE_AA, shift=4)
    else:
        cv2.polylines(img, [pts], True, color, th + 1, cv2.LINE_AA, shift=4)
        cv2.circle(img, tuple(pts[0]), 4 * th * 16, color, -1, cv2.LINE_AA, shift=4)
    center = np.asarray(corners).mean(axis=0)
    size = int(np.linalg.norm(corners[1] - corners[0]) * 0.15) + 6
    cv2.drawMarker(img, tuple(int(v) for v in center), color, cv2.MARKER_CROSS, size, th, cv2.LINE_AA)
    x0, y0 = np.asarray(corners).min(axis=0)
    return int(x0), int(y0)


class MarkerHold:
    """Для окна: метка, пропавшая на долю секунды, рисуется последним положением (пунктиром), а не исчезает."""

    def __init__(self, hold=0.6):
        self.hold, self.last = hold, {}

    def update(self, found):
        """Возвращает {id: (углы, сколько секунд не видно)} для меток, пропавших не дольше hold."""
        now = time.time()
        for i, c in found.items():
            self.last[i] = (c, now)
        self.last = {i: v for i, v in self.last.items() if now - v[1] <= self.hold}
        return {i: (c, now - t) for i, (c, t) in self.last.items() if i not in found}
