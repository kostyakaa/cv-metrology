"""Программная фокусировка по максимуму резкости (дисперсия Лапласиана в заданной области кадра).

Поиск по сетке в три стадии: весь диапазон 0..1023 с шагом coarse_step, затем
+-coarse_step вокруг максимума с шагом fine_step, затем +-fine_step с шагом 1.
На каждой точке - медиана резкости по samples кадрам (против шума) после паузы
settle (механика объектива не мгновенная).
"""
import time

import cv2
import numpy as np

from common.camera import grab_fresh, sharpness


class FocusAborted(Exception):
    pass


def center_roi(frame, half=150):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    return max(cx - half, 0), max(cy - half, 0), min(cx + half, w), min(cy + half, h)


def roi_around(points, frame, pad=0.25, min_half=40):
    """Прямоугольник вокруг точек (например углов метки) с запасом pad от размера."""
    h, w = frame.shape[:2]
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    px = max((x1 - x0) * pad, min_half)
    py = max((y1 - y0) * pad, min_half)
    return (int(max(x0 - px, 0)), int(max(y0 - py, 0)), int(min(x1 + px, w)), int(min(y1 + py, h)))


def measure(cap, roi, samples=3, settle=0.2):
    time.sleep(settle)
    x0, y0, x1, y1 = roi
    vals, frame = [], None
    for _ in range(samples):
        frame = grab_fresh(cap, flush=1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(sharpness(gray[y0:y1, x0:x1]))
    return float(np.median(vals)), frame


def autofocus(cap, roi, coarse_step=32, fine_step=4, samples=3, settle=0.2, on_step=None):
    """Возвращает (лучший фокус, его резкость, все замеры [(фокус, резкость)]).
    on_step(frame, focus, sharpness) вызывается на каждой точке; если вернёт True - FocusAborted."""
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    results = []

    def scan(values):
        out = []
        for f in values:
            cap.set(cv2.CAP_PROP_FOCUS, int(f))
            s, frame = measure(cap, roi, samples, settle)
            out.append((int(f), s))
            if on_step is not None and on_step(frame, int(f), s):
                raise FocusAborted()
        results.extend(out)
        return max(out, key=lambda t: t[1])

    best_f, _ = scan(range(0, 1024, coarse_step))
    coarse = [s for _, s in results]
    if np.mean(coarse) > 1e-6 and (max(coarse) - min(coarse)) / np.mean(coarse) < 0.05:
        print("ВНИМАНИЕ: резкость почти не меняется по всему диапазону фокуса (<5%) - "
              "возможно, CAP_PROP_FOCUS не поддерживается этой камерой/драйвером.")
    best_f, _ = scan(range(max(best_f - coarse_step, 0), min(best_f + coarse_step, 1023) + 1, max(fine_step, 1)))
    best_f, best_s = scan(range(max(best_f - fine_step, 0), min(best_f + fine_step, 1023) + 1))

    cap.set(cv2.CAP_PROP_FOCUS, best_f)
    time.sleep(settle)
    return best_f, best_s, results
