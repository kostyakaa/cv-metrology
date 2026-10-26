"""
Точная программная автофокусировка по максимуму резкости (дисперсия
Лапласиана в центральной области кадра), алгоритм - common/focus.py
(сетка в три стадии: шаг 32 по всему диапазону -> шаг 4 -> шаг 1).

Найденный фокус СРАЗУ сохраняется в calib/camera_params_<index>.json, и все
остальные скрипты (detect_markers, survey_markers, calibrate_distortion, ...)
выставляют его при открытии камеры - драйвер сбрасывает фокус при каждом
переподключении, поэтому сохранять и выставлять заново нужно обязательно.

Для меток удобнее фокусироваться прямо в detect_markers.py (клавиша f) - там
резкость считается по самой метке.

Запуск:
    python scripts/autofocus.py --camera 1
    python scripts/autofocus.py --camera 1 --roi 150 --coarse-step 16 --samples 5

Управление в окне просмотра:
    r         - повторить автофокусировку с нуля
    . / ,     - подвинуть фокус вручную +1 / -1
    > / <     - подвинуть фокус вручную +10 / -10
    s         - сохранить текущий (подвинутый вручную) фокус
    q/ESC     - выход
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.camera import open_camera, read_frame, save_focus, sharpness  # noqa: E402
from common.focus import autofocus, center_roi  # noqa: E402


def render_curve(all_results, chosen_focus, width=640, height=220):
    img = np.full((height, width, 3), 255, np.uint8)
    if not all_results:
        return img

    by_focus = {}
    for f, s in all_results:
        by_focus[f] = max(by_focus.get(f, 0.0), s)
    pts = sorted(by_focus.items())
    ys = [s for _, s in pts]
    ymax = max(ys) * 1.05 + 1e-6

    def to_xy(f, s):
        x = int(f / 1023 * (width - 1))
        y = int((height - 25) - (s / ymax) * (height - 45))
        return x, y

    for i in range(1, len(pts)):
        cv2.line(img, to_xy(*pts[i - 1]), to_xy(*pts[i]), (160, 160, 160), 1)
    for f, s in pts:
        cv2.circle(img, to_xy(f, s), 2, (0, 0, 200), -1)

    chosen_s = by_focus.get(chosen_focus, max(ys))
    cv2.drawMarker(img, to_xy(chosen_focus, chosen_s), (0, 150, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.putText(img, f"focus={chosen_focus}  sharpness={chosen_s:.1f}",
                (8, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--roi", type=int, default=150, help="половина стороны центрального квадрата для метрики резкости, px")
    ap.add_argument("--coarse-step", type=int, default=32)
    ap.add_argument("--fine-step", type=int, default=4)
    ap.add_argument("--samples", type=int, default=3, help="кадров на точку (медиана), против шума")
    ap.add_argument("--settle", type=float, default=0.2, help="пауза после установки фокуса, сек")
    ap.add_argument("--fourcc", type=str, default="", help="'' - сырой поток как есть; 'MJPG' - сжатый")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, fourcc=args.fourcc or None)
    roi = center_roi(read_frame(cap), args.roi)
    print(__doc__)

    def run():
        print("Автофокус по центру кадра...")
        f, s, res = autofocus(cap, roi, args.coarse_step, args.fine_step, args.samples, args.settle,
                              on_step=lambda frame, fv, sv: print(f"  focus={fv:4d}  sharpness={sv:9.1f}"))
        print(f"ИТОГ: focus={f}  sharpness={s:.1f}. Сохранено: {save_focus(args.camera, f)}")
        return f, res

    focus, all_results = run()
    win = f"autofocus [{args.camera}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    x0, y0, x1, y1 = roi

    while True:
        frame = read_frame(cap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s_now = sharpness(gray[y0:y1, x0:x1])

        vis = frame.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(vis, f"focus={focus}  sharpness={s_now:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        curve = render_curve(all_results, focus)
        curve = cv2.resize(curve, (vis.shape[1], int(curve.shape[0] * vis.shape[1] / curve.shape[1])))
        cv2.imshow(win, np.vstack([vis, curve]))

        key = cv2.waitKey(1) & 0xFF
        step = {ord('.'): 1, ord(','): -1, ord('>'): 10, ord('<'): -10}.get(key)
        if key in (27, ord('q')):
            break
        elif key == ord('r'):
            focus, all_results = run()
        elif step is not None:
            focus = int(np.clip(focus + step, 0, 1023))
            cap.set(cv2.CAP_PROP_FOCUS, focus)
        elif key == ord('s'):
            print(f"Фокус {focus} сохранён: {save_focus(args.camera, focus)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
