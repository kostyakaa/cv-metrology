"""
Детектор простой контрастной точки на живом видео, вручную (без станка) -
проверить, что в принципе можно устойчиво находить метку/цель на кадре, и
увидеть, на сколько пикселей её нужно сдвинуть до центра кадра (прообраз
финального шага визуального сервоинга).

Режимы (--mode):
  blob      (по умолчанию) - CLAHE (локальный контраст) + бинаризация
            (адаптивный порог / Otsu / ручной) + поиск контуров, фильтр по
            площади и круглости, центр по моментам (субпиксельно).
            Адаптивный порог и CLAHE особенно помогают, когда камера плохо
            различает контраст (неровный свет, слабая метка).
  template  - выделите точку мышью (ЛКМ, протяните прямоугольник вокруг
            метки), дальше она ищется через cv2.matchTemplate с
            параболическим субпикселем в окне вокруг последней найденной
            позиции - прообраз будущего визуального сервоинга.

Окно показывается в исходном разрешении камеры (трекбар "zoom %" = 100% по
умолчанию), трекбаром можно уменьшить/увеличить окно просмотра - это только
масштаб отображения, детекция всегда считается по кадру в полном разрешении.

В обоих режимах рядом с найденной точкой показывается стрелка и dx/dy в px
до геометрического центра кадра - именно на столько пикселей нужно сдвинуть
картинку (камеру), чтобы точка оказалась точно в центре.

Запуск:
    python scripts/detect_point.py --camera 0
    python scripts/detect_point.py --camera 1 --mode template

Управление:
  blob:      трекбары method (0 адаптивный/1 Otsu/2 ручной), thresh, block,
             C, invert, clahe, zoom %
  template:  ЛКМ + drag - выбрать шаблон, r - сбросить выбор, трекбар zoom %
  общее:     q/ESC - выход
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.camera import open_camera, read_frame  # noqa: E402
from common.display import add_zoom_trackbar, get_zoom, show_scaled  # noqa: E402


def draw_center_offset(vis, point, frame_w, frame_h):
    """Стрелка и текст: на сколько px сдвинуть картинку, чтобы point оказался в центре кадра."""
    center = (frame_w // 2, frame_h // 2)
    dx = center[0] - point[0]
    dy = center[1] - point[1]
    p = (int(round(point[0])), int(round(point[1])))
    cv2.drawMarker(vis, center, (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
    cv2.arrowedLine(vis, p, center, (0, 165, 255), 2, tipLength=0.2)
    cv2.putText(vis, f"center offset: dx={dx:+.1f} dy={dy:+.1f} px",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return dx, dy


def run_blob(cap, args):
    win = "detect_point [blob]"
    win_mask = "detect_point [mask]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_mask, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("method", win, 0, 2, lambda v: None)   # 0=adaptive 1=otsu 2=manual
    cv2.createTrackbar("thresh", win, 127, 255, lambda v: None)
    cv2.createTrackbar("block", win, 24, 99, lambda v: None)  # block = 2*pos+3 (нечётный, >=3)
    cv2.createTrackbar("C", win, 20, 40, lambda v: None)      # C = pos-20
    cv2.createTrackbar("invert", win, 0, 1, lambda v: None)
    cv2.createTrackbar("clahe", win, 1, 1, lambda v: None)
    add_zoom_trackbar(win)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    print("method: 0-адаптивный (по умолчанию, лучше при плохом контрасте) / 1-Otsu / 2-ручной (thresh)")
    print("q/ESC - выход")

    while True:
        frame = read_frame(cap)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cv2.getTrackbarPos("clahe", win):
            gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        method = cv2.getTrackbarPos("method", win)
        invert = cv2.getTrackbarPos("invert", win)
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        if method == 0:  # адаптивный - устойчив к неровному освещению/слабому контрасту
            block = 2 * cv2.getTrackbarPos("block", win) + 3
            c_val = cv2.getTrackbarPos("C", win) - 20
            mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresh_type, block, c_val)
        elif method == 1:  # Otsu
            t, mask = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
            cv2.setTrackbarPos("thresh", win, int(t))
        else:  # ручной
            t = cv2.getTrackbarPos("thresh", win)
            _, mask = cv2.threshold(blur, t, 255, thresh_type)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < args.min_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < args.min_circularity:
                continue
            candidates.append((c, area, circularity))

        if candidates:
            c, area, circularity = max(candidates, key=lambda t: t[1])
            M = cv2.moments(c)
            if M["m00"] > 0:
                px = M["m10"] / M["m00"]
                py = M["m01"] / M["m00"]
                cv2.drawContours(vis, [c], -1, (0, 255, 0), 1)
                cv2.drawMarker(vis, (int(round(px)), int(round(py))), (0, 0, 255),
                                cv2.MARKER_CROSS, 16, 2)
                cv2.putText(vis, f"({px:.2f}, {py:.2f}) px  area={area:.0f}  circ={circularity:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                draw_center_offset(vis, (px, py), w, h)
        else:
            cv2.putText(vis, "точка не найдена - подкрутите method/thresh/block/C/clahe",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        zoom = get_zoom(win)
        show_scaled(win, vis, zoom)
        show_scaled(win_mask, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), zoom)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_template(cap, args):
    win = "detect_point [template]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    add_zoom_trackbar(win)

    state = {"drag_start": None, "box": None, "last_box": None}

    def on_mouse(event, x, y, flags, _):
        z = get_zoom(win) / 100.0
        x, y = x / z, y / z  # координаты окна -> координаты полного кадра
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (x, y)
            state["box"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"]:
            x0, y0 = state["drag_start"]
            state["box"] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"]:
            box = state["box"]
            state["drag_start"] = None
            if box and box[2] - box[0] > 4 and box[3] - box[1] > 4:
                state["last_box"] = tuple(int(round(v)) for v in box)
            state["box"] = None

    cv2.setMouseCallback(win, on_mouse)
    print("Протяните прямоугольник ЛКМ вокруг контрастной точки, r - сброс, q/ESC - выход")

    template = None
    tpl_size = None
    last_pos = None

    while True:
        frame = read_frame(cap)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        if state["drag_start"] and state["box"]:
            b = tuple(int(round(v)) for v in state["box"])
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)

        if state["last_box"] is not None and template is None:
            x0, y0, x1, y1 = state["last_box"]
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w), min(y1, h)
            template = gray[y0:y1, x0:x1].copy()
            tpl_size = template.shape[::-1]  # (w, h)
            last_pos = ((x0 + x1) / 2, (y0 + y1) / 2)
            state["last_box"] = None
            print(f"Шаблон выбран: {tpl_size[0]}x{tpl_size[1]} px")

        if template is not None:
            r = args.search_radius
            tw, th = tpl_size
            cx, cy = int(last_pos[0]), int(last_pos[1])
            x0 = max(cx - tw // 2 - r, 0)
            y0 = max(cy - th // 2 - r, 0)
            x1 = min(cx + tw // 2 + r, w)
            y1 = min(cy + th // 2 + r, h)
            search = gray[y0:y1, x0:x1]

            if search.shape[0] >= th and search.shape[1] >= tw:
                res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                _, maxval, _, maxloc = cv2.minMaxLoc(res)
                mx, my = maxloc

                def parab(m, c, p):
                    denom = m - 2 * c + p
                    return 0.5 * (m - p) / denom if abs(denom) > 1e-9 else 0.0

                dxp = parab(res[my, mx - 1], res[my, mx], res[my, mx + 1]) if 0 < mx < res.shape[1] - 1 else 0.0
                dyp = parab(res[my - 1, mx], res[my, mx], res[my + 1, mx]) if 0 < my < res.shape[0] - 1 else 0.0

                px = x0 + mx + dxp + tw / 2
                py = y0 + my + dyp + th / 2
                last_pos = (px, py)

                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 1)
                cv2.drawMarker(vis, (int(round(px)), int(round(py))), (0, 0, 255),
                               cv2.MARKER_CROSS, 16, 2)
                cv2.putText(vis, f"({px:.2f}, {py:.2f}) px  score={maxval:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                draw_center_offset(vis, (px, py), w, h)
            else:
                cv2.putText(vis, "цель у края кадра, потеряна", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        show_scaled(win, vis, get_zoom(win))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r'):
            template = None
            last_pos = None
            print("Шаблон сброшен")

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--mode", choices=["blob", "template"], default="blob")
    ap.add_argument("--min-area", type=float, default=20.0, help="минимальная площадь контура, px^2 (режим blob)")
    ap.add_argument("--min-circularity", type=float, default=0.4,
                     help="минимальная круглость контура 0..1 (режим blob); 1.0 = идеальный круг")
    ap.add_argument("--search-radius", type=int, default=40, help="запас поиска вокруг шаблона, px (режим template)")
    ap.add_argument("--fourcc", type=str, default="",
                     help="'' (по умолчанию) - не трогать формат, камера отдаёт сырой поток как есть. "
                          "'MJPG' - принудительно сжатый формат, только если сырой не даёт нужный fps/разрешение "
                          "(хуже для субпиксельной точности). Точные форматы камеры: scripts/list_formats.py")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, fourcc=args.fourcc or None)
    if args.mode == "blob":
        run_blob(cap, args)
    else:
        run_template(cap, args)


if __name__ == "__main__":
    main()
