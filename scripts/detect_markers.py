"""
Живая детекция печатных меток ArUco (из scripts/make_markers.py) - 4 угла
стола. Для каждой найденной метки: ID, к какому углу стола она относится,
центр и 4 угла в пикселях (субпиксельно). Если видны все ожидаемые метки -
рисуется контур стола по их центрам.

Фокус: при открытии камеры выставляется сохранённый (calib/camera_params_<N>.json).
Клавиша f - автофокус по самой метке (резкость считается в области метки,
если её не видно - по центру кадра); результат сохраняется и подхватывается
всеми остальными скриптами.

Привязка меток к координатам станка - scripts/survey_markers.py.

Запуск:
    python scripts/detect_markers.py --camera 0
    python scripts/detect_markers.py --camera 1 --width 1280 --height 720

Управление:
    f       - автофокус по метке и сохранение фокуса (ESC - прервать)
    p       - напечатать координаты всех найденных меток в консоль
    s       - сохранить их в calib/markers_px_cam<index>.json (с размером кадра)
    j       - тест дрожания: копит --samples кадров при неподвижной камере и
              печатает std центров меток в px (чем меньше, тем стабильнее детекция)
    q/ESC   - выход
    трекбар zoom % - масштаб окна (детекция всегда по полному кадру)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.camera import load_saved_focus, open_camera, read_frame, save_focus  # noqa: E402
from common.display import (MarkerHold, add_zoom_trackbar, draw_marker, draw_texts, get_zoom,  # noqa: E402
                            show_scaled, text_px)
from common.focus import FocusAborted, autofocus, center_roi, roi_around  # noqa: E402
from common.markers import CORNER_LABELS, detect, detect_precise, make_detector  # noqa: E402


def describe(found):
    lines = []
    for marker_id in sorted(found):
        c = found[marker_id]
        center = c.mean(axis=0)
        corners = ", ".join(f"({x:.2f},{y:.2f})" for x, y in c)
        lines.append(f"ID {marker_id} [{CORNER_LABELS.get(marker_id, 'доп.')}]: "
                     f"центр=({center[0]:.2f},{center[1]:.2f})  углы={corners}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fourcc", type=str, default="", help="'' - сырой поток как есть; 'MJPG' - сжатый")
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3], help="ожидаемые ID меток (в порядке обхода контура)")
    ap.add_argument("--samples", type=int, default=30, help="кадров для теста дрожания (j)")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, fourcc=args.fourcc or None)
    detector = make_detector()
    detector_precise = make_detector(precise=True)
    hold = MarkerHold()

    win = f"detect_markers [{args.camera}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    add_zoom_trackbar(win)
    print(__doc__)

    jitter = None  # {id: [центры]} во время теста дрожания
    out_path = ROOT / "calib" / f"markers_px_cam{args.camera}.json"
    focus = load_saved_focus(args.camera)
    if focus is None:
        print("Фокус не сохранён - наведите камеру на метку и нажмите f")

    def focus_tick(frame, f, s):
        vis = frame.copy()
        draw_texts(vis, [(f"автофокус: фокус {f}, резкость {s:.0f}   (ESC - прервать)", (8, 6), text_px(vis),
                          (255, 255, 255), (0, 100, 200))])
        show_scaled(win, vis, get_zoom(win))
        return (cv2.waitKey(1) & 0xFF) == 27

    while True:
        frame = read_frame(cap)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = detect_precise(detector, detector_precise, gray) if jitter is not None else detect(detector, gray)

        vis = frame.copy()
        size = text_px(vis)
        texts = []
        for marker_id, c in found.items():
            x, y = draw_marker(vis, c, (0, 220, 0))
            center = c.mean(axis=0)
            texts.append((f"ID{marker_id} {CORNER_LABELS.get(marker_id, '')}  ({center[0]:.1f}, {center[1]:.1f})",
                          (x, max(0, y - size - 10)), size, (255, 255, 255), (0, 110, 0)))
        for marker_id, (c, ago) in hold.update(found).items():
            x, y = draw_marker(vis, c, (0, 150, 255), lost=True)
            texts.append((f"ID{marker_id}: нет в кадре {ago:.1f} с", (x, max(0, y - size - 10)), size,
                          (255, 255, 255), (0, 100, 200)))

        missing = [i for i in args.ids if i not in found]
        if not missing:
            poly = np.array([found[i].mean(axis=0) for i in args.ids], np.int32)
            cv2.polylines(vis, [poly], True, (255, 0, 255), max(1, w // 640), cv2.LINE_AA)
        status = f"найдено {len(args.ids) - len(missing)}/{len(args.ids)}"
        if missing:
            status += f", нет: {missing}"
        status += f"   фокус: {focus if focus is not None else 'авто - нажмите f!'}"
        texts.append((status, (8, 6), size, (255, 255, 255), (0, 90, 0) if not missing else (0, 0, 150)))

        if jitter is not None:
            for marker_id, c in found.items():
                jitter.setdefault(marker_id, []).append(c.mean(axis=0))
            n = max((len(v) for v in jitter.values()), default=0)
            texts.append((f"тест дрожания (точный детектор): {n}/{args.samples}", (8, size + 16), size,
                          (255, 255, 255), (0, 100, 200)))
            if n >= args.samples:
                print(f"Дрожание центров меток за {args.samples} кадров, точный детектор (std, px):")
                for marker_id in sorted(jitter):
                    pts = np.array(jitter[marker_id])
                    sx, sy = pts.std(axis=0)
                    print(f"  ID {marker_id}: std x={sx:.4f} y={sy:.4f}  (кадров с меткой: {len(pts)})")
                jitter = None
        draw_texts(vis, texts)

        show_scaled(win, vis, get_zoom(win))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('f'):
            if found:
                target = np.array([w / 2, h / 2])
                nearest = min(found, key=lambda i: np.linalg.norm(found[i].mean(axis=0) - target))
                roi = roi_around(found[nearest], frame)
                print(f"Автофокус по метке ID{nearest}...")
            else:
                roi = center_roi(frame)
                print("Меток нет - автофокус по центру кадра...")
            try:
                focus, s, _ = autofocus(cap, roi, samples=2, settle=0.15, on_step=focus_tick)
                print(f"Фокус {focus} (резкость {s:.0f}) сохранён: {save_focus(args.camera, focus)}")
            except FocusAborted:
                print("Автофокус прерван - фокус мог остаться не на месте, повторите f")
        elif key == ord('p'):
            print("\n".join(describe(found)) or "Метки не найдены")
        elif key == ord('j'):
            jitter = {}
            print(f"Тест дрожания: держите камеру и метки неподвижно, {args.samples} кадров...")
        elif key == ord('s'):
            found = detect_precise(detector, detector_precise, gray)
            data = {
                "frame_size": [w, h],
                "markers": {str(i): {"center": found[i].mean(axis=0).tolist(), "corners": found[i].tolist(),
                                     "corner_label": CORNER_LABELS.get(i)} for i in sorted(found)},
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Сохранено {len(found)} меток в {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
