"""
Живой просмотр камеры с подстройкой параметров (фокус, экспозиция, баланс
белого и т.д.) через трекбары. Показывает резкость (дисперсию Лапласиана) в
центральной области кадра - по ней удобно искать оптимальный фокус.

Запуск:
    python scripts/camera_params.py --camera 0
    python scripts/camera_params.py --camera 1 --width 3264 --height 2448

Управление:
    a       - автоскан фокуса 0..1023 с шагом 32: печатает резкость для каждого
              значения и фиксирует найденный максимум (выключает автофокус)
    f       - переключить автофокус on/off
    e       - переключить авто/ручную экспозицию
    s       - сохранить текущие параметры в calib/camera_params_<index>.json
    l       - загрузить ранее сохранённые параметры
    q/ESC   - выход

Важно: поддержка свойств (focus/exposure/...) сильно зависит от драйвера
камеры и backend'а. Если трекбар визуально не влияет на картинку - скорее
всего эта property для вашей камеры под DSHOW не поддерживается. Диапазон
CAP_PROP_EXPOSURE и смысл CAP_PROP_AUTO_EXPOSURE (0/1 или 0.25/0.75) тоже
зависят от драйвера - подбирайте по факту.

Производительность: cap.set() на части драйверов стоит дорого (десятки-сотни
мс, переинициализация пайплайна). Поэтому каждое свойство отправляется в
камеру только тогда, когда вы реально подвинули соответствующий трекбар, а
не на каждом кадре - иначе и просадка fps, и борьба Focus с Autofocus (когда
обоим каждый кадр шлют противоречивые команды, камера дёргает линзу от края
до края).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.camera import open_camera, read_frame, grab_fresh, sharpness  # noqa: E402

# имя -> (cv2.CAP_PROP_*, минимум, максимум) диапазон для трекбара
PROPS = {
    "Focus":        (cv2.CAP_PROP_FOCUS, 0, 1023),
    "Autofocus":    (cv2.CAP_PROP_AUTOFOCUS, 0, 1),
    "Exposure":     (cv2.CAP_PROP_EXPOSURE, -13, 0),
    "AutoExposure": (cv2.CAP_PROP_AUTO_EXPOSURE, 0, 3),
    "Brightness":   (cv2.CAP_PROP_BRIGHTNESS, 0, 255),
    "Contrast":     (cv2.CAP_PROP_CONTRAST, 0, 255),
    "Saturation":   (cv2.CAP_PROP_SATURATION, 0, 255),
    "Gain":         (cv2.CAP_PROP_GAIN, 0, 255),
    "WB_Temp":      (cv2.CAP_PROP_WB_TEMPERATURE, 2000, 10000),
    "AutoWB":       (cv2.CAP_PROP_AUTO_WB, 0, 1),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def nothing(_):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="индекс камеры")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--roi", type=int, default=120, help="половина стороны центрального квадрата для резкости, px")
    ap.add_argument("--fourcc", type=str, default="",
                     help="'' (по умолчанию) - не трогать формат, камера отдаёт сырой поток как есть. "
                          "'MJPG' - принудительно сжатый формат, только если сырой не даёт нужный fps/разрешение "
                          "(хуже для субпиксельной точности). Точные форматы камеры: scripts/list_formats.py")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, fourcc=args.fourcc or None)

    win = f"camera_params [{args.camera}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for name, (_prop, lo, hi) in PROPS.items():
        cv2.createTrackbar(name, win, 0, hi - lo, nothing)

    cfg_path = Path(__file__).resolve().parent.parent / "calib" / f"camera_params_{args.camera}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    last_pos = {name: None for name in PROPS}

    def apply_changed():
        """Шлёт в камеру только те свойства, чей трекбар реально изменился
        с прошлого кадра - критично для fps и чтобы Focus не воевал с Autofocus."""
        for name, (prop, lo, hi) in PROPS.items():
            pos = cv2.getTrackbarPos(name, win)
            if pos != last_pos[name]:
                cap.set(prop, lo + pos)
                last_pos[name] = pos

    def sync_trackbars_from_camera():
        for name, (prop, lo, hi) in PROPS.items():
            val = cap.get(prop)
            pos = clamp(int(round(val - lo)), 0, hi - lo)
            cv2.setTrackbarPos(name, win, pos)
            last_pos[name] = pos

    sync_trackbars_from_camera()

    print(__doc__)

    while True:
        apply_changed()
        frame = read_frame(cap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        r = args.roi
        cx, cy = w // 2, h // 2
        roi = gray[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
        sharp = sharpness(roi)

        vis = frame.copy()
        cv2.rectangle(vis, (cx - r, cy - r), (cx + r, cy + r), (0, 255, 0), 1)
        cv2.putText(vis, f"sharpness(center)={sharp:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"focus={cap.get(cv2.CAP_PROP_FOCUS):.0f} autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS):.0f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('f'):
            cur = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0 if cur else 1.0)
            sync_trackbars_from_camera()
        elif key == ord('e'):
            cur = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0 if cur else 1.0)
            sync_trackbars_from_camera()
        elif key == ord('a'):
            print("Автоскан фокуса 0..1023 (шаг 32)...")
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            results = []
            for f in range(0, 1024, 32):
                cap.set(cv2.CAP_PROP_FOCUS, f)
                time.sleep(0.2)
                fr = grab_fresh(cap, flush=1)
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                roi2 = g[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
                s = sharpness(roi2)
                results.append((f, s))
                print(f"  focus={f:4d}  sharpness={s:9.1f}")
            best_focus, best_s = max(results, key=lambda t: t[1])
            cap.set(cv2.CAP_PROP_FOCUS, best_focus)
            sync_trackbars_from_camera()
            print(f"Лучший фокус: {best_focus} (sharpness={best_s:.1f}). Зафиксирован, автофокус выключен.")
        elif key == ord('s'):
            data = {name: cap.get(prop) for name, (prop, _, _) in PROPS.items()}
            cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"Сохранено в {cfg_path}")
        elif key == ord('l'):
            if cfg_path.exists():
                data = json.loads(cfg_path.read_text())
                for name, (prop, lo, hi) in PROPS.items():
                    if name in data:
                        cap.set(prop, data[name])
                sync_trackbars_from_camera()
                print(f"Загружено из {cfg_path}")
            else:
                print(f"Файл {cfg_path} не найден")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
