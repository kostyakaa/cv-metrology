"""
Перебирает индексы камер 0..N (Windows, DSHOW) или камеры из /dev/video* (Linux, V4L2), открывает каждую и
показывает живой кадр с подписью индекса/разрешения - чтобы понять, какой
индекс в системе соответствует какой физической камере (1080p / IMX179).

Запуск:
    python scripts/list_cameras.py [--max-index 6]

Управление:
    n       - следующая камера
    q/ESC   - выход полностью
"""
import argparse

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.system import CAMERA_BACKEND, camera_devices, camera_project_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-index", type=int, default=6, help="максимальный индекс камеры для проверки")
    args = ap.parse_args()

    found_any = False
    names = {d["index"]: d["name"] for d in camera_devices()}  # Linux: только узлы с видео, не метаданные
    for idx, name in names.items():
        ids = camera_project_ids(name)
        print(f"{idx}: {name}" + (f" = камера {', '.join(ids)} проекта" if ids else ""))
    indices = list(names) or range(args.max_index + 1)  # не определить - перебор номеров
    for idx in indices:
        cap = cv2.VideoCapture(idx, CAMERA_BACKEND)
        if not cap.isOpened():
            cap.release()
            continue

        found_any = True
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Камера {idx}{' (' + names[idx] + ')' if idx in names else ''}: {w:.0f}x{h:.0f}. 'n' - следующая, q - выход.")

        win = f"camera {idx}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        stop_all = False
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"  камера {idx}: не удалось прочитать кадр")
                break
            cv2.putText(frame, f"index={idx}  {w:.0f}x{h:.0f}  {names.get(idx, '')}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            if key in (27, ord('q')):
                stop_all = True
                break

        cap.release()
        cv2.destroyWindow(win)
        if stop_all:
            break

    if not found_any:
        print("Камеры не найдены ни по одному индексу.")


if __name__ == "__main__":
    main()
