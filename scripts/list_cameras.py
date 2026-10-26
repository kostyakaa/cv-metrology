"""
Перебирает индексы камер 0..N, открывает каждую по очереди (backend DSHOW) и
показывает живой кадр с подписью индекса/разрешения - чтобы понять, какой
индекс в системе соответствует какой физической камере (1080p / IMX179).

Запуск:
    python scripts/list_cameras.py [--max-index 6]

Управление:
    n       - следующая камера
    q/ESC   - выход полностью
"""
import argparse

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-index", type=int, default=6, help="максимальный индекс камеры для проверки")
    args = ap.parse_args()

    found_any = False
    for idx in range(args.max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        found_any = True
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Камера {idx}: {w:.0f}x{h:.0f}. 'n' - следующая, q - выход.")

        win = f"camera {idx}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        stop_all = False
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"  камера {idx}: не удалось прочитать кадр")
                break
            cv2.putText(frame, f"index={idx}  {w:.0f}x{h:.0f}", (10, 30),
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
