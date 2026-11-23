"""
Точный список форматов, которые реально поддерживает камера - разрешение,
fps и тип потока: сырой (YUY2/NV12/RGB24/I420/...) или сжатый (MJPG/H264/...).

На Linux форматы берутся из V4L2 (v4l2-ctl, пакет v4l-utils): список камер и режимов
(YUYV - сырой, MJPG - сжатый, с максимальным fps), номер камеры - как в --camera.

Через OpenCV это надёжно не узнать: cv2.get(CAP_PROP_FOURCC) под backend'ом
DSHOW часто возвращает 0 или не то, что камера реально отдаёт, а cv2.set()
может молча не подействовать. Здесь форматы читаются напрямую из DirectShow
media types (через pygrabber) - то же, что показывает диалог свойств камеры.

Установка (однократно, только Windows):
    pip install pygrabber

Запуск:
    python scripts/list_formats.py              - все камеры и все их форматы
    python scripts/list_formats.py --device 0   - только одна камера (индекс как в списке ниже)

Почему это важно: для детекции по пикселям (центроид/matchTemplate/субпиксель)
нужен СЫРОЙ поток (YUY2, NV12, RGB24, I420...) - в нём нет блочных артефактов.
MJPG/H264 - это сжатие внутри каждого кадра, оно квантует и слегка "размазывает"
границы, что может сбивать субпиксельную точность. Берите сырой формат на
нужном разрешении; на сжатый переходите только если сырой не даёт нужного
fps из-за полосы USB (и тогда это осознанный компромисс точность/скорость).
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.system import camera_devices, camera_formats, camera_project_ids  # noqa: E402

RAW_FORMATS = {"YUYV", "YUY2", "NV12", "I420", "YV12", "RGB24", "RGB32", "UYVY", "Y800", "GREY", "Y8"}


def describe(media_type_str):
    tag = "СЫРОЙ" if media_type_str.upper() in RAW_FORMATS else "сжатый"
    return f"{media_type_str} ({tag})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None,
                    help="номер камеры из списка или номер камеры проекта; без аргумента - все камеры")
    args = ap.parse_args()

    cams = camera_devices()
    if not cams:
        print("Камеры не найдены (Windows: нужен pygrabber; Linux: /dev/video*, группа video).")
        return
    for d in cams:
        ids = camera_project_ids(d["name"])
        if args.device is not None and args.device not in (d["index"], *map(int, ids)):
            continue
        print(f"[{d['index']}] {d['name']}" + (f"  (камера {', '.join(ids)} проекта)" if ids else ""))
        try:
            for fourcc, w, h, fps in sorted(camera_formats(d["index"], raw_device=True), key=lambda f: (f[0], f[1] * f[2])):
                print(f"    {w:5d}x{h:<5d}  {describe(fourcc):<16s}  до ~{fps:.0f} fps")
        except Exception as e:
            print(f"    не удалось получить форматы: {e}")


if __name__ == "__main__":
    main()
