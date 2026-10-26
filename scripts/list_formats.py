"""
Точный список форматов, которые реально поддерживает камера - разрешение,
fps и тип потока: сырой (YUY2/NV12/RGB24/I420/...) или сжатый (MJPG/H264/...).

Через OpenCV это надёжно не узнать: cv2.get(CAP_PROP_FOURCC) под backend'ом
DSHOW часто возвращает 0 или не то, что камера реально отдаёт, а cv2.set()
может молча не подействовать. Здесь форматы читаются напрямую из DirectShow
media types (через pygrabber) - то же, что показывает диалог свойств камеры.

Установка (однократно):
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

from pygrabber.dshow_graph import FilterGraph

RAW_FORMATS = {"YUY2", "NV12", "I420", "YV12", "RGB24", "RGB32", "UYVY", "Y800", "GREY", "Y8"}


def describe(media_type_str):
    tag = "СЫРОЙ" if media_type_str.upper() in RAW_FORMATS else "сжатый"
    return f"{media_type_str} ({tag})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None, help="индекс камеры из списка; без аргумента - все камеры")
    args = ap.parse_args()

    graph = FilterGraph()
    devices = graph.get_input_devices()
    if not devices:
        print("DirectShow не видит ни одной камеры.")
        return

    for idx, name in enumerate(devices):
        if args.device is not None and idx != args.device:
            continue
        print(f"[{idx}] {name}")
        g = FilterGraph()
        try:
            g.add_video_input_device(idx)
            vi = g.get_input_device()
            cur_w, cur_h = vi.get_current_format()
            print(f"    сейчас выставлено: {cur_w}x{cur_h}")
            for f in vi.get_formats():
                fps = max(f["min_framerate"], f["max_framerate"])
                print(f"    [{f['index']:2d}] {f['width']:5d}x{f['height']:<5d}  "
                      f"{describe(f['media_type_str']):<16s}  до ~{fps:.0f} fps")
        except Exception as e:
            print(f"    не удалось получить форматы: {e}")
        finally:
            g.remove_filters()


if __name__ == "__main__":
    main()
