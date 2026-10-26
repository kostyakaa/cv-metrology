"""
Генерирует лист печатных меток ArUco для углов стола - SVG в точных
миллиметрах (печатать в масштабе 100%).

Метки из словаря DICT_4X4_50: сетка 6x6 "пикселей"-модулей (4x4 бита данных +
чёрная рамка). При размере 15 мм модуль = 2.5 мм - крупно, чётко печатается
лазерником и уверенно читается камерой. ID зашит в узор, так что метки не
перепутать, и по нему же известна ориентация метки. Каждая метка даёт 4 угла
с субпиксельной точностью - 4 метки = 16 опорных точек.

Назначение ID (так их ждёт scripts/detect_markers.py):
    ID 0 - угол стола X0 Y0
    ID 1 - угол Xmax Y0
    ID 2 - угол Xmax Ymax
    ID 3 - угол X0 Ymax
Клеить все метки одинаково: верхней стороной (где надпись на листе) к +Y.

Запуск:
    python scripts/make_markers.py                          -> print/aruco_markers_15mm.svg
    python scripts/make_markers.py --size 20 --ids 0 1 2 3 4 5

Печать: открыть SVG в Chrome/Edge -> Ctrl+P -> масштаб 100% (НЕ "по размеру
страницы"). После печати проверьте линейкой контрольную линию 100 мм. Вырезать
по пунктиру: белое поле вокруг метки (по умолчанию 5 мм) обязательно - без него
детектор хуже отделяет чёрную рамку от фона. Матовая бумага, жёсткое ровное
основание, без бликов.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.markers import CORNER_LABELS, get_dictionary  # noqa: E402

PAGE_MARGIN = 8.0
GAP = 15.0
LABEL_H = 14.0
RULER_MM = 100


def marker_modules(dictionary, marker_id):
    n = dictionary.markerSize + 2  # биты + рамка в 1 модуль с каждой стороны
    return cv2.aruco.generateImageMarker(dictionary, marker_id, n) == 0  # True = чёрный


def module_path(modules, x0, y0, m):
    """Один <path> на метку: соседние модули без швов антиалиасинга при печати."""
    parts = []
    for r, c in zip(*np.nonzero(modules)):
        parts.append(f"M{x0 + c * m:.4f},{y0 + r * m:.4f}h{m:.4f}v{m:.4f}h{-m:.4f}z")
    return f'<path d="{"".join(parts)}" fill="black"/>'


def build(ids, size, margin):
    dictionary = get_dictionary()
    cut = size + 2 * margin
    cols = 2 if len(ids) <= 4 else 3
    rows = int(np.ceil(len(ids) / cols))
    cell_w, cell_h = cut + GAP, cut + LABEL_H
    grid_w = cols * cell_w - GAP
    width = max(grid_w, RULER_MM + 10) + 2 * PAGE_MARGIN
    title_h = 10.0
    ruler_y = PAGE_MARGIN + title_h + rows * cell_h + 6
    notes_y = ruler_y + 16
    height = notes_y + 3 * 5 + PAGE_MARGIN

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}mm" height="{height:.2f}mm" '
        f'viewBox="0 0 {width:.4f} {height:.4f}" shape-rendering="crispEdges">',
        f'<rect width="{width:.4f}" height="{height:.4f}" fill="white"/>',
        f'<text x="{PAGE_MARGIN}" y="{PAGE_MARGIN + 5}" font-family="Arial, sans-serif" font-size="4.2" '
        f'font-weight="bold">Метки ArUco DICT_4X4_50, {size:g}x{size:g} мм (модуль {size / 6:g} мм)</text>',
    ]
    black_rects = []
    gt_corners = {}

    grid_x0 = (width - grid_w) / 2
    for k, marker_id in enumerate(ids):
        r, c = divmod(k, cols)
        cx0 = grid_x0 + c * cell_w
        cy0 = PAGE_MARGIN + title_h + r * cell_h
        mx0, my0 = cx0 + margin, cy0 + margin

        modules = marker_modules(dictionary, marker_id)
        m = size / modules.shape[0]
        svg.append(module_path(modules, mx0, my0, m))
        for rr, cc in zip(*np.nonzero(modules)):
            black_rects.append((mx0 + cc * m, my0 + rr * m, m, m))
        gt_corners[marker_id] = np.array(
            [[mx0, my0], [mx0 + size, my0], [mx0 + size, my0 + size], [mx0, my0 + size]], np.float64)

        svg.append(f'<rect x="{cx0:.4f}" y="{cy0:.4f}" width="{cut:.4f}" height="{cut:.4f}" fill="none" '
                   f'stroke="#999" stroke-width="0.2" stroke-dasharray="1,1"/>')
        label = CORNER_LABELS.get(marker_id, "доп. метка")
        svg.append(f'<text x="{cx0:.4f}" y="{cy0 + cut + 5:.4f}" font-family="Arial, sans-serif" '
                   f'font-size="3.5" font-weight="bold">ID {marker_id}</text>')
        svg.append(f'<text x="{cx0:.4f}" y="{cy0 + cut + 9.5:.4f}" font-family="Arial, sans-serif" '
                   f'font-size="3">угол {label}</text>')

    rx0 = (width - RULER_MM) / 2
    svg.append(f'<line x1="{rx0:.4f}" y1="{ruler_y:.4f}" x2="{rx0 + RULER_MM:.4f}" y2="{ruler_y:.4f}" '
               f'stroke="black" stroke-width="0.3"/>')
    for t in range(0, RULER_MM + 1, 10):
        tick = 4 if t % 50 == 0 else 2.5
        svg.append(f'<line x1="{rx0 + t:.4f}" y1="{ruler_y - tick:.4f}" x2="{rx0 + t:.4f}" y2="{ruler_y:.4f}" '
                   f'stroke="black" stroke-width="0.3"/>')
    svg.append(f'<text x="{rx0:.4f}" y="{ruler_y + 5:.4f}" font-family="Arial, sans-serif" font-size="3">'
               f'контроль масштаба печати: от 0 до {RULER_MM} должно быть ровно {RULER_MM} мм</text>')

    notes = [
        "Печать: масштаб 100%, не 'по размеру страницы'. Резать по пунктиру, белое поле оставить.",
        "Клеить на жёсткое ровное основание, верхней стороной (где ID) к +Y станка.",
        "Матовая бумага, без бликов. Метки не перепутать: ID читается камерой из узора.",
    ]
    for i, line in enumerate(notes):
        svg.append(f'<text x="{PAGE_MARGIN}" y="{notes_y + i * 5:.4f}" font-family="Arial, sans-serif" '
                   f'font-size="3">{line}</text>')
    svg.append("</svg>")

    return {"width": width, "height": height, "svg": "\n".join(svg),
            "black_rects": black_rects, "corners_mm": gt_corners}


def rasterize(sheet, px_per_mm):
    """Растр тех же чёрных прямоугольников, что в SVG (без текста) - для превью/проверки детекции."""
    img = np.full((int(round(sheet["height"] * px_per_mm)), int(round(sheet["width"] * px_per_mm))), 255, np.uint8)
    for x, y, w, h in sheet["black_rects"]:
        x0, y0 = int(round(x * px_per_mm)), int(round(y * px_per_mm))
        x1, y1 = int(round((x + w) * px_per_mm)), int(round((y + h) * px_per_mm))
        img[y0:y1, x0:x1] = 0
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=15.0, help="сторона метки (чёрная часть), мм")
    ap.add_argument("--margin", type=float, default=5.0, help="белое поле вокруг метки до линии реза, мм")
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", type=str, default=None, help="путь к SVG (по умолчанию print/aruco_markers_<size>mm.svg)")
    args = ap.parse_args()

    sheet = build(args.ids, args.size, args.margin)
    out = Path(args.out) if args.out else ROOT / "print" / f"aruco_markers_{args.size:g}mm.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet["svg"], encoding="utf-8")
    print(f"Сохранено: {out}  (лист {sheet['width']:.0f}x{sheet['height']:.0f} мм, метки ID {args.ids})")


if __name__ == "__main__":
    main()
