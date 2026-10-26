"""
Генерирует калибровочные доски ChArUco - SVG в точных миллиметрах.

ChArUco = обычная шахматка + маленькие коды ArUco в белых клетках. Углы клеток
ищутся как у шахматки (субпиксельно), а коды дают каждому углу уникальный номер
и точные координаты на доске. Поэтому доску НЕ обязательно видеть целиком -
калибровка идёт по любому видимому куску (у обычной шахматки частичная
детекция в OpenCV ненадёжна).

Доски (пресеты в common/markers.py):
    10mm - 16x23 клеток, 160x230 мм - обзорная камера, отдельный лист A4
    5mm  - 12x9 клеток,  60x45 мм  - поле зрения кадра ~40-100 мм
    3mm  - 12x9 клеток,  36x27 мм  - поле зрения ~25-60 мм
    2mm  - 12x9 клеток,  24x18 мм  - поле зрения ~15-40 мм
Маленькие доски печатаются на одном листе, у каждой свой диапазон ID - соседняя
доска в кадре детектору не мешает.

Как выбрать: берите доску, у которой по ширине кадра помещается примерно 8-20
клеток. Мельче - больше углов на кадр (а для дисторсии нужны углы у краёв
кадра), но клетка в кадре должна быть не меньше ~50 px, иначе коды не читаются.

Запуск:
    python scripts/make_chessboard.py              -> print/chessboard_charuco_10mm.svg + print/charuco_small_boards.svg
    python scripts/make_chessboard.py --board 3mm  -> print/charuco_3mm.svg (одна доска)

Печать: открыть SVG в Chrome/Edge -> Ctrl+P -> масштаб 100% (НЕ "по размеру
страницы"), проверить линейкой контрольную линию. Доску 2 мм печатать с
максимальным качеством (1200 dpi) - модуль кода там 0.2 мм, коды должны быть
чёткими под лупой. Наклеить на стекло/ровную жёсткую плиту, матовая бумага.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.markers import CHARUCO_BOARDS, make_charuco_board  # noqa: E402

A4_PRINTABLE = (189.0, 276.0)  # A4 минус ~10 мм полей браузера по умолчанию
SMALL_SHEET = ["5mm", "3mm", "2mm"]
MARGIN = 10.0


def exact_px_per_mm(square, marker, marker_modules):
    """Наименьшее целое px/мм, при котором клетки, коды и их модули ложатся точно в целые пиксели."""
    sizes = [square, marker, (square - marker) / 2, marker / marker_modules]
    for ppm in range(1, 201):
        if all(abs(s * ppm - round(s * ppm)) < 1e-6 for s in sizes):
            return ppm
    raise ValueError("размеры клетки/кода не дают целой сетки пикселей - выберите более круглые значения")


def raster_to_rects(img):
    """Чёрные пиксели -> прямоугольники (горизонтальные отрезки, склеенные по вертикали)."""
    h, w = img.shape
    black = img < 128
    rects, open_runs = [], {}
    for y in range(h + 1):
        runs = set()
        if y < h:
            row = np.concatenate([[False], black[y], [False]])
            edges = np.flatnonzero(np.diff(row.astype(np.int8)))
            runs = set(zip(edges[0::2], edges[1::2]))
        for run in list(open_runs):
            if run not in runs:
                y0 = open_runs.pop(run)
                rects.append((run[0], y0, run[1] - run[0], y - y0))
        for run in runs:
            open_runs.setdefault(run, y)
    return rects


def board_rects(p, x0, y0):
    """Чёрные прямоугольники доски в мм листа; левый верхний угол доски в (x0, y0)."""
    board = make_charuco_board(**p)
    ppm = exact_px_per_mm(p["square"], p["marker"], board.getDictionary().markerSize + 2)
    img = board.generateImage((int(round(p["cols"] * p["square"] * ppm)), int(round(p["rows"] * p["square"] * ppm))),
                              marginSize=0, borderBits=1)
    return [(x0 + x / ppm, y0 + y / ppm, w / ppm, h / ppm) for x, y, w, h in raster_to_rects(img)]


def describe(name):
    p = CHARUCO_BOARDS[name]
    last_id = p["first_id"] + p["cols"] * p["rows"] // 2 - 1
    return (f"ChArUco {name}: {p['cols']}x{p['rows']} клеток по {p['square']:g} мм, коды {p['marker']:g} мм, "
            f"ID {p['first_id']}-{last_id}. Калибровка: --board {name}")


def text(x, y, s, size=3.2, bold=False):
    weight = ' font-weight="bold"' if bold else ""
    return f'<text x="{x:.4f}" y="{y:.4f}" font-family="Arial, sans-serif" font-size="{size}"{weight}>{s}</text>'


def ruler(x0, y, length):
    out = [f'<line x1="{x0:.4f}" y1="{y:.4f}" x2="{x0 + length:.4f}" y2="{y:.4f}" stroke="black" stroke-width="0.3"/>']
    for t in range(0, int(length) + 1, 5):
        tick = 4 if t % 10 == 0 else 2
        out.append(f'<line x1="{x0 + t:.4f}" y1="{y - tick:.4f}" x2="{x0 + t:.4f}" y2="{y:.4f}" '
                   f'stroke="black" stroke-width="0.3"/>')
    out.append(text(x0, y + 5, f"контроль масштаба: от 0 до {length:g} должно быть ровно {length:g} мм", 3))
    return out


def page(width, height, rects, extra):
    path = "".join(f"M{x:.4f},{y:.4f}h{w:.4f}v{h:.4f}h{-w:.4f}z" for x, y, w, h in rects)
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}mm" height="{height:.2f}mm" '
        f'viewBox="0 0 {width:.4f} {height:.4f}" shape-rendering="crispEdges">',
        f'<rect width="{width:.4f}" height="{height:.4f}" fill="white"/>',
        f'<path d="{path}" fill="black"/>',
        *extra,
        "</svg>",
    ])


def build_single(name):
    p = CHARUCO_BOARDS[name]
    bw, bh = p["cols"] * p["square"], p["rows"] * p["square"]
    width, height = bw + 2 * MARGIN, bh + 2 * MARGIN + 12
    rects = board_rects(p, MARGIN, MARGIN)
    ty = MARGIN + bh + 6
    extra = [text(MARGIN, ty, f"{describe(name)}. Проверка: 10 клеток = {10 * p['square']:g} мм."),
             text(MARGIN, ty + 4.5, "Печать в масштабе 100%. Наклеить на стекло/ровную жёсткую плиту, матовая бумага.")]
    return {"width": width, "height": height, "svg": page(width, height, rects, extra), "black_rects": rects,
            "boards": {name: (MARGIN, MARGIN)}}


def build_small_sheet(names):
    pads = {n: max(2 * CHARUCO_BOARDS[n]["square"], 4.0) for n in names}  # белое поле вокруг доски до линии реза
    cut_w = {n: CHARUCO_BOARDS[n]["cols"] * CHARUCO_BOARDS[n]["square"] + 2 * pads[n] for n in names}
    width = max(max(cut_w.values()), 70.0) + 2 * MARGIN

    rects, extra, boards = [], [], {}
    y = MARGIN
    extra.append(text(MARGIN, y + 4, "Маленькие доски ChArUco для головной камеры", 4, bold=True))
    y += 10
    for n in names:
        p = CHARUCO_BOARDS[n]
        bw, bh = p["cols"] * p["square"], p["rows"] * p["square"]
        extra.append(text(MARGIN, y + 3, describe(n), 2.6))
        y += 6
        extra.append(f'<rect x="{MARGIN:.4f}" y="{y:.4f}" width="{cut_w[n]:.4f}" height="{bh + 2 * pads[n]:.4f}" '
                     f'fill="none" stroke="#999" stroke-width="0.2" stroke-dasharray="1,1"/>')
        bx, by = MARGIN + pads[n], y + pads[n]
        rects += board_rects(p, bx, by)
        boards[n] = (bx, by)
        y += bh + 2 * pads[n] + 8

    extra += ruler(MARGIN, y + 4, 50)
    y += 14
    notes = ["Резать по пунктиру. Берите доску, у которой по ширине кадра ~8-20 клеток.",
             "Печать 100%. Доску 2 мм - с максимальным качеством (1200 dpi), коды чёткие под лупой.",
             "Наклеить на ровное жёсткое основание, матовая бумага, без бликов."]
    for i, line in enumerate(notes):
        extra.append(text(MARGIN, y + i * 4.5, line, 2.8))
    height = y + len(notes) * 4.5 + MARGIN
    return {"width": width, "height": height, "svg": page(width, height, rects, extra), "black_rects": rects,
            "boards": boards}


def rasterize(sheet, px_per_mm):
    img = np.full((int(round(sheet["height"] * px_per_mm)), int(round(sheet["width"] * px_per_mm))), 255, np.uint8)
    for x, y, w, h in sheet["black_rects"]:
        x0, y0 = int(round(x * px_per_mm)), int(round(y * px_per_mm))
        x1, y1 = int(round((x + w) * px_per_mm)), int(round((y + h) * px_per_mm))
        img[y0:y1, x0:x1] = 0
    return img


def save(sheet, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet["svg"], encoding="utf-8")
    print(f"Сохранено: {out}  (лист {sheet['width']:.0f}x{sheet['height']:.0f} мм, доски: {', '.join(sheet['boards'])})")
    if sheet["width"] > A4_PRINTABLE[0] or sheet["height"] > A4_PRINTABLE[1]:
        print(f"ВНИМАНИЕ: лист больше печатной области A4 (~{A4_PRINTABLE[0]:.0f}x{A4_PRINTABLE[1]:.0f} мм).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(CHARUCO_BOARDS), default=None,
                    help="одна доска на отдельном листе; без аргумента - лист 10 мм и лист маленьких досок")
    args = ap.parse_args()

    if args.board:
        save(build_single(args.board), ROOT / "print" / f"charuco_{args.board}.svg")
    else:
        save(build_single("10mm"), ROOT / "print" / "chessboard_charuco_10mm.svg")
        save(build_small_sheet(SMALL_SHEET), ROOT / "print" / "charuco_small_boards.svg")


if __name__ == "__main__":
    main()
