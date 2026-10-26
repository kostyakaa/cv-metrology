"""Общие настройки меток ArUco (углы стола) и калибровочных досок ChArUco."""
import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50

# ID -> угол стола в координатах станка
CORNER_LABELS = {
    0: "X0 Y0",
    1: "Xmax Y0",
    2: "Xmax Ymax",
    3: "X0 Ymax",
}

# Доски ChArUco: словарь 5x5 (не путается с метками углов стола 4x4). У каждой доски свой
# диапазон ID, поэтому соседняя доска в кадре не мешает. Первые 250 кодов DICT_5X5_1000
# совпадают с DICT_5X5_250 - лист 10 мм, напечатанный раньше, остаётся валидным.
CHARUCO_DICT = cv2.aruco.DICT_5X5_1000
CHARUCO_BOARDS = {
    "10mm": {"cols": 16, "rows": 23, "square": 10.0, "marker": 7.0, "first_id": 0},    # 160x230 мм, ID 0-183
    "5mm": {"cols": 12, "rows": 9, "square": 5.0, "marker": 3.5, "first_id": 200},     # 60x45 мм, ID 200-253
    "3mm": {"cols": 12, "rows": 9, "square": 3.0, "marker": 2.1, "first_id": 300},     # 36x27 мм, ID 300-353
    "2mm": {"cols": 12, "rows": 9, "square": 2.0, "marker": 1.4, "first_id": 400},     # 24x18 мм, ID 400-453
}


def make_charuco_board(cols, rows, square, marker, first_id=0):
    ids = np.arange(first_id, first_id + cols * rows // 2, dtype=np.int32)
    return cv2.aruco.CharucoBoard((cols, rows), square, marker, cv2.aruco.getPredefinedDictionary(CHARUCO_DICT), ids)


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT)


def make_detector(precise=False):
    """precise=False - быстрый детектор для живого окна; precise=True - для замеров (уточнение углов
    по всей длине сторон, APRILTAG: на реальных кадрах центр метки стабилен до ~0.005 px против ~0.02 px
    у быстрого, но ~60 мс на кадр 640x480 вместо ~4 мс)."""
    p = cv2.aruco.DetectorParameters()
    p.minDistanceToBorder = 1              # по умолчанию 3: метка у самого края кадра терялась
    p.adaptiveThreshWinSizeMax = 53        # окна бинаризации и для крупных модулей / неровного света
    p.polygonalApproxAccuracyRate = 0.05   # стороны у краёв кадра изогнуты дисторсией объектива
    p.cornerRefinementWinSize = 15         # верхняя граница; реальное окно ~0.3 модуля метки
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG if precise else cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(get_dictionary(), p)


def detect(detector, gray):
    """Возвращает {id: углы (4, 2) float32}; порядок углов ArUco: ЛВ, ПВ, ПН, ЛН метки."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}


def detect_precise(fast, precise, gray, pad=0.3):
    """Для замеров: быстрый поиск по всему кадру, затем точное уточнение углов только в окне вокруг каждой
    найденной метки (цена не растёт с разрешением кадра). Если точный детектор метку в окне не нашёл -
    остаются углы быстрого.

    Точный (APRILTAG) подгоняет прямые по всей длине сторон: в тестах тёмная кромка рядом с белым полем
    метки сдвигала центр у быстрого (SUBPIX) до 2.3 px, у точного - 0. Свои углы он отдаёт в соглашении
    "от края пикселя" (+0.5 px к остальному OpenCV) - здесь приводится к центру пикселя."""
    h, w = gray.shape
    out = {}
    for marker_id, c in detect(fast, gray).items():
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        m = max(x1 - x0, y1 - y0) * pad + 10
        X0, Y0 = int(max(x0 - m, 0)), int(max(y0 - m, 0))
        sub = detect(precise, gray[Y0:int(min(y1 + m, h)), X0:int(min(x1 + m, w))])
        out[marker_id] = sub[marker_id] + np.array([X0 - 0.5, Y0 - 0.5], np.float32) if marker_id in sub else c
    return out
