"""Геометрия по калибровке камеры: точная отмена дисторсии точек и ход станка в осях камеры.

Ход станка в осях камеры (V_cam) нужен для высоты: станок двигает камеру относительно стола чистым
сдвигом, поэтому пара кадров с известным сдвигом - стереопара с известной базой.
"""
import cv2
import numpy as np

# cv2.undistortPoints по умолчанию делает 5 итераций: при бочке k1=-0.3 это ошибка до 2-3 px у краёв
# доски и до 14 px в углах кадра. Итерации до сходимости - ошибка < 0.001 px в рабочей зоне.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)


def undistort_points(pts, K, D, model="pinhole", P="K"):
    """Точки кадра (N, 2) -> те же точки без дисторсии: в пикселях камеры P (по умолчанию K) или,
    при P=None, в нормализованных координатах (x/z, y/z)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    P = K if isinstance(P, str) else P
    if model == "fisheye":
        out = cv2.fisheye.undistortPoints(pts, K, D, None, None, P, UNDISTORT_CRITERIA)
    else:
        out = cv2.undistortPoints(pts, K, D, None, None, P, UNDISTORT_CRITERIA)
    return out.reshape(-1, 2)


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def machine_motion(views, offsets, K, D, model):
    """По кадрам сетки (доска неподвижна, станок сдвигает камеру на известные ходы):
      - какая точка доски видна в главной точке против заказанных ходов (при чистом сдвиге камеры она
        смещается ровно на ход; центр камеры из позы для этого не годится - ошибка поворота позы 0.2 град
        на высоте 70 мм уже 0.2 мм) -> реальный ход на 1 мм команды, перпендикулярность осей, поворот доски;
      - расстояние камеры до доски по каждому кадру (по позе - по глубине поза точна) -> наклон плоскости
        хода к доске;
      - поворот доски в осях камеры -> ход станка в осях камеры V_cam (3x2, мм на 1 мм команды X / Y).
    Возвращает None, если кадров сетки меньше 4."""
    pairs = [(v, o) for v, o in zip(views, offsets) if o is not None]
    if len(pairs) < 4:
        return None
    seen, moves, dist, Rs = [], [], [], []
    for (obj, img), off in pairs:
        und = undistort_points(img, K, D, model)
        obj = obj.reshape(-1, 3).astype(np.float64)
        Hm, _ = cv2.findHomography(und, obj[:, :2], 0)  # пиксели без дисторсии -> мм доски
        q = Hm @ np.array([K[0, 2], K[1, 2], 1.0])
        seen.append(q[:2] / q[2])
        moves.append(off)
        ok, rvec, tvec = cv2.solvePnP(obj, und, K, None)
        R, _ = cv2.Rodrigues(rvec)
        Rs.append(R)
        dist.append(float((-R.T @ tvec).ravel()[2]))  # z камеры в осях доски (знак - куда смотрит ось z доски)
    seen, moves, dist = np.array(seen), np.array(moves, float), np.array(dist)
    X = np.hstack([moves, np.ones((len(moves), 1))])
    coef, *_ = np.linalg.lstsq(X, seen, rcond=None)
    Mx = coef[:2].T  # 2x2: смещение по доске (мм) на 1 мм команды по X / по Y
    resid = seen - X @ coef
    gz, *_ = np.linalg.lstsq(X, dist, rcond=None)  # z камеры в осях доски на 1 мм команды
    dist_resid = dist - X @ gz
    sx, sy = np.linalg.norm(Mx, axis=0)
    cosang = np.dot(Mx[:, 0], Mx[:, 1]) / (sx * sy)

    R = mean_rotation(Rs)  # доска -> камера; при чистом сдвиге одинаков во всех кадрах
    rot_spread = [float(np.degrees(np.linalg.norm(cv2.Rodrigues(Ri @ R.T)[0]))) for Ri in Rs]
    V = R @ np.vstack([Mx, gz[:2]])  # ход камеры в её осях, мм на 1 мм команды
    n = np.cross(V[:, 0], V[:, 1])
    n = n / np.linalg.norm(n) * (1 if n[2] > 0 else -1)  # нормаль плоскости хода, от камеры к столу
    nb = R[:, 2] * (1 if R[2, 2] > 0 else -1)            # нормаль доски, от камеры
    return {
        "views": len(seen),
        "real_per_cmd_x": float(sx), "real_per_cmd_y": float(sy),
        "axes_angle_deg": float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
        "board_rotation_deg": float(np.degrees(np.arctan2(Mx[1, 0], Mx[0, 0]))),
        "residual_rms_mm": float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
        "camera_height_mm": float(abs(gz[2])),
        "motion": {
            "V_cam": V.tolist(),
            "normal_cam": n.tolist(),
            "camera_tilt_deg": float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
            "board_tilt_to_motion_deg": float(np.degrees(np.arccos(np.clip(abs(nb @ n), -1, 1)))),
            "board_distance_mm": float(abs(gz[2]) * abs(nb @ n)),
            "distance_resid_rms_mm": float(np.sqrt(np.mean(dist_resid ** 2))),
            "pose_rotation_spread_deg": float(np.max(rot_spread)),
            "note": "V_cam: сдвиг камеры в её осях (x вправо, y вниз по кадру, z по оси объектива), мм на 1 мм "
                    "команды X / Y; normal_cam - нормаль плоскости хода (от камеры)",
        },
    }
