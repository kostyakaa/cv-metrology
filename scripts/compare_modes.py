"""
Сравнение режимов камеры (разрешение + кодек): какой режим кропит сенсор,
какой масштабирует, где пиксели неквадратные (растянутое 16:9 и т.п.).

Как работает: камера неподвижно смотрит на лист с метками ArUco (из
scripts/make_markers.py). Скрипт по очереди включает каждый режим, находит
одни и те же углы меток и подгоняет аффинное преобразование "пиксели режима
-> пиксели эталонного режима" (эталон - самый большой по площади). Все режимы
видят одно и то же оптическое изображение, поэтому при чистом кропе/масштабе
это преобразование строго аффинное (дисторсия на это не влияет) - и невязка
подгонки должна быть субпиксельной. Большая невязка = камера в этом режиме
дополнительно обрабатывает картинку (цифровой зум/исправление) или что-то
сдвинулось во время замера.

Для каждого режима печатается:
    scale x/y   - сколько пикселей эталона в 1 пикселе режима по осям
    aspect      - scale_y/scale_x; != 1.00 значит пиксели неквадратные (растяжение)
    FOV x/y     - какую долю поля зрения эталона покрывает режим (100% = без кропа)
    offset      - где левый верхний угол кадра режима в пикселях эталона
    rms         - невязка подгонки, px эталона

Подготовка:
  - камера и лист с метками НЕПОДВИЖНЫ всё время замера;
  - хотя бы одна метка ближе к центру кадра (её увидят и кропнутые режимы),
    лучше 2-4 метки по полю - точнее границы кадра;
  - фокус фиксированный: сохранённый (autofocus.py / detect_markers.py f) выставляется
    сам, --focus N - явно задать другой; автофокус меняет масштаб между режимами.

Запуск:
    python scripts/compare_modes.py --camera 0 --focus 480
    python scripts/compare_modes.py --camera 0 --modes YUY2:640x480 MJPG:1920x1080 MJPG:3264x2448

Без --modes режимы берутся из DirectShow (нужен pygrabber, как для
list_formats.py). Результат сохраняется в calib/modes_cam<index>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.camera import open_camera, set_focus  # noqa: E402
from common.markers import detect_precise, make_detector  # noqa: E402


def parse_mode(text):
    fourcc, _, size = text.rpartition(":")
    w, h = (int(v) for v in size.lower().split("x"))
    return fourcc.upper(), w, h


def modes_from_directshow(device_index):
    from pygrabber.dshow_graph import FilterGraph
    g = FilterGraph()
    g.add_video_input_device(device_index)
    try:
        formats = g.get_input_device().get_formats()
    finally:
        g.remove_filters()
    modes = []
    for f in formats:
        m = (f["media_type_str"].upper(), f["width"], f["height"])
        if m not in modes:
            modes.append(m)
    return modes


def measure_mode(args, mode, detectors):
    fourcc, w_req, h_req = mode
    cap = open_camera(args.camera, w_req, h_req, fourcc=fourcc or None, verbose=False)
    try:
        if args.focus is not None:
            set_focus(cap, args.focus)
        for _ in range(args.warmup):
            cap.read()
        time.sleep(0.2)
        acc, size = {}, None
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                continue
            size = (frame.shape[1], frame.shape[0])
            for marker_id, c in detect_precise(*detectors, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).items():
                acc.setdefault(marker_id, []).append(c)
    finally:
        cap.release()
    markers = {i: np.mean(v, axis=0) for i, v in acc.items()}
    return size, markers


def fit_mode_to_ref(pts_mode, pts_ref, size):
    """Аффинка mode->ref по МНК и её разбор на масштаб/аспект/кроп."""
    X = np.hstack([pts_mode, np.ones((len(pts_mode), 1))])
    A, *_ = np.linalg.lstsq(X, pts_ref, rcond=None)  # (3, 2): [x y 1] @ A = [X Y]
    pred = X @ A
    rms = float(np.sqrt(np.mean(np.sum((pred - pts_ref) ** 2, axis=1))))
    L = A[:2, :].T  # [[a b], [d e]]
    sx, sy = float(np.hypot(L[0, 0], L[1, 0])), float(np.hypot(L[0, 1], L[1, 1]))
    angle = float(np.degrees(np.arctan2(L[1, 0], L[0, 0])))
    w, h = size
    frame = np.array([[-0.5, -0.5, 1], [w - 0.5, -0.5, 1], [w - 0.5, h - 0.5, 1], [-0.5, h - 0.5, 1]])
    rect = frame @ A
    return {"affine": A.tolist(), "scale_x": sx, "scale_y": sy, "aspect": sy / sx, "angle_deg": angle,
            "rect_in_ref": rect.tolist(), "rms_px": rms}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--modes", type=str, nargs="+", default=None,
                     help="режимы вида FOURCC:WxH (например YUY2:1280x720); без аргумента - все из DirectShow")
    ap.add_argument("--focus", type=int, default=None, help="зафиксировать фокус на этом значении во всех режимах")
    ap.add_argument("--warmup", type=int, default=8, help="кадров на прогрев после переключения режима")
    ap.add_argument("--frames", type=int, default=5, help="кадров на усреднение углов меток")
    args = ap.parse_args()

    if args.modes:
        modes = [parse_mode(m) for m in args.modes]
    else:
        try:
            modes = modes_from_directshow(args.camera)
        except ImportError:
            print("Нет pygrabber - поставьте его (pip install pygrabber) или передайте --modes вручную.")
            return
    print(f"Режимов к проверке: {len(modes)}. Камеру и лист с метками не трогать!")

    detectors = (make_detector(), make_detector(precise=True))
    results = []
    for mode in modes:
        label = f"{mode[0]}:{mode[1]}x{mode[2]}"
        print(f"  {label} ...", end=" ", flush=True)
        size, markers = measure_mode(args, mode, detectors)
        if size is None:
            print("кадры не читаются")
            continue
        applied = size == (mode[1], mode[2])
        print(f"кадр {size[0]}x{size[1]}{'' if applied else ' (РЕЖИМ НЕ ПРИМЕНИЛСЯ)'}, меток: {sorted(markers)}")
        results.append({"mode": label, "requested": list(mode[1:]), "size": list(size),
                        "applied": applied, "markers": markers})

    usable = [r for r in results if r["applied"] and r["markers"]]
    if not usable:
        print("Ни в одном режиме не найдено меток - проверьте лист, фокус и свет.")
        return
    ref = max(usable, key=lambda r: r["size"][0] * r["size"][1])
    rw, rh = ref["size"]
    print(f"\nЭталон: {ref['mode']} ({rw}x{rh}). Доли поля зрения - относительно него.\n")
    print(f"{'режим':<18}{'кадр':>11}{'scale x':>9}{'scale y':>9}{'aspect':>8}{'FOV x':>8}{'FOV y':>8}"
          f"{'offset (ref px)':>18}{'rms':>7}")

    report = {"camera": args.camera, "reference": ref["mode"], "reference_size": ref["size"], "modes": []}
    for r in usable:
        common = sorted(set(r["markers"]) & set(ref["markers"]))
        if not common:
            print(f"{r['mode']:<18}  нет общих меток с эталоном (кроп не захватывает метки?)")
            continue
        pm = np.vstack([r["markers"][i] for i in common])
        pr = np.vstack([ref["markers"][i] for i in common])
        fit = fit_mode_to_ref(pm, pr, r["size"])
        rect = np.array(fit["rect_in_ref"])
        x0, y0 = rect.min(axis=0)
        x1, y1 = rect.max(axis=0)
        fov_x, fov_y = (x1 - x0) / rw * 100, (y1 - y0) / rh * 100
        print(f"{r['mode']:<18}{r['size'][0]:>6}x{r['size'][1]:<4}{fit['scale_x']:>9.3f}{fit['scale_y']:>9.3f}"
              f"{fit['aspect']:>8.3f}{fov_x:>7.1f}%{fov_y:>7.1f}%{f'({x0:.0f}, {y0:.0f})':>18}{fit['rms_px']:>7.2f}")
        warn = []
        if abs(fit["aspect"] - 1) > 0.01:
            warn.append("пиксели неквадратные - кадр растянут/сжат по одной оси")
        if fov_x < 98 or fov_y < 98:
            warn.append("режим кропит поле зрения")
        if fit["rms_px"] > 1.0:
            warn.append("большая невязка - не чистый кроп/масштаб или камера сдвинулась")
        if len(common) < 2:
            warn.append("только одна общая метка - границы кадра экстраполированы, точность ниже")
        for wtxt in warn:
            print(f"{'':<18}  ! {wtxt}")
        report["modes"].append({"mode": r["mode"], "size": r["size"], "common_markers": common,
                                "fov_percent": [fov_x, fov_y], **fit})

    wider = [m["mode"] for m in report["modes"] if max(m["fov_percent"]) > 102]
    if wider:
        print(f"\nЭталон {ref['mode']} сам кропнут: режимы {wider} видят больше него (FOV > 100%).")

    out = ROOT / "calib" / f"modes_cam{args.camera}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    not_applied = [r["mode"] for r in results if not r["applied"]]
    if not_applied:
        print(f"\nНе применились (драйвер выдал другой размер): {not_applied}")
    print(f"\nСохранено: {out}")
    print("Калибровку (calibrate_distortion.py) делайте в том режиме, в котором будете работать.")


if __name__ == "__main__":
    main()
