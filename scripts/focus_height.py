"""Motorized Z focus experiment, started with j in scan_plane. Compare two stacks offline."""
import argparse
import json
import math
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stack_positions(start, span, step, sign, limits, feed):
    if limits is None or sign not in (-1, 1) or span is None:
        raise ValueError("Задайте --focus-z-span, --z-up-sign и --z-limits перед пробой")
    if not np.isfinite([start, span, step, feed, *limits]).all() or not (0 < step <= span and feed >= 1):
        raise ValueError("Неверный диапазон, шаг или подача Z")
    if min(start, start+sign*span) < limits[0] or max(start, start+sign*span) > limits[1]:
        raise ValueError("Маршрут Z выходит за указанные границы GRBL")
    count = math.ceil(span/step)+1
    if not 5 <= count <= 201 or span/(count-1) < .01:
        raise ValueError("Нужно 5–201 остановок Z с шагом от 0,01 мм")
    return np.linspace(start, start+sign*span, count)


def peak_position(samples):
    z = np.asarray([r['z'] for r in samples], float)
    scores = np.asarray([r['score'] for r in samples], float)
    if len(z) < 5 or not np.isfinite([z, scores]).all():
        return dict(valid=False, reason='недостаточно данных')
    sort = np.argsort(z)
    z, scores = z[sort], scores[sort]
    i = int(np.argmax(scores))
    if i in (0, len(z)-1) or np.any(np.diff(z) <= 0):
        return dict(valid=False, reason='максимум на границе или повтор позиции')
    best = samples[sort[i]]
    if best.get("saturation", 0.) > .05 or best.get("gray_std", 255.) < 3:
        return dict(valid=False, reason="на вершине пересвет или низкий контраст ROI")
    rival = scores[np.abs(np.arange(len(z))-i) > 1].max()
    noise = np.median([r.get('score_std', 0.) for r in samples])
    if scores[i] - rival <= max(3*noise, .02*scores[i], 1e-9):
        return dict(valid=False, reason='плоский или неоднозначный пик')
    local = z[i-1:i+2] - z[i]
    a,b,_ = np.polyfit(local, scores[i-1:i+2], 2)
    if a >= 0 or not local[0] <= -b/(2*a) <= local[2]:
        return dict(valid=False, reason='неустойчивая аппроксимация пика')
    return dict(valid=True, z_peak_mm=float(z[i]-b/(2*a)), sampled_peak_mm=float(z[i]),
                prominence=float(scores[i]-rival), score_std=float(noise))


def acquire(tool):
    from survey_markers import Abort
    from common.camera import read_frame
    args, grbl = tool.args, tool.grbl
    grbl.require_idle()
    if not args.confirm:
        raise Abort("Для пробы Z включите подтверждения")
    start = grbl.z_position()
    frame_name = grbl.reported_frame
    try:
        levels = stack_positions(start, args.focus_z_span, args.focus_z_step, args.z_up_sign,
                                 args.z_limits, args.z_feed)
    except ValueError as exc:
        raise Abort(str(exc)) from exc
    if not tool.confirm(f"{frame_name} Z {start:.3f} -> {levels[-1]:.3f} -> {start:.3f}, {len(levels)} остановок, "
                        f"F{args.z_feed:g}. Текстура в центре, зазор по всему пути проверен?"):
        return
    if not hasattr(tool, 'focus_session'):
        tool.focus_session = uuid.uuid4().hex
    folder = tool.scans_dir / ('focus_' + time.strftime('%Y%m%d-%H%M%S') + '_' + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    data = dict(session=tool.focus_session, focus=tool.focus, profile_id=tool.profile_id,
                camera_mode=getattr(args, "camera_mode", None),
                xy=tool.position().tolist(), up_sign=args.z_up_sign, limits=args.z_limits,
                feed=args.z_feed, start_z=start, coordinate_frame=frame_name, complete=False, samples=[])
    report = folder/'focus.json'

    def save():
        tmp = report.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(report)

    expected = start
    def move_to(target):
        nonlocal expected
        current = grbl.z_position()
        if (grbl.reported_frame != frame_name or not np.isfinite(current) or abs(current-expected) > .025
                or not args.z_limits[0] <= target <= args.z_limits[1]):
            raise RuntimeError("Z расходится с ожидаемой позицией")
        grbl.jog_z(float(target-current), args.z_feed, tool.tick)
        expected = grbl.z_position()
        if grbl.reported_frame != frame_name or not np.isfinite(expected) or abs(expected-target) > .025:
            raise RuntimeError("Контроллер не подтвердил положение Z")
        return expected

    save()
    try:
        for i, target in enumerate(levels):
            actual = move_to(target)
            until = time.monotonic() + args.scan_settle
            while time.monotonic() < until:
                if tool.tick():
                    raise Abort("Отмена пробы Z")
            patches, scores = [], []
            for _ in range(max(3, args.scan_frames)):
                frame = read_frame(tool.cap)
                if tool.tick(frame):
                    raise Abort("Отмена пробы Z")
                x, y = np.rint(tool.K[:2, 2]).astype(int)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                patch = gray[max(0,y-48):y+49, max(0,x-48):x+49]
                if patch.size < 100:
                    raise ValueError("ROI вне изображения")
                patches.append(patch)
                smooth = cv2.GaussianBlur(patch, (3,3), .7)
                dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
                dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
                scores.append(float(np.mean(dx*dx + dy*dy)))
            image = np.mean(patches, axis=0).astype(np.uint8)
            if not cv2.imwrite(str(folder/f'roi_{i:03}.png'), image):
                raise OSError("Не удалось записать ROI")
            data['samples'].append(dict(z=actual, score=float(np.mean(scores)), score_std=float(np.std(scores)),
                                        saturation=float(np.mean(image>=250)), gray_std=float(image.std())))
            save()
        move_to(start)
    except (Exception, KeyboardInterrupt) as exc:
        data['error'] = str(exc)
        save()
        # Camera geometry is unknown until the operator recovers the original height.
        raise SystemExit(f"Проба прервана: {exc}. Проверьте положение Z и повторите привязку. {folder}") from exc
    data['complete'] = True
    data['peak'] = peak_position(data['samples'])
    save()
    tool.say(f"Проба Z: {data['peak']}. {folder}")


def compare(table, part):
    def read(folder):
        return json.loads((Path(folder)/'focus.json').read_text(encoding='utf-8'))
    a, b = read(table), read(part)
    for k in ('session', 'profile_id', 'focus', 'up_sign', 'coordinate_frame'):
        if a[k] != b[k]:
            raise ValueError(f"Разные условия съёмки: {k}")
    if not all(r['complete'] and r['peak']['valid'] for r in (a, b)):
        raise ValueError("Обе серии должны завершиться с выраженным внутренним пиком")
    return (b['peak']['z_peak_mm'] - a['peak']['z_peak_mm']) * a['up_sign']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table')
    parser.add_argument('part')
    args = parser.parse_args()
    print(f"Высота относительно опоры, мм: {compare(args.table, args.part):.4f}")
