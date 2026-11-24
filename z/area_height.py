"""Multi-reference signed height map in home-session command XY. Works offline and after b."""
import argparse
import copy
import json
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import binned_statistic

from area_consensus import robust_consensus
from measure_height import ScanResult, add_height_args, load_scan, replay_args
from pose_refinement import stereo_positions


def reference_indices(positions, first, count):
    selected = [first]
    while len(selected) < min(count, len(positions)):
        distances = np.linalg.norm(positions[:, None] - positions[selected], axis=2).min(axis=1)
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return selected


def reconstruct(frames, meta, args, out, first_scan=None, tick=None, datum_from=None):
    pos = stereo_positions(frames, meta, args)
    first = meta["ref_index"]
    V = np.asarray(meta["V_cam"])
    to_xy = np.linalg.inv(np.column_stack([V, [0, 0, 1]]))[:2]
    guess = meta["guess"]
    n = np.asarray(guess["n"])
    plane = meta.get("global_plane")
    started = time.monotonic()
    pitch = args.cell_mm
    if not np.isfinite(pitch) or pitch <= 0 or not 2 <= args.references <= len(pos):
        raise ValueError("Проверьте --cell-mm и --references")
    if not np.isfinite(args.agree_mm) or args.agree_mm <= 0:
        raise ValueError("Порог согласия должен быть положительным")
    refs = reference_indices(pos, first, args.references)
    roi = args.area_mm or meta.get("scan_plan", {}).get("area_mm")
    margin = args.area_margin if args.area_mm else meta.get("scan_plan", {}).get("area_margin", 0.)
    if roi and (not np.isfinite([*roi, margin]).all() or min(roi) <= 0 or margin < 0):
        raise ValueError("Неверная область измерения")
    robust = getattr(args, "fusion", "range") == "robust"
    tiles, cameras, offsets = [], [], []
    for ref in refs:
        offset = V @ (pos[ref] - pos[first])
        if ref == first and first_scan is not None:
            scan = first_scan
        else:
            keep = np.argsort(np.linalg.norm(pos-pos[ref], axis=1), kind='stable')[:len(pos) if ref == first else 25]
            local_ref = int(np.flatnonzero(keep == ref)[0])
            local_guess = (n, guess["d"] - n @ offset, guess["kind"])
            global_plane = None
            if plane:
                gn = np.asarray(plane["n"])
                global_plane = gn, plane["d"] - gn @ offset
            settings = copy.copy(args)
            settings.fill_planes = False
            if ref != first:
                settings.table_roi = None
            scan = ScanResult([frames[i] for i in keep], pos[keep], local_ref, meta["camera_matrix"],
                              np.array(meta["dist_coeffs"]), meta["model"], V, local_guess,
                              global_plane, settings, tick=tick)
        if ref == first:
            normal, distance, datum = scan.n_t.copy(), scan.d_t, scan.ref_kind
            if datum_from is not None:
                normal = np.asarray(datum_from["normal"], float)
                distance = float(datum_from["distance"])
                datum = datum_from["table_reference"]
                if (not np.allclose(datum_from["ref_position"], pos[first]) or normal.shape != (3,)
                        or not np.isclose(np.linalg.norm(normal), 1) or not np.isfinite(distance)):
                    raise ValueError("Опорная плоскость несовместима с текущим сканом")
        world = scan.X[scan.valid] + offset
        xy = world @ to_xy.T
        z = distance - world @ normal
        if robust:
            depth = np.full(scan.valid.shape, np.nan, np.float32)
            depth[scan.valid] = z
            cameras.append((scan.Ks, depth))
            offsets.append(offset)
            tiles.append((xy, z, scan.score[scan.valid], world))
        else:
            tiles.append((xy, z))
        del scan
    if roi:
        extent = np.asarray(roi) / 2 + margin
        low, high = -extent, extent
    else:
        observed = [tile[0] for tile in tiles if tile[0].size]
        if not observed:
            raise ValueError("Нет измеренных ячеек. Укажите область для отчёта о пустом скане")
        low = np.min([p.min(axis=0) for p in observed], axis=0)
        high = np.max([p.max(axis=0) for p in observed], axis=0)
    lo = np.floor(low / pitch).astype(int)
    w, h = np.ceil(high / pitch).astype(int) - lo + 1
    if int(w)*int(h)*len(refs) > 40_000_000:
        raise ValueError("Слишком мелкая сетка для области")
    layers = np.full((len(refs), h, w), np.nan, np.float32)
    if robust:
        scores = np.full_like(layers,np.nan)
        points = np.full((*layers.shape,3),np.nan,np.float32)
    for i, (layer, tile) in enumerate(zip(layers,tiles)):
        xy, z = tile[:2]
        ij = np.rint(xy / pitch).astype(int) - lo
        x, y = ij.T
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if valid.any():
            cell, index = np.unique(y[valid]*w+x[valid], return_inverse=True)
            layer.flat[cell] = binned_statistic(index, z[valid], 'median', bins=np.arange(len(cell)+1)-.5).statistic
            if robust:
                # Keep a real 3D sample nearest the median for reprojection.
                residual = np.abs(z[valid] - layer.flat[cell[index]])
                order = np.lexsort((-tile[2][valid], residual, index))
                first_sample = np.r_[True, np.diff(index[order]) != 0]
                representative = order[first_sample]
                scores[i].flat[cell] = tile[2][valid][representative]
                points[i].reshape(-1,3)[cell] = tile[3][valid][representative]
    votes = np.isfinite(layers).sum(axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        spread = np.nanmax(layers, axis=0) - np.nanmin(layers, axis=0)
        height = np.nanmedian(layers, axis=0)
    confirmed = (votes >= 2) & (spread <= args.agree_mm)
    extras = {}
    if robust:
        baseline, old_confirmed = height, confirmed
        height, method, support, rejected = robust_consensus(
            layers, points, cameras, np.asarray(offsets), normal, distance, args.agree_mm, height, confirmed)
        confirmed = method > 0
        extras = dict(layer_height_mm=layers, layer_score=scores, reference_indices=refs,
                      baseline_height_mm=baseline, baseline_confirmed=old_confirmed,
                      confirmed_height_mm=np.where(confirmed,height,np.nan),
                      single=(votes == 1) & ~confirmed, conflict=(votes >= 2) & ~confirmed,
                      confirmation_method=method, agreeing_views=support, rejected_observations=rejected)
    start = pos[first] + lo * pitch
    result = dict(seconds=time.monotonic()-started, references=refs, cell_mm=pitch, xy_origin_command_mm=start.tolist(),
                  ref_position=pos[first].tolist(), area_mm=roi, area_margin=margin,
                  table_reference=datum, normal=normal.tolist(), distance=float(distance),
                  view_selection=getattr(args,"view_selection","diverse"), map_views=args.map_views,
                  agreement_mm=args.agree_mm, cells=int(votes.size),
                  observed=int((votes > 0).sum()), confirmed=int(confirmed.sum()),
                  conflicting=int(((votes >= 2) & ~confirmed).sum()))
    if robust:
        result.update(fusion="robust", baseline_confirmed=int(old_confirmed.sum()),
                      majority_confirmed=int((method == 2).sum()), reprojection_confirmed=int((method == 3).sum()),
                      single=int(extras["single"].sum()), rejected_observations=int(rejected.sum()))
    if args.pose_tag_height is not None and "pose_fit" in meta:
        result["pose_fit"] = meta["pose_fit"]
    if roi:
        xx, yy = np.meshgrid((lo[0]+np.arange(w))*pitch, (lo[1]+np.arange(h))*pitch)
        edge = np.maximum(abs(xx)-roi[0]/2, abs(yy)-roi[1]/2)
        ring = abs(edge) <= pitch*.75
        result["requested_perimeter_band_coverage"] = float(confirmed[ring].mean()) if ring.any() else None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "area"
        stage.mkdir()
        np.savez_compressed(stage/"area.npz", height_mm=height, votes=votes.astype(np.uint16), spread_mm=spread,
                            confirmed=confirmed, xy_origin=start, cell_mm=pitch, **extras)
        image = np.full((*height.shape, 3), 38, np.uint8)
        image[votes == 1] = [150, 150, 150]
        image[(votes >= 2) & ~confirmed] = [35, 100, 220]
        image[confirmed] = [100, 180, 40]
        if not cv2.imwrite(str(stage/"coverage.png"), image):
            raise OSError("Не удалось сохранить coverage.png")
        (stage/"area.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        stage.rename(out)
    print(out, result)
    return result


def main():
    parser = add_height_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('folder', type=Path)
    parser.add_argument('--fusion', choices=('range','robust'), default='range',
                        help='Робастное объединение и геометрическая проверка в архивном режиме')
    parser.add_argument('--datum-from', type=Path, help='area.json с плоскостью сравниваемого скана')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out должен быть новым каталогом")
    frames, meta = load_scan(args.folder)
    datum = json.loads(args.datum_from.read_text(encoding="utf-8")) if args.datum_from else None
    reconstruct(frames, meta, replay_args(parser, meta), args.out, datum_from=datum)


if __name__ == '__main__':
    main()
