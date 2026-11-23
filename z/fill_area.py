"""Complete explicitly outlined planar tops in a signed XY map."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from stereo import Rectifier, _ransac_plane, half


def footprint(xy, height, polygon, K, V, normal, distance, shape):
    points = xy @ V.T
    points[..., 2] += (distance - height - points @ normal) / normal[2]
    pixels = points @ K.T
    finite = np.isfinite(height) & (pixels[..., 2] > 0)
    uv = np.zeros((*height.shape, 2), dtype=int)
    uv[finite] = np.rint(pixels[finite, :2] / pixels[finite, 2, None]).astype(int)
    u, v = np.moveaxis(uv, -1, 0)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    inside = finite & (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    inside[inside] = mask[v[inside], u[inside]] != 0
    return inside


def fit_surface(xy, height, mask, support, origin, pitch):
    # Nearby cells share image samples. Hold out entire 2 mm spatial blocks.
    index = np.rint((xy - origin) / pitch).astype(int)
    block = np.floor(index * pitch / 2).astype(int)
    held = (block[..., 0] + 3 * block[..., 1]) % 4 == 0
    train, test = support & ~held, support & held
    if train.sum() < 30 or test.sum() < 12:
        return None, dict(reason="insufficient spatial support", train=int(train.sum()), holdout=int(test.sum()))
    xyz = np.column_stack((xy[train], height[train]))
    fit = _ransac_plane(xyz, .20, 400, np.random.default_rng(0))
    if fit is None:
        return None, dict(reason="degenerate plane")
    n, d, inliers = fit
    long_axis = int(np.argmax(np.ptp(xy[mask], axis=0)))
    bounds = np.linspace(xy[mask, long_axis].min(), xy[mask, long_axis].max(), 4)
    occupied = np.histogram(xyz[inliers, long_axis], bins=bounds)[0]
    spans = np.ptp(xyz[inliers, :2], axis=0) / np.maximum(np.ptp(xy[mask], axis=0), pitch)
    info = dict(train=int(train.sum()), holdout=int(test.sum()), inliers=int(inliers.sum()),
                span_fraction=spans.tolist(), long_axis_bins=occupied.tolist())
    if np.min(spans) < .55 or np.min(occupied) < 3 or abs(n[2]) < .985:
        return None, dict(info, reason="uneven support or excessive tilt")
    prediction = (d - xy @ n[:2]) / n[2]
    p95 = float(np.percentile(np.abs(prediction[test]-height[test]), 95))
    info.update(p95_mm=p95, normal=n.tolist(), distance=float(d))
    if p95 > .35:
        return None, dict(info, reason="holdout p95 exceeds 0.35 mm")
    return prediction, info


def statistics(data, mask, expected):
    observed = mask & (data["votes"] > 0)
    confirmed = mask & data["confirmed"]
    values = data["height_mm"][observed]
    report = dict(cells=int(mask.sum()), observed=int(observed.sum()), confirmed=int(confirmed.sum()))
    if len(values):
        error = values - np.clip(values, *expected)
        report.update(median_mm=float(np.median(values)),
                      p95_error_mm=float(np.percentile(np.abs(error),95)),
                      within_010=int(np.count_nonzero(np.abs(error) <= .1)))
    if confirmed.any():
        stable = data["height_mm"][confirmed]
        report["confirmed_p95_error_mm"] = float(np.percentile(np.abs(stable - np.clip(stable, *expected)), 95))
    return report


def read_area(folder):
    with np.load(folder/"area.npz", allow_pickle=False) as data:
        grid = dict(data)
    return grid, json.loads((folder/"area.json").read_text(encoding="utf-8"))


def run(scan, annotations, near, baseline, out):
    data, meta = read_area(near)
    old, old_meta = read_area(baseline)
    height = data["height_mm"]
    pitch = float(data["cell_mm"])
    origin = data["xy_origin"] - meta["ref_position"]
    if (height.shape != old["height_mm"].shape or not np.allclose(data["xy_origin"], old["xy_origin"])
            or float(old["cell_mm"]) != pitch
            or not np.allclose(meta["normal"], old_meta["normal"]) or meta["distance"] != old_meta["distance"]):
        raise ValueError("Карты должны использовать общую сетку и плоскость стола")
    camera = json.loads((scan/"scan.json").read_text(encoding="utf-8"))
    annotation = json.loads(annotations.read_text(encoding="utf-8"))
    if annotation["frame"] != camera["ref_index"]:
        raise ValueError("Разметьте центральный опорный кадр")
    raw = cv2.imread(str(scan/f"frame_{annotation['frame']:02}.png"),cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError("Не удалось прочитать опорный кадр")
    K, D = np.array(camera["camera_matrix"]), np.array(camera["dist_coeffs"])
    rect = Rectifier(K,D,raw.shape[::-1],camera["model"])
    image, K = half(rect(raw.astype(np.float32)/256),K)
    yy, xx = np.indices(height.shape)
    xy = np.stack([origin[0]+xx*pitch, origin[1]+yy*pitch],axis=-1)
    n, d, V = np.asarray(meta["normal"]), meta["distance"], np.asarray(camera["V_cam"])
    measured = (data["votes"] > 0) & np.isfinite(height)
    confirmed = data["confirmed"] & measured
    filled = np.zeros(height.shape, bool)
    model = height.copy()
    regions, report = {}, {}
    preview = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    excluded = np.zeros(height.shape, bool)
    for region in annotation["regions"]:
        if not region["fill"]:
            evaluation_height = np.full(height.shape, region["evaluation_z_mm"])
            excluded |= footprint(xy,evaluation_height,np.asarray(region["polygon"],np.int32),K,V,n,d,image.shape)
    for region in annotation["regions"]:
        name = region["name"]
        poly = np.asarray(region["polygon"], np.int32)
        cv2.polylines(preview,[poly],True,(70,220,245),2,cv2.LINE_AA)
        def projected(z):
            return footprint(xy,z,poly,K,V,n,d,image.shape)
        fit_info = {}
        if region["fill"]:
            candidates = confirmed & (height > .5) & projected(height)
            if not candidates.any():
                report[name] = dict(reason="no observations inside polygon")
                continue
            seed = np.median(height[candidates])
        else:
            # Choose the plane used to map the evaluation window, not the measured heights.
            seed = region["evaluation_z_mm"]
        mask = projected(np.full(height.shape,seed))
        regions[name] = mask
        if region["fill"] and mask.any():
            # Stereo windows that straddle a depth discontinuity can mix top and side.
            interior = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
            support = candidates & interior
            prediction, fit_info = fit_surface(xy,height,mask,support,origin,pitch)
            if prediction is not None:
                # Use the fitted heights to place the image outline on the XY grid.
                mask = projected(prediction)
                regions[name] = mask
                missing = mask & ~measured & ~filled & ~excluded
                model[missing] = prediction[missing]
                filled |= missing
        report[name] = dict(fit=fit_info, filled=int((filled & mask).sum()),
                            all_views=statistics(old,mask,region["expected_mm"]),
                            nearest=statistics(data,mask,region["expected_mm"]))
    assert np.array_equal(model[measured],height[measured])
    assert not (filled & (measured | excluded)).any()
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/"surfaces.npz", height_mm=height, model_mm=model, measured=measured,
                        filled=filled, unknown=~(measured|filled), confirmed=data["confirmed"],
                        votes=data["votes"], spread_mm=data["spread_mm"], cell_mm=pitch,
                        xy_origin=data["xy_origin"], excluded=excluded,
                        region_names=np.asarray(list(regions)), region_masks=np.stack(list(regions.values()))
                        if regions else np.empty((0, *height.shape), bool))
    (out/"regions.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    if not cv2.imwrite(str(out/"annotations.png"),preview):
        raise OSError("Не удалось сохранить разметку")
    from render_area_surfaces import render
    render(out, xy, pitch, old["height_mm"], height, model, filled, data["confirmed"])
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ("scan","annotations","nearest","all_views","out"):
        ap.add_argument(field, type=Path)
    args = ap.parse_args()
    run(args.scan, args.annotations, args.nearest, args.all_views, args.out)
