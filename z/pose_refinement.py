"""Refine relative stereo poses using stationary ArUco tags at a known plane height."""
import cv2
import numpy as np
from common.markers import detect_precise, make_detector
from common.mosaic import pixel_to_machine
from common.odometry import plane_points
from common.registration import adjust, neighbours


def stereo_positions(frames, meta, args):
    pos = np.asarray(meta["positions"], float)
    height = getattr(args, "pose_tag_height", None)
    if height is None:
        return pos
    if not np.isfinite(height):
        raise ValueError("Некорректная высота опорных меток")
    previous = meta.get("pose_fit", {})
    if previous.get("tag_height_mm") == height:
        return pos + np.asarray(previous["offsets"])
    K, D = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    V = np.array(meta["V_cam"])
    ref = meta["ref_index"]
    g = meta["guess"]
    n, d = np.array(g["n"]), g["d"]-height
    fast, precise = make_detector(), make_detector(precise=True)
    observations = []
    for frame, p in zip(frames, pos):
        local = pixel_to_machine(K, V, n, d-n @ V @ (p-pos[ref]), [0., 0.])
        tags = detect_precise(fast, precise, np.clip(frame, 0, 255).astype(np.uint8))
        observations.append({mid: plane_points(corners, K, D, meta["model"], local).mean(axis=0)
                             for mid, corners in tags.items() if 4 <= mid < 50})
    edges = []
    for i,j in neighbours(pos):
        ids = observations[i].keys() & observations[j].keys()
        if not ids:
            continue
        steps = [observations[i][m] - observations[j][m] for m in ids]
        step = np.median(steps, axis=0)
        scatter = np.linalg.norm(steps-step, axis=1).max()
        if scatter <= .15 and np.linalg.norm(step-(pos[j]-pos[i])) <= .75:
            edges.append(dict(i=i, j=j, delta_mm=step.tolist(), scatter_mm=float(scatter)))
    def connected(links):
        adj = [set() for _ in pos]
        for e in links:
            adj[e['i']].add(e['j'])
            adj[e['j']].add(e['i'])
        reached, pending = {ref}, [ref]
        while pending:
            for j in adj[pending.pop()] - reached:
                reached.add(j)
                pending.append(j)
        return len(reached) == len(pos)
    test, train = edges[4::5], [e for i,e in enumerate(edges) if i%5 != 4]
    offsets = np.zeros_like(pos)
    diag = dict(tag_height_mm=height, observations=[sorted(m) for m in observations],
                accepted_edges=len(edges), test_edges=len(test), applied=False)
    if len(test) >= 3 and connected(train):
        anchor = [dict(i=ref, offset=[0., 0.])]
        cv = adjust(pos, train, anchor)
        def rms(p):
            return float(np.sqrt(np.mean([np.sum((p[e['j']]-p[e['i']]-e['delta_mm'])**2) for e in test])))
        before, after = rms(pos), rms(pos+cv)
        diag.update(rms_grbl_mm=before, rms_corrected_mm=after)
        if after < before:
            fitted = adjust(pos, edges, anchor)
            fitted -= fitted[ref].copy()
            if np.linalg.norm(fitted, axis=1).max() <= .75:
                offsets = fitted
                diag['applied'] = True
    diag['offsets'] = offsets.tolist()
    meta['pose_fit'] = diag
    print(f"Коррекция стереопоз по меткам: применена={diag['applied']}, принято пар={len(edges)}")
    return pos + offsets
