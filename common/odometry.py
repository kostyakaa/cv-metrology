"""Relative camera travel from motion of plane features, in calibrated XY units."""
import cv2
import numpy as np


def reduced(gray):
    """Subsample with known pixel centres (small pixel x corresponds to full pixel 2*x)."""
    return cv2.pyrDown(gray)


def local_h(record):
    H = np.asarray(record['H'], float).copy()
    H[:2] -= np.asarray(record['position'])[:, None] * H[2]
    return H


def plane_points(pixels, K, D, model, H):
    p = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    if model == 'fisheye':
        u = cv2.fisheye.undistortPoints(p, K, D, P=K)
    else:
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-7)
        if hasattr(cv2, 'undistortPointsIter'):
            u = cv2.undistortPointsIter(p, K, D, None, K, criteria)
        else:
            u = cv2.undistortPoints(p, K, D, P=K, criteria=criteria)
    return cv2.perspectiveTransform(u, H).reshape(-1, 2)


def consensus(delta, pixels, shape):
    """Equal vote per occupied image region; reject competing translations or clustered support."""
    finite = np.isfinite(delta).all(axis=1)
    delta, pixels = delta[finite], pixels[finite]
    if len(delta) < 10:
        return None
    h, w = shape
    bins = (np.clip((pixels[:, 0]*6/w).astype(int), 0, 5) +
            6*np.clip((pixels[:, 1]*4/h).astype(int), 0, 3))
    regions = [np.median(delta[bins == k], axis=0) for k in np.unique(bins) if (bins == k).sum() >= 2]
    if len(regions) < 3:
        return None
    centre = np.median(regions, axis=0)
    residual = np.linalg.norm(delta-centre, axis=1)
    keep = residual < .24
    if (keep.sum() < 10 or keep.mean() < .60 or len(set(bins[keep])) < 3
            or np.any(np.ptp(pixels[keep], axis=0) < [.13*w, .13*h])):
        return None
    regions = [np.median(delta[keep & (bins == k)], axis=0) for k in np.unique(bins[keep])]
    step = np.median(regions, axis=0)
    return dict(delta_mm=step.tolist(), tracks=int(keep.sum()), regions=len(regions),
                scatter_mm=float(np.median(np.linalg.norm(delta[keep]-step, axis=1))))


class TileFeatures:
    def __init__(self, gray, K, D, model, record, sift):
        small = reduced(gray)
        # Local contrast helps on the unpainted table and in dim corners.
        enhanced = cv2.createCLAHE(2., (8, 8)).apply(small)
        keys, self.descriptors = sift.detectAndCompute(enhanced, None)
        self.pixels = np.array([k.pt for k in keys], float).reshape(-1, 2)*2
        self.shape = gray.shape
        if keys:
            self.local = plane_points(self.pixels, K, D, model, local_h(record))

    def displacement(self, other, matcher):
        if any(d is None or len(d) < 10 for d in (self.descriptors, other.descriptors)):
            return None

        def matches(a, b):
            pairs = matcher.knnMatch(a, b, k=2)
            return {p.queryIdx: p.trainIdx for p, q in pairs if p.distance < .75*q.distance}

        forward = matches(self.descriptors, other.descriptors)
        reverse = matches(other.descriptors, self.descriptors)
        pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
        if len(pairs) < 10:
            return None
        i, j = np.transpose(pairs)
        # A stationary feature moves opposite to the camera in the local metric frame.
        delta = self.local[i]-other.local[j]
        estimate = consensus(delta, self.pixels[i], self.shape)
        if estimate:
            keep = np.linalg.norm(delta-estimate['delta_mm'], axis=1) < .24
            # Feature locations delimit where this apparent scale applies on the table.
            location = self.local[i[keep]]
            _, idx = np.unique(np.floor(location/4), axis=0, return_index=True)
            estimate['local_support_mm'] = location[idx].tolist()
            estimate['other_support_mm'] = other.local[j[keep][idx]].tolist()
        return estimate


def flow_step(a, b, K, D, model, Ha, Hb):
    """Short-baseline travel between two pyrDown preview frames, without a GRBL motion seed."""
    p = cv2.goodFeaturesToTrack(a, 350, .012, 8)
    if p is None or len(p) < 10:
        return None
    q, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None)
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None)
    keep = (ok.ravel() != 0) & (ok2.ravel() != 0) & (np.linalg.norm(back-p, axis=2).ravel() < .5)
    p, q = p[keep].reshape(-1, 2), q[keep].reshape(-1, 2)
    if len(p) < 10:
        return None
    pa = plane_points(2*p, K, D, model, Ha)
    pb = plane_points(2*q, K, D, model, Hb)
    return consensus(pa-pb, p, a.shape)


def motion_leg(samples, before, after, K, D, model, status=(), feed=None):
    """Integrate the complete preview chain from one stationary capture to the next."""
    Ha, Hb = local_h(before), local_h(after)
    t0, t1 = samples[0][0], samples[-1][0]
    total = np.zeros(2)
    path = []
    # Frame receipt and status receipt use the same monotonic clock. Exposure time is unknown.
    duration = t1-t0
    telemetry = {s['status_s']: s['position'] for s in status if 0 < s['status_s'] < duration}
    telemetry.update({0.: before['position'], duration: after['position']})
    times = sorted(telemetry)
    gx, gy = np.array([telemetry[t] for t in times]).T
    max_excess = 0.
    previous_h = Ha
    for (ta, a), (tb, b) in zip(samples, samples[1:]):
        # Only local plane geometry varies with pose. XY translation never seeds flow.
        f = (tb-t0)/max(t1-t0, 1e-9)
        current_h = (1-f)*Ha + f*Hb
        step = flow_step(a, b, K, D, model, previous_h, current_h)
        if step is None:
            return dict(valid=False, reason='lost optical flow', steps=path, failed_at_s=tb-t0)
        total += step['delta_mm']
        t = tb-t0
        grbl = np.array([np.interp(t,times,gx),np.interp(t,times,gy)])
        disagreement = total-(grbl-before['position'])
        path.append(dict(t_s=t, visual_xy_since_stop=total.tolist(), grbl_xy=grbl.tolist(),
                         visual_minus_grbl_mm=disagreement.tolist(), **step))
        if status and feed and 0 < t < duration:
            k = np.clip(np.searchsorted(times,t), 1, len(times)-1)
            # Allow a status interval and camera buffering before judging transient disagreement.
            allowance = .5+(times[k]-times[k-1]+.30)*feed/60
            max_excess = max(max_excess, np.linalg.norm(disagreement)-allowance)
        previous_h = current_h
    return dict(valid=True, delta_mm=total.tolist(), steps=path, timing_consistent=bool(max_excess == 0))

