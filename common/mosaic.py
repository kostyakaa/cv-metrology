"""Metric plane mosaic. Image matching measures disagreement; never changes machine coordinates."""
import cv2
import numpy as np


def pixel_to_machine(K, V, n, d, position):
    # Ray/plane intersection, then displacement required to put that point on optical axis.
    B = np.linalg.inv(np.column_stack([V, [0., 0., 1.]]))[:2]
    H = np.vstack([d * B, n]) @ np.linalg.inv(K)
    H[:2] += np.asarray(position)[:, None] * H[2]
    return H


def overlap_error(a, b, mask):
    """Translation of content a->b in already metric-warped images, with forward/backward checks."""
    ga, gb = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b)]
    mask = cv2.erode(mask.astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
    p = cv2.goodFeaturesToTrack(ga, 500, .02, 12, mask=mask)
    if p is None or len(p) < 20:
        return dict(valid=False, reason='insufficient texture')
    q, ok, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p, None)
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, q, None)
    good = (ok.ravel() != 0) & (ok2.ravel() != 0) & (np.linalg.norm(back-p, axis=2).ravel() < .5)
    delta = (q-p).reshape(-1, 2)[good]
    if len(delta) < 20:
        return dict(valid=False, reason='inconsistent matches')
    med = np.median(delta, axis=0)
    keep = np.linalg.norm(delta-med, axis=1) < 1.
    valid = bool(keep.sum() >= 20 and keep.mean() >= .7)
    return dict(valid=valid, shift_px=med.tolist(), tracks=int(keep.sum()),
                scatter_px=float(np.median(np.linalg.norm(delta-med, axis=1))),
                reason='translation consensus' if valid else 'parallax or ambiguous texture')


class Mosaic:
    def __init__(self, polygon, ppm=10):
        self.polygon = np.asarray(polygon, np.float32)
        self.origin = self.polygon.min(axis=0)
        self.ppm = ppm
        self.size = tuple((np.ceil(np.ptp(self.polygon, axis=0)*ppm)+1).astype(int))
        if np.prod(self.size) > 25_000_000:
            raise ValueError('Mosaic exceeds 25 megapixels; reduce --ppm')
        self.T = np.array([[ppm,0,-ppm*self.origin[0]], [0,ppm,-ppm*self.origin[1]], [0,0,1.]])
        self.weight = np.zeros(self.size[::-1], np.float32)
        self.best = np.zeros((*self.size[::-1],3), np.uint8)
        self.region = np.zeros(self.size[::-1], np.uint8)
        cv2.fillConvexPoly(self.region, np.round((self.polygon-self.origin)*ppm).astype(np.int32), 1)
        self.previous = None
        self.errors = []

    def add(self, image, valid, H, position, raw=None, K=None, D=None, model='pinhole', diagnose=True):
        transform = self.T @ H
        warp = cv2.warpPerspective(image, transform, self.size, flags=cv2.INTER_CUBIC)
        if raw is not None:
            # Compose plane projection and lens distortion before sampling source once.
            yy,xx=np.indices(self.size[::-1],dtype=np.float32)
            points=np.stack([xx,yy],axis=-1).reshape(-1,1,2)
            uv=cv2.perspectiveTransform(points,np.linalg.inv(transform))
            norm=cv2.perspectiveTransform(uv,np.linalg.inv(K))
            if model=='fisheye':
                src=cv2.fisheye.distortPoints(norm.astype(np.float64),K,D)
            else:
                xyz=np.concatenate([norm.reshape(-1,2),np.ones((len(norm),1))],axis=1)
                src=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)[0]
            maps=src.reshape(*self.size[::-1],2).astype(np.float32)
            warp=cv2.remap(raw,maps[...,0],maps[...,1],cv2.INTER_CUBIC)
        mask = cv2.warpPerspective(valid.astype(np.uint8), self.T @ H, self.size,
                                   flags=cv2.INTER_NEAREST) * self.region
        if diagnose and self.previous is not None:
            old, oldmask, oldpos = self.previous
            e = overlap_error(old, warp, (oldmask & mask) > 0)
            e.update(position=list(position), previous_position=list(oldpos))
            if e['valid']:
                # Content shift is the negative of actual-minus-commanded camera travel.
                e['visual_minus_command_mm'] = (-np.array(e['shift_px'])/self.ppm).tolist()
            self.errors.append(e)
        # Prefer optical centre in source coordinates, not distance to clipped output boundary.
        sh,sw=image.shape[:2]
        sy,sx=np.indices((sh,sw),dtype=np.float32)
        quality=np.exp(-2*((sx-(sw-1)/2)/(sw/2))**2-2*((sy-(sh-1)/2)/(sh/2))**2)*valid
        w=cv2.warpPerspective(quality,transform,self.size)*mask
        choose=w>self.weight
        self.best[choose]=warp[choose]
        self.weight=np.maximum(self.weight,w)
        self.previous = (warp, mask, list(position))

    def image(self):
        return self.best.copy()

    def xy(self, pixel):
        return self.origin + np.asarray(pixel)/self.ppm
