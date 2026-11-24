"""Camera-only geometry for metric height maps with calibrated translations."""
import json
from pathlib import Path
import cv2
import numpy as np

from stereo import Rectifier


class Capture:
    def __init__(self, scan, datum, dims, margin, pitch, axes_file=None):
        self.scan = Path(scan)
        self.meta = json.loads((self.scan/"scan.json").read_text(encoding="utf-8"))
        self.datum = json.loads(Path(datum).read_text(encoding="utf-8"))
        self.K = np.array(self.meta["camera_matrix"])
        self.V = np.array(self.meta["V_cam"])
        if axes_file is not None:
            from hashlib import sha256
            fitted = json.loads(Path(axes_file).read_text(encoding="utf8"))
            if (not fitted["accepted"] or fitted["scan_sha256"] != sha256((self.scan/"scan.json").read_bytes()).hexdigest()
                    or not np.allclose(self.V, fitted["V_before"])):
                raise ValueError("Axis validation failed or refers to another scan")
            self.V = cv2.Rodrigues(np.asarray(fitted["rotvec_rad"]))[0] @ self.V
        self.positions = np.array(self.meta["positions"])
        if (len(dims) != 2 or not np.all(np.isfinite([*dims, margin, pitch]))
                or min(dims) <= 0 or margin < 0 or pitch <= 0):
            raise ValueError("Invalid XY grid")
        if self.positions.ndim != 2 or self.positions.shape[1] != 2 or not np.isfinite(self.positions).all():
            raise ValueError("Invalid machine positions")
        self.ref = self.meta["ref_index"]
        self.home = self.positions[self.ref]
        if not np.allclose(self.home,self.datum["ref_position"]):
            raise ValueError("Incompatible datum reference")
        self.O = (self.positions-self.home) @ self.V.T
        self.n = np.array(self.datum["normal"])
        self.d = float(self.datum["distance"])
        if not np.isfinite(self.d) or self.d <= 0:
            raise ValueError("Invalid table distance")
        if not np.isclose(np.linalg.norm(self.n),1) or self.n[2] <= 0:
            raise ValueError("Invalid table normal")
        self.pitch = pitch
        half = np.ceil((np.asarray(dims)/2+margin)/pitch).astype(int)
        self.origin = -half*pitch
        self.shape = tuple((2*half+1)[::-1])
        iy,ix = np.indices(self.shape)
        self.xy = self.origin + pitch*np.stack((ix,iy),axis=-1)
        self.images = []
        for i in range(len(self.O)):
            raw = cv2.imread(str(self.scan/f"frame_{i:02}.png"),cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
                raise ValueError(f"Missing or invalid frame {i}")
            if i == 0:
                rect = Rectifier(self.K,np.array(self.meta["dist_coeffs"]),raw.shape[::-1],self.meta["model"])
                self.valid = rect.valid
            if raw.shape != self.valid.shape:
                raise ValueError("Inconsistent frame size")
            self.images.append(rect((raw.astype(np.float32)/256)).round().clip(0,255).astype(np.uint8))

    def world(self, xy, height):
        xyz = xy @ self.V.T
        xyz[...,2] += (self.d-height-xyz@self.n)/self.n[2]
        return xyz

    def grid_homography(self, view, height, step, origin):
        # Exact mapping from texel centres on a table-parallel plane to rectified pixel centres.
        tangents = self.V.copy()
        tangents[2] -= self.n@self.V/self.n[2]
        return self.K @ np.column_stack((tangents*step, self.world(origin,height)-self.O[view]))

    def project(self, view, xyz):
        q = (xyz-self.O[view])@self.K.T
        return q[...,:2]/q[...,2,None]

    def to_xy(self, xyz):
        return xyz @ np.linalg.inv(np.column_stack((self.V,[0.,0.,1.])))[:2].T


def triangulate_batch(a,b,oa,ob,K):
    """Two translated perspective cameras, LS in normalized image coordinates."""
    f = K.diagonal()[:2]
    ra = (a-K[:2,2])/f
    rb = (b-K[:2,2])/f
    disparity = ra-rb
    delta = ob[:2]-oa[:2] + ra*oa[2]-rb*ob[2]
    depth = np.sum(disparity*delta,axis=-1)/np.maximum(np.sum(disparity**2,axis=-1),1e-18)
    xy = ((depth[...,None]-oa[2])*ra+oa[:2]+(depth[...,None]-ob[2])*rb+ob[:2])/2
    return np.column_stack((xy,depth))


def camera_pairs(centres):
    """Spatially interleaved disjoint short/diagonal baselines on the scan grid."""
    delta = centres[:,None]-centres
    dist = np.linalg.norm(delta,axis=-1)
    step = np.median(np.min(np.where(dist>1e-6,dist,np.inf),axis=1))
    # Prefer spread in orientation while preserving the highest attainable number of short pairs.
    pairs = []
    occupied = set()
    for i in np.argsort(np.sum(dist<=1.55*step,axis=1),kind="stable"):
        if i in occupied:
            continue
        candidates = [j for j in range(len(dist)) if j not in occupied and j!=i
                      and .65*step <= dist[i,j] <= 2.1*step]
        if not candidates:
            continue
        def priority(j):
            direction = delta[i,j]/dist[i,j]
            redundancy = sum((direction@(delta[a,b]/dist[a,b]))**4 for a,b in pairs)
            return redundancy + abs(dist[i,j]/step - 1)
        j=min(candidates,key=priority)
        pairs.append((int(i),j))
        occupied.update((i,j))
    return pairs
