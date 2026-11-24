"""Independent SIFT correspondences and held-out pose residuals on the archived scan."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

from pair_geometry import Capture, triangulate_batch
from stereo import triangulate


def matches(query, train):
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64))
    return {a.queryIdx:a.trainIdx for a,b in matcher.knnMatch(query,train,k=2) if a.distance<.75*b.distance}


def audit(cap, out):
    start=time.monotonic()
    cv2.setRNGSeed(14)
    detector=cv2.SIFT_create(nfeatures=4500,contrastThreshold=.012)
    observations=[]
    for image in cap.images:
        keys,desc=detector.detectAndCompute(image,(cap.valid&(image<250)).astype(np.uint8))
        observations.append((np.asarray([k.pt for k in keys]),desc))
    ref=cap.ref
    p,desc=observations[ref]
    tracks=[{} for _ in p]
    pairs={}
    for j,(q,other) in enumerate(observations):
        if j==ref or other is None or desc is None:
            continue
        forward,reverse=matches(desc,other),matches(other,desc)
        edges=[(a,b) for a,b in forward.items() if reverse.get(b)==a]
        if not edges:
            continue
        a,b=np.array(edges).T
        xyz=triangulate_batch(p[a],q[b],cap.O[ref],cap.O[j],cap.K)
        e=np.linalg.norm(cap.project(j,xyz)-q[b],axis=1)
        visible=(xyz[:,2]>0)&(e<1.)
        # Wrong motion sign reverses the triangulated depths for the same matched pixels.
        pairs[str(j)]=dict(mutual=len(edges),positive=int((xyz[:,2]>0).sum()),
                           within_one_px=int(visible.sum()),positive_if_sign_reversed=int((xyz[:,2]<0).sum()))
        for ai,bi,z in zip(a[visible],b[visible],xyz[visible]):
            tracks[ai][j]=(q[bi],float(cap.d-z@cap.n))
    points=[]; rms=[]; valid_views=[]; residuals={i:[] for i in range(len(cap.O))}
    for ai,t in enumerate(tracks):
        if len(t)<4:
            continue
        med=np.median([item[1] for item in t.values()])
        ids=[j for j in t if abs(t[j][1]-med)<=.50]
        if len(ids)<4:
            continue
        # Hold out one whole view, cycling over IDs to diagnose each camera.
        withheld=ids.pop(ai%len(ids))
        pixels=np.array([p[ai]]+[t[j][0] for j in ids])
        used=[ref,*ids]
        X,error=triangulate(pixels,cap.O[used],cap.K)
        residual=cap.project(withheld,X)-t[withheld][0]
        residuals[withheld].append(residual)
        if error>.7 or np.linalg.norm(residual)>1.25:
            continue
        points.append(X); rms.append(error); valid_views.append(len(used)+1)
    xyz=np.asarray(points).reshape(-1,3)
    heights=cap.d-xyz@cap.n
    xy=cap.to_xy(xyz)
    ids=np.rint((xy-cap.origin)/cap.pitch).astype(int)
    inside=(ids>=0).all(axis=1)&(ids[:,0]<cap.shape[1])&(ids[:,1]<cap.shape[0])
    report=dict(pairs=pairs,tracks=len(points),inside=int(inside.sum()),cells=int(len(np.unique(ids[inside],axis=0))),
                reprojection_median_px=float(np.median(rms)) if rms else None,seconds=time.monotonic()-start,
                heldout_px={i:dict(count=len(v),median_xy=np.median(v,axis=0).tolist(),
                                   p95=float(np.percentile(np.linalg.norm(v,axis=1),95)))
                            for i,v in residuals.items() if v})
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"tracks.npz",xyz=xyz,z=heights,xy=xy,rms_px=rms,views=valid_views)
    (out/"tracks.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print({k:v for k,v in report.items() if k not in ("pairs","heldout_px")})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan",type=Path)
    ap.add_argument("--datum-from",required=True,type=Path)
    ap.add_argument("--axes",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    audit(Capture(args.scan,args.datum_from,[68,47],7,.5,args.axes),args.out)
