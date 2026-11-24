"""Estimate one rigid rotation of the calibrated translation basis, using epipolar residuals only."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from audit_tracks import matches
from pair_geometry import Capture


def residual(rot, first, second, delta, K):
    R=cv2.Rodrigues(rot)[0]
    line=np.cross(delta@R.T,first)@np.linalg.inv(K)
    # Point-to-line distance in the target image, in full-resolution pixels.
    target=second@K.T
    return np.sum(line*target,axis=1)/np.maximum(np.linalg.norm(line[:,:2],axis=1),1e-10)


def fit(cap,path):
    cv2.setRNGSeed(42)
    detector=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012)
    features=[]
    for image in cap.images:
        kp,desc=detector.detectAndCompute(image,(cap.valid&(image<250)).astype(np.uint8))
        features.append((np.array([k.pt for k in kp]),desc))
    p=[]; q=[]; translations=[]; test=[]; kinds=[]; pair_ids=[]
    dist=np.linalg.norm(cap.positions[:,None]-cap.positions,axis=2)
    step=np.median(np.min(np.where(dist>0,dist,np.inf),axis=1))
    edges=np.column_stack(np.where(np.triu((dist>.65*step)&(dist<1.55*step),1)))
    inv=np.linalg.inv(cap.K)
    for k,(i,j) in enumerate(edges):
        a,da=features[i]; b,db=features[j]
        if da is None or db is None or min(len(da),len(db))<2:
            continue
        fwd,rev=matches(da,db),matches(db,da)
        pairs=[(ia,ib) for ia,ib in fwd.items() if rev.get(ib)==ia]
        if not pairs:
            continue
        ia,ib=np.array(pairs).T
        first=np.column_stack((a[ia],np.ones(len(ia))))@inv.T
        second=np.column_stack((b[ib],np.ones(len(ib))))@inv.T
        shift=np.broadcast_to(cap.O[j]-cap.O[i],first.shape)
        # Broad gate excludes unrelated descriptor matches before the fit or evaluation.
        keep=abs(residual(np.zeros(3),first,second,shift,cap.K))<8
        p.extend(first[keep]); q.extend(second[keep]); translations.extend(shift[keep])
        test.extend([k%4==1]*sum(keep)); pair_ids.extend([k]*sum(keep))
        d=abs(cap.positions[i]-cap.positions[j])
        kind='diagonal' if d.min()>.5*step else ('X' if d[0]>d[1] else 'Y')
        kinds.extend([kind]*sum(keep))
    p,q,t,held,kinds,ids=map(np.asarray,(p,q,translations,test,kinds,pair_ids))
    if not held.any() or not (~held).any():
        raise ValueError("Insufficient independent camera pairs for fitting")
    train=~held
    result=least_squares(lambda r:residual(r,p[train],q[train],t[train],cap.K),np.zeros(3),
                         bounds=(-.035,.035),loss='soft_l1',f_scale=.5)
    before=abs(residual(np.zeros(3),p,q,t,cap.K)); after=abs(residual(result.x,p,q,t,cap.K))
    def stats(mask):
        return dict(count=int(mask.sum()),pairs=int(len(set(ids[mask]))),
                    before=np.percentile(before[mask],[50,90]).tolist(),
                    after=np.percentile(after[mask],[50,90]).tolist())
    metrics={k:stats(held&(kinds==k)) for k in ('X','Y','diagonal') if (held&(kinds==k)).any()}
    accepted=(len(metrics)==3 and all(m['pairs']>=2 and np.all(np.array(m['after'])<m['before'])
                                      for m in metrics.values()) and np.max(abs(result.x))<.0349)
    data=dict(accepted=bool(accepted),scan_sha256=sha256((cap.scan/'scan.json').read_bytes()).hexdigest(),
              rotvec_rad=result.x.tolist(),angle_deg=float(np.rad2deg(np.linalg.norm(result.x))),
              V_before=cap.V.tolist(),V_after=(cv2.Rodrigues(result.x)[0]@cap.V).tolist(),
              train=stats(train),heldout=stats(held),heldout_direction=metrics,
              heldout_pairs=[edges[k].tolist() for k in np.unique(ids[held])],
              statistic='absolute epipolar point-to-line distance [median,p90] in native pixels',
              fit='one rigid rotation; camera lengths, K, distortion and reference camera fixed')
    with path.open('x',encoding='utf8') as f:
        json.dump(data,f,indent=2)
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scan',type=Path)
    p.add_argument('--datum-from',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    fit(Capture(a.scan,a.datum_from,[68,47],7,.5),a.out)
