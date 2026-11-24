"""Offline metric reconstruction from disjoint calibrated image pairs."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

from pair_geometry import Capture, camera_pairs, triangulate_batch


def peaks(volume, levels, margin=.06):
    index = volume.argmax(axis=0)
    best = np.take_along_axis(volume,index[None],axis=0)[0].astype(float)
    def sample(offset):
        near = np.clip(index+offset,0,len(levels)-1)
        return np.take_along_axis(volume,near[None],axis=0)[0].astype(float)
    left,right = sample(-1),sample(1)
    curvature = 2*best-left-right
    step = levels[1]-levels[0]
    rival = np.full(best.shape,-2.)
    for i,v in enumerate(volume):
        np.maximum(rival,v,out=rival,where=np.abs(i-index)*step>=1.)
    valid = ((index>0)&(index<len(levels)-1)&(curvature>1e-4)&(left>-1)&(right>-1)
             &(best>.60)&(best-rival>margin))
    correction = np.clip(.5*(right-left)/np.maximum(curvature,1e-6),-.5,.5)
    return np.where(valid,levels[index]+step*correction,np.nan),best,best-rival


def pair_search(cap, pair, levels):
    # Uniform sampling in physical XY on every hypothesis plane. Each camera pair produces its own peak.
    sub, pad = 4, 6
    step = cap.pitch/sub
    h,w = cap.shape
    size = ((w-1)*sub+1+2*pad,(h-1)*sub+1+2*pad)
    origin = cap.origin-pad*step
    samples = np.s_[pad:pad+h*sub:sub,pad:pad+w*sub:sub]
    volumes = np.full((2,len(levels),h,w),-2,np.float16)
    available = np.zeros((h,w),bool)
    textured = np.zeros((h,w),bool)
    src = [cap.images[v].astype(np.float32) for v in pair]
    masks = [(cap.valid & (im<250)).astype(np.uint8) for im in src]
    for k,z in enumerate(levels):
        warped,valid = [], []
        for j,v in enumerate(pair):
            H = cap.grid_homography(v,z,step,origin)
            warped.append(cv2.warpPerspective(src[j],H,size,flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP))
            valid.append(cv2.warpPerspective(masks[j],H,size,flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP))
        a,b = warped
        products = cv2.merge((a,b,a*a,b*b,a*b))
        for m,win in enumerate((5,9)):
            mean = cv2.boxFilter(products,-1,(win,win))[samples]
            ma,mb,aa,bb,ab = np.moveaxis(mean,-1,0)
            va,vb = aa-ma*ma,bb-mb*mb
            visible = cv2.erode(valid[0]&valid[1],np.ones((win,win),np.uint8))[samples]>0
            good = visible & (va>4) & (vb>4)
            available |= visible
            textured |= good
            score = (ab-ma*mb)/np.sqrt(np.maximum(va,4)*np.maximum(vb,4))
            volumes[m,k] = np.where(good,np.clip(score,-1,1),-2)
    small,_,_ = peaks(volumes[0],levels)
    z,score,uniqueness = peaks(volumes[1],levels)
    return z,score,uniqueness,small,available,textured


def verify_pair(cap, pair, z, sample_xy=None, return_xy=False):
    """Measure a 2D subpixel match locally, including transverse and reverse-flow residuals."""
    a,b = pair
    finite = np.isfinite(z)
    result = np.full((*z.shape,7 if return_xy else 5),np.nan,np.float32)
    if not finite.any():
        return np.moveaxis(result,-1,0)
    xy = cap.xy if sample_xy is None else sample_xy
    xyz = cap.world(xy[finite],z[finite])
    pa,pb = [cap.project(v,xyz).astype(np.float32).reshape(-1,1,2) for v in pair]
    lk = dict(winSize=(11,11),maxLevel=0,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
              criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,24,.005))
    target,ok,_ = cv2.calcOpticalFlowPyrLK(cap.images[a],cap.images[b],pa,pb.copy(),**lk)
    reverse,ok2,_ = cv2.calcOpticalFlowPyrLK(cap.images[b],cap.images[a],target,pa.copy(),**lk)
    p,q = pa[:,0,:],target[:,0,:]
    position = triangulate_batch(p,q,cap.O[a],cap.O[b],cap.K)
    depth = cap.d-position@cap.n
    error = np.maximum(np.linalg.norm(cap.project(a,position)-p,axis=1),
                       np.linalg.norm(cap.project(b,position)-q,axis=1))
    closure = np.linalg.norm(reverse[:,0]-p,axis=1)
    shift = np.linalg.norm(cap.to_xy(position)-xy[finite],axis=1)
    # Absolute depth sensitivity to 1 pixel displacement along the epipolar line.
    sensitivity = (xyz[:,2]-cap.O[b,2]) / np.maximum(np.linalg.norm(q-p,axis=1),.001)
    ok = (ok[:,0]>0)&(ok2[:,0]>0)&(position[:,2]>0)
    result[finite,:5] = np.column_stack((np.where(ok,depth,np.nan),error,closure,shift,sensitivity))
    if return_xy:
        result[finite,5:] = cap.to_xy(position)
    return np.moveaxis(result,-1,0)


def fuse(z, sigma, tol, min_pairs):
    """A unique largest cluster and a strict majority among available disjoint pairs."""
    h,w = z.shape[1:]
    fit = np.full((h,w),np.nan,np.float32)
    spread = fit.copy()
    support = np.zeros(z.shape,bool)
    n = np.isfinite(z).sum(axis=0)
    for y,x in zip(*np.where(n>0)):
        values = z[:,y,x]
        ids = np.flatnonzero(np.isfinite(values))
        ids = ids[np.argsort(values[ids])]
        groups = [ids[k:np.searchsorted(values[ids],values[i]+tol,side="right")]
                  for k,i in enumerate(ids)]
        best = max(groups,key=len)
        if sum(len(g)==len(best) for g in groups)>1 or 2*len(best)<=len(ids):
            continue
        weights = 1/np.maximum(sigma[best,y,x],.02)**2
        fit[y,x] = np.average(values[best],weights=weights)
        spread[y,x] = np.ptp(values[best])
        support[best,y,x]=True
    return fit,support,spread,support.sum(axis=0)>=min_pairs


def foreground_occlusions(cap, fit, confirmed, candidates, pairs, tol, sample_xy=None, support=None):
    """Conservative ray test using only already measured foreground in the same rectified pixel."""
    blocked=np.zeros(candidates.shape,bool)
    if sample_xy is None:
        foreground=cap.world(cap.xy[confirmed],fit[confirmed])
    else:
        measured=support & confirmed
        foreground=cap.world(sample_xy[measured],candidates[measured])
    for v in sorted({v for pair in pairs for v in pair}):
        h,w=cap.images[v].shape
        def pixels(points):
            uv=np.rint(cap.project(v,points)).astype(int)
            inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            return uv[:,1]*w+uv[:,0],inside
        index,inside=pixels(foreground)
        buffer=np.full(h*w,np.inf,np.float32)
        np.minimum.at(buffer,index[inside],foreground[inside,2])
        for i,pair in enumerate(pairs):
            if v not in pair:
                continue
            present=np.isfinite(candidates[i])
            xy=cap.xy if sample_xy is None else sample_xy[i]
            xyz=cap.world(xy[present],candidates[i,present])
            index,inside=pixels(xyz)
            behind=np.zeros(len(xyz),bool)
            behind[inside]=xyz[inside,2]>buffer[index[inside]]+tol
            blocked[i,present] |= behind
    return blocked


def run(args):
    from hashlib import sha256
    started=time.monotonic()
    cap=Capture(args.scan,args.datum_from,args.area_mm,args.area_margin,args.cell_mm)
    if args.step<=0 or args.below<0 or args.above>=cap.d or args.cell_mm<=0:
        raise ValueError("Invalid search range or grid")
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    pairs=camera_pairs(cap.O)
    levels=np.arange(-args.below,args.above+args.step*.5,args.step)
    files=[cap.scan/f"frame_{i:02}.png" for i in range(len(cap.O))]
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in [cap.scan/"scan.json",*files]}
    provenance=dict(ref_position=cap.home.tolist(),files=hashes,pairs=pairs,reference=cap.ref,normal=cap.n.tolist(),distance=cap.d,
                    datum=str(args.datum_from),cell_mm=cap.pitch,area_mm=args.area_mm,area_margin=args.area_margin,
                    scan=str(cap.scan),below=args.below,above=args.above,step=args.step,
                    pair_tolerance_mm=.35,min_disjoint_pairs=3,opencv=cv2.__version__,numpy=np.__version__)
    records=[]
    for i,pair in enumerate(pairs):
        z,score,gap,small,visible,texture=pair_search(cap,pair,levels)
        depth,err,closure,shift,sensitivity=verify_pair(cap,pair,z)
        record=dict(photo_z=z,score=score,peak_gap=gap,small_z=small,visible=visible,texture=texture,
                    triangulated_z=depth,reprojection_px=err,closure_px=closure,xy_shift_mm=shift,
                    sensitivity_mm_px=sensitivity)
        np.savez_compressed(out/f"pair_{i:02}.npz",**record)
        records.append(record)
        print("pair",pair,"unique peaks",int(np.isfinite(z).sum()),flush=True)
    layers={k:np.stack([r[k] for r in records]) for k in records[0]}
    z=layers["photo_z"]
    valid=(np.abs(z-layers["small_z"])<=.35)&(layers["reprojection_px"]<=.45)&(layers["closure_px"]<=.5)
    valid &= (layers["xy_shift_mm"]<=cap.pitch/2)&(np.abs(z-layers["triangulated_z"])<=.35)
    sigma=layers["sensitivity_mm_px"]*np.maximum(.20,layers["reprojection_px"])
    measured=np.where(valid,layers["triangulated_z"],np.nan)
    for label,candidates in [("photo",z),("geometry",measured)]:
        fit,support,spread,confirmed=fuse(candidates,sigma if label=="geometry" else np.ones_like(z),.35,3)
        # Foreground is evidence only for visibility. It cannot create a confirmation.
        blocked=foreground_occlusions(cap,fit,confirmed,candidates,pairs,.35)
        if label=="geometry":
            fit,support,spread,confirmed=fuse(np.where(blocked,np.nan,candidates),sigma,.35,3)
        n=np.isfinite(candidates).sum(axis=0)
        count=support.sum(axis=0)
        conflict=(n>=2)&(2*count<=n)
        no_texture=~layers["texture"].any(axis=0)&layers["visible"].any(axis=0)
        occluded=blocked.any(axis=0)&~confirmed&~conflict
        uncertain=np.isfinite(fit)&~confirmed&~occluded&~conflict
        no_data=~(confirmed|conflict|uncertain|no_texture|occluded)
        directory=out if label=="geometry" else out/"photo"
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/"area.npz",height_mm=fit,confirmed=confirmed,measured_confirmed=confirmed,
            measured_uncertain=uncertain,conflict=conflict,no_texture=no_texture,no_data=no_data,occluded=occluded,
            filled_or_inferred=np.zeros(cap.shape,bool),votes=n,agreeing_pairs=count,support=support,
            view_pairs=pairs,spread_mm=spread,blocked_pairs=blocked,xy_origin=cap.home+cap.origin,cell_mm=cap.pitch,
            geometric_field=layers["visible"].any(axis=0),texture_evidence=layers["texture"].any(axis=0))
        provenance.update(engine=label,cells=fit.size,confirmed=int(confirmed.sum()),seconds=time.monotonic()-started)
        (directory/"area.json").write_text(json.dumps(provenance,indent=2),encoding="utf8")
        image=np.full((*cap.shape,3),40,np.uint8)
        image[uncertain]=(145,145,145); image[conflict]=(40,90,210); image[confirmed]=(85,165,65)
        if not cv2.imwrite(str(directory/"coverage.png"),image):
            raise OSError("Failed to write coverage")
        print(label,provenance["confirmed"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--datum-from",type=Path,required=True,help="area.json defining the table plane")
    ap.add_argument("--area-mm",nargs=2,type=float,default=[68,47])
    ap.add_argument("--area-margin",type=float,default=7)
    ap.add_argument("--cell-mm",type=float,default=.5)
    ap.add_argument("--above",type=float,default=30)
    ap.add_argument("--below",type=float,default=18)
    ap.add_argument("--step",type=float,default=.25)
    run(ap.parse_args())
