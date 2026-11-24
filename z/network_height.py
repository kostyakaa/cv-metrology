"""Offline calibrated stereo on an overlapping network of short physical baselines."""
import argparse
import json
import time
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from epipolar_height import rectify_pair, bin_pair
from pair_geometry import Capture
from pair_height import foreground_occlusions, verify_pair


def adjacent_pairs(positions, extended=False):
    """Use the actual two machine coordinates. Only immediate neighbours along rows/columns."""
    p = np.asarray(positions)
    distance = np.linalg.norm(p[:, None] - p, axis=-1)
    nearest = np.min(np.where(distance > 1e-6, distance, np.inf), axis=1)
    step = np.median(nearest)
    edges = []
    for i, j in zip(*np.triu_indices(len(p), 1)):
        delta = abs(p[i] - p[j])
        short = .7 * step < distance[i, j] < 1.3 * step and delta.min() < .15 * step
        double_single = np.max(abs(np.sort(delta)/step-[1,2])) < .15
        triple = np.max(abs(np.sort(delta)/step-[0,3])) < .15
        if short or (extended and (double_single or triple)):
            edges.append((int(i), int(j)))
    return edges


def independent(edges, indices):
    """Maximum vertex-disjoint matching on the bipartite nearest-neighbour grid."""
    graph = {}
    for k in indices:
        a, b = edges[k]
        graph.setdefault(a, []).append((b, int(k)))
    matched = {}
    def augment(a, seen):
        for b, k in graph[a]:
            if b in seen:
                continue
            seen.add(b)
            if b not in matched or augment(matched[b][0], seen):
                matched[b] = a, k
                return True
        return False
    for a in graph:
        augment(a, set())
    return [k for _, k in matched.values()]


def bipartition(pairs):
    links = {}
    for a,b in pairs:
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    colors = {}
    for root in links:
        if root in colors:
            continue
        colors[root] = 0
        queue = [root]
        while queue:
            a = queue.pop()
            for b in links[a]:
                if b not in colors:
                    colors[b] = 1 - colors[a]
                    queue.append(b)
                elif colors[b] == colors[a]:
                    raise ValueError("Expected a bipartite XY grid")
    return [(a,b) if colors[a] == 0 else (b,a) for a,b in pairs]


def consensus(z, sigma, pairs, tol=.35, minimum=3):
    edges = bipartition(pairs)
    support = np.zeros(z.shape, bool)
    height = np.full(z.shape[1:], np.nan, np.float32)
    span = height.copy()
    max_votes = np.zeros(height.shape, np.uint8)
    rejected = np.zeros_like(max_votes)
    count = np.isfinite(z).sum(axis=0)
    for y,x in zip(*np.where(count)):
        val = z[:,y,x]
        ids = np.flatnonzero(np.isfinite(val))
        ids = ids[np.argsort(val[ids], kind="stable")]
        total = len(independent(edges, ids))
        max_votes[y,x] = total
        groups = []
        for j,i in enumerate(ids):
            if j and val[ids[j-1]] == val[i]:
                continue
            window = ids[j:np.searchsorted(val[ids],val[i]+tol,side="right")]
            if len(window) < max(1,(total+1)//2):
                continue
            # Prefer low-residual edges when multiple maximum matchings exist.
            ordered = window[np.argsort(sigma[window,y,x],kind="stable")]
            group = independent(edges, ordered)
            if len(group)*2 > total:
                groups.append((len(group), float(np.median(val[group])), np.ptp(val[group]), group))
        if not groups:
            continue
        largest = max(g[0] for g in groups)
        tied = [g for g in groups if g[0] == largest]
        if np.ptp([g[1] for g in tied]) > tol/2:
            continue
        _,_,width,chosen = min(tied,key=lambda g:g[2])
        weight = 1/np.maximum(sigma[chosen,y,x],.02)**2
        height[y,x] = np.average(val[chosen],weights=weight)
        span[y,x] = width
        support[chosen,y,x] = True
        rejected[y,x] = np.count_nonzero(abs(val[ids]-height[y,x]) > tol)
    return height, support, span, max_votes, rejected, support.sum(axis=0)>=minimum


def local_peak(a,b,u,v,disps,r=4):
    """Unregularized local search at a fixed image pixel across the entire depth interval."""
    x0 = max(r, u-int(disps[-1]))
    x1 = min(b.shape[1]-1-r, u-int(disps[0]))
    if (u<r or u+r>=a.shape[1] or v<r or v+r>=a.shape[0] or x1-x0<5):
        return None
    patch = a[v-r:v+r+1,u-r:u+r+1]
    if patch.std() < 2:
        return None
    stripe = b[v-r:v+r+1,x0-r:x1+r+1]
    scores = cv2.matchTemplate(stripe,patch,cv2.TM_CCOEFF_NORMED)[0]
    i = int(scores.argmax())
    if i==0 or i==len(scores)-1:
        return None
    peak = float(scores[i])
    rival = max(scores[:max(0,i-2)].max(initial=-1),scores[i+3:].max(initial=-1))
    curvature = 2*peak-scores[i-1]-scores[i+1]
    if peak <= .65 or peak-rival<=.065 or curvature<=.003:
        return None
    offset = .5*(scores[i+1]-scores[i-1])/curvature
    return u-(x0+i+offset), peak, peak-rival


def measure_pair(cap,pair,below,above,textured_ray=False,multiscale=False):
    R,O,base,focal,origin,images,masks,disps = rectify_pair(cap,pair,below,above)
    a,b = [im.astype(np.uint8) for im in images]
    low,high = int(disps[0]), int(disps[-1])
    num = 16*int(np.ceil((high-low+1)/16))
    # This aggregate is only a proposal. A raw local peak and reverse correspondence
    # must independently survive the full disparity search before a pair casts a vote.
    proposal = cv2.StereoSGBM_create(minDisparity=low,numDisparities=num,blockSize=5,
        P1=40,P2=160,uniquenessRatio=10,disp12MaxDiff=1).compute(a,b)/16
    v,u = np.where((proposal>=low)&(proposal<=high)&masks[0])
    ds = proposal[v,u]
    q = np.rint(u-ds).astype(int).clip(0,b.shape[1]-1)
    valid = masks[1][v,q]
    v,u,ds = v[valid],u[valid],ds[valid]
    def points(x,y,d):
        Z = focal*base/d
        return np.column_stack(((x+origin[0])*Z/focal,(y+origin[1])*Z/focal,Z))@R+O
    xyz = points(u,v,ds)
    z = cap.d-xyz@cap.n
    samples = np.column_stack((cap.to_xy(xyz),z,u,v,ds,(focal*base)/ds**2))
    if textured_ray:
        # Native two-direction structure makes an optical-flow match locally identifiable.
        qa=cv2.cornerMinEigenVal(a,5)
        qb=cv2.cornerMinEigenVal(b,5)
        matched_x=np.rint(u-ds).astype(int).clip(0,b.shape[1]-1)
        samples=np.column_stack((samples,np.minimum(qa[v,u],qb[v,matched_x])))
    (initial,uu,vv,dd,sens,sx,sy),mixed = bin_pair(cap,samples,below,above)
    windows=np.zeros(cap.shape,np.uint8)
    large_valid=[cv2.erode(mask.astype(np.uint8),np.ones((11,11),np.uint8)) for mask in masks] if multiscale else None
    target = np.full((4,*cap.shape),np.nan,np.float32)
    for y,x in zip(*np.where(np.isfinite(initial)&~mixed)):
        u0,v0 = int(uu[y,x]),int(vv[y,x])
        fwd = local_peak(a,b,u0,v0,disps)
        radius=4
        q=int(round(u0-dd[y,x]))
        if multiscale and (fwd is None or abs(fwd[0]-dd[y,x])>.8) and large_valid[0][v0,u0] and large_valid[1][v0,q]:
            mid=local_peak(a,b,u0,v0,disps,6)
            big=local_peak(a,b,u0,v0,disps,9)
            if mid is not None and big is not None and abs(mid[0]-big[0])<=.35:
                fwd=mid
                radius=6
        if fwd is None or abs(fwd[0]-dd[y,x])>.8:
            continue
        q = int(round(u0-fwd[0]))
        rev = local_peak(b,a,q,v0,-disps[::-1],radius)
        if rev is None or abs(fwd[0]+rev[0])>.55 or not masks[1][v0,q]:
            continue
        windows[y,x]=2*radius+1
        target[:,y,x] = fwd[0],min(fwd[1],rev[1]),min(fwd[2],rev[2]),abs(fwd[0]+rev[0])
    disp,score,gap,lr = target
    keep = np.isfinite(disp)
    xyz = points(uu[keep],vv[keep],disp[keep])
    xy = np.full((*cap.shape,2),np.nan)
    xy[keep] = cap.to_xy(xyz)
    height = np.full(cap.shape,np.nan,np.float32)
    height[keep] = cap.d-xyz@cap.n
    depth,err,back,shift,_,native_x,native_y = verify_pair(cap,pair,height,xy,return_xy=True)
    native_xy=np.stack((native_x,native_y),axis=-1)
    cell = np.rint((xy[keep]-cap.origin)/cap.pitch).astype(int)
    iy,ix = np.where(keep)
    shifted = np.zeros(cap.shape,bool)
    shifted[iy,ix] = (cell[:,0]!=ix)|(cell[:,1]!=iy)
    valid = ((abs(depth-height)<=.35)&(err<=.55)&(back<=.55)&
             (shift<=cap.pitch/2)&~shifted&(depth>=-below)&(depth<=above))
    # Reject a native triangulation that fell across the original XY cell boundary.
    if valid.any():
        pos=np.rint((native_xy[valid]-cap.origin)/cap.pitch).astype(int)
        yy,xx=np.where(valid)
        valid[yy,xx]=(pos[:,0]==xx)&(pos[:,1]==yy)
    method=np.where(valid,1,0).astype(np.uint8)
    if multiscale:
        # A stripe can localize disparity even when unconstrained 2D flow is singular.
        # Verify such observations with fresh full-resolution reciprocal native-scale peaks.
        nR,nO,nbase,nf,zero,imgs,ms,nd=rectify_pair(cap,pair,below,above,scale=1.)
        na,nb=(im.astype(np.uint8) for im in imgs)
        native_mask=[cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8)) for m in ms]
        def stripe(im):
            gx=cv2.Sobel(im,cv2.CV_32F,1,0); gy=cv2.Sobel(im,cv2.CV_32F,0,1)
            xx,yy,prod=np.moveaxis(cv2.blur(cv2.merge((gx*gx,gy*gy,gx*gy)),(13,13)),-1,0)
            trace=xx+yy
            return (xx>.55*trace)&(xx*yy-prod*prod<.10*trace**2)&(xx>16)
        stripe_a,stripe_b=stripe(na),stripe(nb)
        yy,xx=np.where(keep&~valid&~shifted)
        rays=(cap.world(xy[yy,xx],height[yy,xx])-nO)@nR.T
        natives=np.rint(rays[:,:2]*nf/rays[:,2,None]-zero).astype(int)
        for y,x,(u0,v0),P in zip(yy,xx,natives,rays):
            q=int(round(u0-nf*nbase/P[2]))
            if not (6<=u0<na.shape[1]-6 and 6<=q<nb.shape[1]-6 and 6<=v0<na.shape[0]-6):
                continue
            if not (native_mask[0][v0,u0] and native_mask[1][v0,q] and stripe_a[v0,u0] and stripe_b[v0,q]):
                continue
            core=local_peak(na,nb,u0,v0,nd,3)
            full=local_peak(na,nb,u0,v0,nd,6)
            if core is None or full is None or abs(core[0]-full[0])>.35:
                continue
            reverse=local_peak(nb,na,int(round(u0-full[0])),v0,-nd[::-1],6)
            if reverse is None or abs(reverse[0]+full[0])>.35:
                continue
            ray=np.array([(u0+zero[0])/nf,(v0+zero[1])/nf,1.])
            xyz=(ray*nf*nbase/full[0])@nR+nO
            native_z=cap.d-xyz@cap.n
            actual_xy=cap.to_xy(xyz)
            if (abs(native_z-height[y,x])>.25 or not -below<=native_z<=above or
                    not np.array_equal(np.rint((actual_xy-cap.origin)/cap.pitch),[x,y])):
                continue
            valid[y,x]=True; depth[y,x]=native_z; method[y,x]=2; native_xy[y,x]=actual_xy
            err[y,x]=np.nan; back[y,x]=abs(reverse[0]+full[0]); shift[y,x]=np.linalg.norm(actual_xy-xy[y,x])
    return dict(height=np.where(valid,depth,np.nan),verification_method=method,window_px=windows,raw_z=height,proposal_z=initial,score=score,
                gap=gap,lr_px=lr,reprojection_px=err,reverse_px=back,shift_mm=shift,mixed=mixed,
                sample_xy=xy.astype(np.float32),native_xy=native_xy,sensitivity=sens)


def solve(args):
    start = time.monotonic()
    cap = Capture(args.scan,args.datum_from,args.area_mm,args.area_margin,args.cell_mm,args.axes)
    if min(args.above,args.below)<0 or args.above>=cap.d or args.above+args.below<=0:
        raise ValueError("Invalid Z search bounds")
    pairs = adjacent_pairs(cap.positions,args.extended)
    if not pairs:
        raise ValueError("No adjacent XY camera pairs")
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,pair in enumerate(pairs):
        rec=measure_pair(cap,pair,args.below,args.above,args.textured_ray,args.multiscale)
        np.savez_compressed(out/f"pair_{i:02}.npz",**rec)
        records.append(rec)
        print(pair,int(np.isfinite(rec['height']).sum()),flush=True)
    z=np.stack([r['height'] for r in records])
    error=np.stack([r['reprojection_px'] for r in records])
    methods=np.stack([r['verification_method'] for r in records])
    sigma=np.stack([r['sensitivity'] for r in records])*np.where(methods==2,.35,np.maximum(.2,error))
    fit,support,span,votes,rejected,confirmed=consensus(z,sigma,pairs)
    confirmed &= (support&(methods==1)).any(axis=0)
    native_xy=np.stack([r["native_xy"] for r in records])
    blocked=foreground_occlusions(cap,fit,confirmed,z,pairs,.35,native_xy,support)
    active=np.where(blocked,np.nan,z)
    fit,support,span,votes,rejected,confirmed=consensus(active,sigma,pairs)
    confirmed &= (support&(methods==1)).any(axis=0)
    occluded=blocked.any(axis=0)&(votes==0)
    conflict=(votes>0)&~np.isfinite(fit)
    uncertain=(votes>0)&~confirmed&~conflict
    field=np.zeros(cap.shape,bool)
    texture=np.zeros(cap.shape,bool)
    for i,img in enumerate(cap.images):
        gray=img.astype(np.float32)
        variance=cv2.blur(gray**2,(9,9))-cv2.blur(gray,(9,9))**2
        valid=cv2.erode((cap.valid&(img<250)).astype(np.uint8),np.ones((9,9),np.uint8))
        for h in np.arange(-args.below,args.above+.01,.5):
            x,y=np.moveaxis(np.rint(cap.project(i,cap.world(cap.xy,h))).astype(int),-1,0)
            inside=(x>=0)&(x<img.shape[1])&(y>=0)&(y<img.shape[0])
            x=x.clip(0,img.shape[1]-1); y=y.clip(0,img.shape[0]-1)
            inside &= valid[y,x]>0
            field |= inside
            texture |= inside&(variance[y,x]>4)
    no_texture=field&~texture&~(votes>0)&~occluded
    no_data=(votes==0)&~occluded&~no_texture
    precision=np.full(cap.shape,np.nan,np.float32)
    information=np.where(support,1/np.maximum(sigma,.02)**2,0).sum(axis=0)
    np.divide(1,np.sqrt(information),where=information>0,out=precision)
    np.savez_compressed(out/'area.npz',height_mm=fit,confirmed=confirmed,measured_confirmed=confirmed,
        measured_uncertain=uncertain,conflict=conflict,no_texture=no_texture,no_data=no_data,occluded=occluded,
        filled_or_inferred=np.zeros_like(confirmed),support=support,votes=votes,view_pairs=pairs,
        rejected_observations=rejected,blocked_pairs=blocked,spread_mm=span,precision_proxy_mm=precision,
        geometric_field=field,texture_evidence=texture,xy_origin=cap.home+cap.origin,cell_mm=cap.pitch)
    colors=np.full((*cap.shape,3),40,np.uint8)
    colors[uncertain]=(145,145,145); colors[conflict]=(40,90,210); colors[confirmed]=(85,165,65)
    if not cv2.imwrite(str(out/'coverage.png'),colors):
        raise OSError("Could not write coverage")
    sources=[cap.scan/'scan.json',*[cap.scan/f'frame_{i:02}.png' for i in range(len(cap.images))]]
    stats=dict(engine='network',visibility='subcell native triangulations',seconds=time.monotonic()-start,pairs=pairs,confirmed=int(confirmed.sum()),
        cells=fit.size,observed=int((votes>0).sum()),conflicts=int(conflict.sum()),uncertain=int(uncertain.sum()),
        occluded=int(occluded.sum()),no_texture=int(no_texture.sum()),no_data=int(no_data.sum()),filled=0,
        scan=str(cap.scan),normal=cap.n.tolist(),distance=cap.d,ref_position=cap.home.tolist(),
        multiscale=args.multiscale,extended=args.extended,textured_ray=args.textured_ray,axes=str(args.axes) if args.axes else None, V_effective=cap.V.tolist(),
        datum=str(args.datum_from),area_mm=args.area_mm,area_margin=args.area_margin,cell_mm=cap.pitch,
        below=args.below,above=args.above,min_disjoint_pairs=3,tolerance_mm=.35,numpy=np.__version__,
        opencv=cv2.__version__,source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'area.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print('confirmed',stats['confirmed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scan',type=Path)
    p.add_argument('--datum-from',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--area-mm',nargs=2,type=float,default=[68,47])
    p.add_argument('--multiscale',action='store_true',help='Check larger same-depth patches and aperture-constrained native disparity')
    p.add_argument('--extended',action='store_true',help='Include (2,1) and (3,0) grid baselines')
    p.add_argument('--textured-ray',action='store_true',help='Select a localizable ray within the median-depth band')
    p.add_argument('--axes',type=Path,help='Validated rotation from fit_stereo_axes.py')
    p.add_argument('--cell-mm',type=float,default=.5)
    p.add_argument('--area-margin',type=float,default=7)
    p.add_argument('--below',type=float,default=18)
    p.add_argument('--above',type=float,default=30)
    solve(p.parse_args())
