"""Fixed-pixel epipolar stereo with reciprocal uniqueness and disjoint-camera validation."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pair_geometry import Capture, camera_pairs
from pair_height import fuse, foreground_occlusions, verify_pair


def axes(offset):
    x=offset/np.linalg.norm(offset)
    y=np.cross([0,0,1],x)
    y/=np.linalg.norm(y)
    return np.stack((x,y,np.cross(x,y)))


def correlation(a,b,win):
    ma,mb,aa,bb,ab=np.moveaxis(cv2.boxFilter(cv2.merge((a,b,a*a,b*b,a*b)),-1,(win,win)),-1,0)
    va,vb=aa-ma*ma,bb-mb*mb
    return (ab-ma*mb)/np.sqrt(np.maximum(va,4)*np.maximum(vb,4)),(va>4)&(vb>4)


def subpeak(volume, disparities, separation):
    idx=volume.argmax(axis=0)
    best=np.take_along_axis(volume,idx[None],axis=0)[0].astype(np.float32)
    left=np.take_along_axis(volume,np.maximum(0,idx-1)[None],axis=0)[0].astype(np.float32)
    right=np.take_along_axis(volume,np.minimum(len(volume)-1,idx+1)[None],axis=0)[0].astype(np.float32)
    den=2*best-left-right
    rival=np.full(idx.shape,-2,np.float32)
    for k,score in enumerate(volume):
        np.maximum(rival,score,out=rival,where=np.abs(idx-k)>=separation)
    gap=best-rival
    valid=(idx>0)&(idx<len(volume)-1)&(best>.65)&(den>.003)&(gap>.065)&(left>-1)&(right>-1)
    disp=disparities[idx]+.5*(right-left)/np.maximum(den,.003)
    return np.where(valid,disp,np.nan), best,gap


def rectify_pair(cap,pair,below,above,scale=.75):
    a,b=pair
    O=cap.O[a]
    R=axes(cap.O[b]-O)
    base=np.linalg.norm(cap.O[b]-O)
    focal=float(cap.K[0,0])*scale
    x,y=cap.origin
    corners=np.array([[i*x,j*y] for i in [-1,1] for j in [-1,1]])
    box=np.vstack([cap.world(corners,np.full(4,level)) for level in (-below,above)])
    bounds=np.vstack([(box-cap.O[v])@R.T for v in pair])
    if bounds[:,2].min()<=1:
        raise ValueError("Requested volume crosses a camera")
    uv=bounds[:,:2]*focal/bounds[:,2,None]
    origin=np.floor(uv.min(axis=0)).astype(int)-8
    w,h=np.ceil(uv.max(axis=0)).astype(int)-origin+9
    Kr=np.array([[focal,0,-origin[0]],[0,focal,-origin[1]],[0,0,1.]])
    H=cap.K@R.T@np.linalg.inv(Kr)
    images=[]; masks=[]
    for v in pair:
        warped=cv2.warpPerspective(cap.images[v],H,(w,h),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        mask=cv2.warpPerspective((cap.valid & (cap.images[v]<250)).astype(np.uint8),
                                 H,(w,h),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_NEAREST)
        images.append(warped.astype(np.float32))
        masks.append(cv2.erode(mask,np.ones((9,9),np.uint8))>0)
    disparities=np.arange(max(1,int(focal*base/bounds[:,2].max())-2),
                          int(np.ceil(focal*base/bounds[:,2].min()))+3)
    return R, O, base, focal, origin, images, masks, disparities


def epipolar_pair(cap,pair,below,above):
    R,O,base,focal,origin,images,masks,disparities=rectify_pair(cap,pair,below,above)
    h,w=images[0].shape
    selected=[]
    for row in range(0,h,36):
        end=min(row+36,h)
        lo,hi=max(row-5,0),min(end+5,h)
        rows=slice(row-lo,end-lo)
        size=(len(disparities),end-row,w)
        left=np.full(size,-2,np.float16)
        right=np.full_like(left,-2)
        for k,d in enumerate(disparities):
            if d>=w: continue
            A,B=images[0][lo:hi,d:],images[1][lo:hi,:-d]
            cc,texture=correlation(A,B,9)
            inner,smalltexture=correlation(A,B,5)
            ok=masks[0][lo:hi,d:] & masks[1][lo:hi,:-d] & texture & smalltexture & (inner>.4)
            # Edges with decorrelated central and surrounding patches lack single-surface support.
            ok[:,:4]=False
            ok[:,-4:]=False
            score=np.where(ok,np.clip(cc,-1,1),-2)[rows]
            left[k,:,d:]=score
            right[k,:,:-d]=score
        disp,best,gap=subpeak(left,disparities,3)
        reverse,_,_=subpeak(right,disparities,3)
        iy,ix=np.where(np.isfinite(disp))
        if not len(iy): continue
        ds=disp[iy,ix]
        q=np.rint(ix-ds).astype(int).clip(0,w-1)
        closure=abs(ds-reverse[iy,q])
        good=np.isfinite(closure)&(closure<=.55)
        iy,ix,ds,closure=iy[good],ix[good],ds[good],closure[good]
        Z=focal*base/ds
        P=np.column_stack(((ix+origin[0])*Z/focal,(iy+row+origin[1])*Z/focal,Z))@R+O
        z=cap.d-P@cap.n
        # Do not average different depths within a cell: save real samples first.
        selected.append(np.column_stack((cap.to_xy(P),z,best[iy,ix],gap[iy,ix],closure,Z/ds)))
    return np.vstack(selected) if selected else np.empty((0,7))


def bin_pair(cap,points,below,above):
    ids=np.rint((points[:,:2]-cap.origin)/cap.pitch).astype(int)
    ok=(ids>=0).all(axis=1)&(ids[:,0]<cap.shape[1])&(ids[:,1]<cap.shape[0])
    ok &= (points[:,2]>=-below)&(points[:,2]<=above)
    occupied=ids[ok,1]*cap.shape[1]+ids[ok,0]
    p=points[ok]
    ncell=np.prod(cap.shape)
    bins=np.bincount(occupied,minlength=ncell)
    order=np.lexsort((p[:,2],occupied))
    starts=np.cumsum(bins)-bins
    cells=np.flatnonzero(bins>=3)
    n=bins[cells]
    def at(fraction):
        return order[starts[cells]+np.floor((n-1)*fraction).astype(int)]
    quartile_span=p[at(.8),2]-p[at(.2),2]
    middle=at(.5)
    if points.shape[1] == 8 and len(cells):
        # On a locally single-depth patch choose a textured real ray near the median.
        slot=np.searchsorted(cells,occupied)
        mapped=slot.clip(0,len(cells)-1)
        valid=(cells[mapped]==occupied)&(abs(p[:,2]-p[middle[mapped],2])<=.175)
        candidates=np.flatnonzero(valid)
        rank=candidates[np.lexsort((-p[candidates,7],occupied[candidates]))]
        chosen=rank[np.r_[True,np.diff(occupied[rank])!=0]]
        middle[slot[chosen]]=chosen
    z=np.full((7,ncell),np.nan,np.float32)
    z[:,cells]=p[middle][:,[2,3,4,5,6,0,1]].T
    mixed=np.zeros(ncell,bool)
    mixed[cells]=quartile_span>.35
    return z.reshape(7,*cap.shape),mixed.reshape(cap.shape)


def solve(args):
    start=time.monotonic()
    cap=Capture(args.scan,args.datum_from,args.area_mm,args.area_margin,args.cell_mm)
    pairs=camera_pairs(cap.O)
    args.out.mkdir(parents=True,exist_ok=False)
    diagnostics=[]
    for i,pair in enumerate(pairs):
        samples=epipolar_pair(cap,pair,args.below,args.above)
        (z,cc,gap,lr,sens,sx,sy),mixed=bin_pair(cap,samples,args.below,args.above)
        depth,error,back,shift,_=verify_pair(cap,pair,z,None if args.cell_center else np.stack((sx,sy),axis=-1))
        record=dict(sample_x=sx,sample_y=sy,raw_z=z,triangulated_z=depth,score=cc,gap=gap,left_right_px=lr,
                    mixed=mixed,reprojection_px=error,reverse_px=back,shift_mm=shift,sensitivity=sens)
        np.savez_compressed(args.out/f"pair_{i:02}.npz",**record)
        diagnostics.append(record)
        print(pair,len(samples),np.isfinite(z).sum(),flush=True)
    compare_stages(cap,pairs,diagnostics,args,start)


def compare_stages(cap,pairs,records,args,start):
    g={k:np.stack([r[k] for r in records]) for k in records[0]}
    raw=g["raw_z"]
    depth=g["triangulated_z"]
    ok=(np.abs(raw-depth)<=.35)&(g["reprojection_px"]<=.55)&(g["reverse_px"]<=.55)&(g["shift_mm"]<=cap.pitch/2)
    sigma=g["sensitivity"]*np.maximum(.20,g["reprojection_px"])
    for name,values,precision in [("reciprocal",raw,np.maximum(.2*g["sensitivity"],.02)),
                                  ("verified",np.where(ok&~g["mixed"],depth,np.nan),sigma)]:
        fit,support,spread,confirmed=fuse(values,precision,.35,3)
        blocked=foreground_occlusions(cap,fit,confirmed,values,pairs,.35)
        active=np.where(blocked,np.nan,values)
        fit,support,spread,confirmed=fuse(active,precision,.35,3)
        count=np.isfinite(active).sum(axis=0)
        conflict=(count>=2)&(support.sum(axis=0)*2<=count)
        occluded=blocked.any(axis=0)&(count==0)
        uncertain=(count>0)&~confirmed&~conflict
        # Visibility/texture diagnostic on the common reference plane; only report no_texture
        # if no image has contrast anywhere along this cell's searched vertical segment.
        if name=="reciprocal":
            field=np.zeros(cap.shape,bool); texture=np.zeros_like(field)
            for v,img in enumerate(cap.images):
                img=img.astype(np.float32)
                var=cv2.blur(img*img,(7,7))-cv2.blur(img,(7,7))**2
                usable=cv2.erode((cap.valid&(img<250)).astype(np.uint8),np.ones((7,7),np.uint8))
                for level in np.arange(-args.below,args.above+.01,.25):
                    uv=np.rint(cap.project(v,cap.world(cap.xy,level))).astype(int)
                    x,y=uv[...,0],uv[...,1]
                    inside=(x>=0)&(x<img.shape[1])&(y>=0)&(y<img.shape[0])
                    x=x.clip(0,img.shape[1]-1);y=y.clip(0,img.shape[0]-1)
                    inside &= usable[y,x]>0
                    field |= inside
                    texture |= inside & (var[y,x]>4)
        no_texture=field&~texture&~(confirmed|conflict|uncertain|occluded)
        directory=args.out if name=="verified" else args.out/name
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/"area.npz",height_mm=fit,confirmed=confirmed,measured_confirmed=confirmed,
            measured_uncertain=uncertain,conflict=conflict,occluded=occluded,no_texture=no_texture,
            no_data=~(confirmed|uncertain|conflict|occluded|no_texture),filled_or_inferred=np.zeros(cap.shape,bool),
            spread_mm=spread,votes=count,support=support,view_pairs=pairs,blocked_pairs=blocked,
            geometric_field=field,texture_evidence=texture,xy_origin=cap.home+cap.origin,cell_mm=cap.pitch,
            precision_proxy_mm=1/np.sqrt(np.maximum(np.where(support,1/np.maximum(precision,.02)**2,0).sum(axis=0),1e-12)))
        params=dict(engine=name,seconds=time.monotonic()-start,confirmed=int(confirmed.sum()),cells=fit.size,
                    normal=cap.n.tolist(),distance=cap.d,ref_position=cap.home.tolist(),pairs=pairs,
                    scan=str(args.scan),datum=str(args.datum_from),cell_mm=cap.pitch,area_mm=args.area_mm,
                    area_margin=args.area_margin,below=args.below,above=args.above,agree_mm=.35,
                    opencv=cv2.__version__,numpy=np.__version__,references="disjoint original frame pairs")
        (directory/"area.json").write_text(json.dumps(params,indent=2),encoding="utf8")
        image=np.full((*cap.shape,3),40,np.uint8)
        image[uncertain]=(145,145,145); image[conflict]=(40,90,210); image[confirmed]=(85,165,65)
        if not cv2.imwrite(str(directory/"coverage.png"),image): raise OSError("Failed to write coverage")
        print(name,int(confirmed.sum()),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan",type=Path)
    parser.add_argument("--datum-from",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--cell-center",action="store_true",help="Replicate the original cell-centre verifier")
    parser.add_argument("--area-mm",type=float,nargs=2,default=[68,47])
    parser.add_argument("--area-margin",type=float,default=7)
    parser.add_argument("--cell-mm",type=float,default=.5)
    parser.add_argument("--below",type=float,default=18)
    parser.add_argument("--above",type=float,default=30)
    solve(parser.parse_args())
