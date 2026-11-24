"""Evaluate signed XY height grids on identical archived masks; controls are used only here."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fill_area import footprint
from stereo import half, Rectifier


def quant(values, p):
    return float(np.percentile(values,p)) if len(values) else None


def evaluation_masks(scan, archive, base):
    meta=json.loads((scan/'scan.json').read_text(encoding='utf8'))
    src=cv2.imread(str(scan/f"frame_{meta['ref_index']:02}.png"),cv2.IMREAD_UNCHANGED)
    K=np.asarray(meta['camera_matrix'])
    image,K=half(Rectifier(K,np.asarray(meta['dist_coeffs']),src.shape[::-1],meta['model'])(src.astype(np.float32)/256),K)
    with np.load(archive) as a:
        origin=a['xy_origin']; pitch=float(a['cell_mm'])
        masks=dict(zip(['bridge','left','right','groove'],a['region_masks'],strict=True))
    indices=np.indices(masks['bridge'].shape)[::-1].transpose(1,2,0)
    xy=origin-np.asarray(base['ref_position'])+pitch*indices
    def rect(x,y,w,h,z):
        poly=np.array([[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]],np.int32)
        return footprint(xy,np.full(xy.shape[:2],z),poly,K,np.asarray(meta['V_cam']),
                         np.asarray(base['normal']),base['distance'],image.shape)
    masks['top']=masks['bridge']|masks['left']|masks['right']
    masks['historical_top']=rect(286,184,384,38,9.925)
    masks['table']=rect(360,272,190,46,0)
    masks['historical_groove']=rect(319,354,296,36,-12.65)
    masks['groove_central_band']=rect(375,371,175,8,-12.65)
    masks['annotated']=masks['top']|masks['groove']|masks['table']
    return masks,xy,origin,pitch


def stats(z,confirmed,observed,conflict,mask,expected):
    c=mask&confirmed
    values=z[c]
    n=int(mask.sum())
    measured=int((mask&observed).sum())
    out=dict(cells=n,observed=measured,confirmed=len(values),conflicts=int((conflict&mask).sum()),
             measured_pct=100*measured/n,confirmed_pct=100*len(values)/n,
             conflict_pct=100*int((conflict&mask).sum())/n,
             confirmed_of_measured_pct=100*len(values)/measured if measured else None,
             median_z_mm=quant(values,50),z_p05_mm=quant(values,5),z_p95_mm=quant(values,95),
             mad_mm=quant(abs(values-np.median(values)),50) if len(values) else None)
    if expected:
        err=abs(values-np.clip(values,*expected))
        out.update(median_error_mm=quant(err,50),p95_error_mm=quant(err,95),
                   within_010=int((err<=.1).sum()),gross_gt_1mm=int((err>1).sum()),
                   gross_pct=float(100*(err>1).mean()) if len(err) else None)
    return out


def extent(points,axis,pitch):
    return float(np.ptp(points@axis)+pitch) if len(points)>1 else None


def evaluate(directory,masks,xy,origin,pitch,base):
    params=json.loads((directory/'area.json').read_text(encoding='utf8'))
    ref_position=params.get('ref_position')
    if ref_position is None:
        ref_position=json.loads(Path(params['datum']).read_text(encoding='utf8'))['ref_position']
    if not (np.allclose(params['normal'],base['normal']) and
            np.isclose(params['distance'],base['distance']) and np.allclose(ref_position,base['ref_position'])):
        raise ValueError(f'Different plane datum: {directory}')
    with np.load(directory/'area.npz') as source:
        d=dict(source)
    z=d['height_mm']; confirmed=d['confirmed']; obs=d['votes']>0
    if z.shape!=masks['top'].shape or not np.allclose(origin,d['xy_origin']) or pitch!=float(d['cell_mm']):
        raise ValueError(f'Different XY grid: {directory}')
    conflict=d.get('conflict',(d['votes']>=2)&~confirmed)
    result={}
    for name,mask in dict(grid=np.ones_like(confirmed),**masks).items():
        ref=None if name in ('annotated','grid') else ((-12.65,-12.65) if 'groove' in name else (0.,0.) if name=='table' else (9.85,10.))
        result[name]=stats(z,confirmed,obs,conflict,mask,ref)
        if 'groove' in name:
            vals=z[confirmed&mask]
            result[name].update(positive=int((vals>0).sum()),positive_pct=float(100*(vals>0).mean()) if len(vals) else None)
    result['inferred']=int(d.get('filled_or_inferred',np.zeros_like(confirmed)).sum())
    levels=[-18,-15,-10,-1,1,8,11,20,25,29,31]
    result['histogram']={'edges_mm':levels,'confirmed_counts':np.histogram(z[confirmed],levels)[0].tolist(),
        'above_20':int((z[confirmed]>20).sum()),'above_25':int((z[confirmed]>25).sum())}
    result['status']={key:int(d[key].sum()) for key in
            ('measured_confirmed','measured_uncertain','conflict','occluded','no_texture','no_data','filled_or_inferred') if key in d}
    if 'geometric_field' in d:
        result['field']=stats(z,confirmed,obs,conflict,d['geometric_field'],None)
        result['texture_proxy']=stats(z,confirmed,obs,conflict,d['texture_evidence'],None)
    if 'support' in d:
        support=d['support']
        counts=support[:,confirmed].sum(axis=0)*2
        result['independent_camera_counts']={int(k):int((counts==k).sum()) for k in np.unique(counts)}
        files=list(directory.glob('pair_*.npz')) or list(directory.parent.glob('pair_*.npz'))
        files.sort(key=lambda f:int(f.stem.split('_')[-1]))
        if len(files)==len(support):
            residuals=[]
            stages={}
            for i,file in enumerate(files):
                with np.load(file) as evidence:
                    residuals.extend(evidence['reprojection_px'][support[i]&confirmed].tolist())
                    for key in ('proposal_z','raw_z','height'):
                        if key in evidence:
                            if key not in stages:
                                stages[key]=np.zeros(z.shape,np.uint16)
                            stages[key]+=np.isfinite(evidence[key])
            result['pair_evidence']={name:{key:{'any':int((layer[mask]>0).sum()),
                          'at_least_three_pairs':int((layer[mask]>=3).sum())}
                          for key,layer in stages.items()} for name,mask in masks.items()}
            r=np.asarray(residuals);r=r[np.isfinite(r)]
            result['native_residual_px']={'median':quant(r,50),'p95':quant(r,95)}
    expected_top=(z>=9.35)&(z<=10.50)
    shape={}
    for key in ('bridge','left','right'):
        mask=masks[key]
        points=xy[mask]
        _,vecs=np.linalg.eigh(np.cov(points.T))
        supported=xy[confirmed&mask&expected_top]
        shape[key]={'annotation_extent_long_short_mm':[extent(points,v,pitch) for v in vecs.T[::-1]],
                    'measured_extent_long_short_mm':[extent(supported,v,pitch) for v in vecs.T[::-1]],
                    'top_consistent_cells':len(supported)}
    _,basis=np.linalg.eigh(np.cov(xy[masks['bridge']].T))
    xyz_top=xy[confirmed&masks['top']&expected_top]
    shape['outline']={'nominal_width_length_mm':[67.67,46.67],
                      'annotation_width_length_mm':[extent(xy[masks['top']],v,pitch) for v in basis.T[::-1]],
                      'supported_width_length_mm':[extent(xyz_top,v,pitch) for v in basis.T[::-1]]}
    edges=masks['top'].astype(np.uint8); kernel=np.ones((3,3),np.uint8)
    outside=cv2.dilate(edges,kernel).astype(bool)&~masks['top']
    inside=masks['top']&~cv2.erode(edges,kernel).astype(bool)
    shape['inner_edge']={'confirmed':int((inside&confirmed).sum()),'top_consistent':int((inside&confirmed&expected_top).sum())}
    shape['outer_edge']={'confirmed':int((outside&confirmed).sum()),'top_consistent':int((outside&confirmed&expected_top).sum())}
    result['xy']=shape
    result['seconds']=params.get('seconds')
    return result,np.where(confirmed,z,np.nan)


def charts(results,grids,origin,pitch,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    names=list(grids)
    fig,axs=plt.subplots((len(names)+2)//3,3,figsize=(15,3.55*((len(names)+2)//3)),layout='constrained',squeeze=False)
    cmap=plt.get_cmap('RdYlBu_r').copy()
    cmap.set_bad('#292e34')
    for ax in axs.flat:
        ax.set_visible(False)
    for ax,name in zip(axs.flat,names):
        ax.set_visible(True); grid=grids[name]; ny,nx=grid.shape
        bounds=[origin[0]-pitch/2,origin[0]+(nx-.5)*pitch,origin[1]-pitch/2,origin[1]+(ny-.5)*pitch]
        im=ax.imshow(grid,origin='lower',extent=bounds,cmap=cmap,norm=Normalize(-15,15),interpolation='nearest')
        ax.set_title(f"{name}   ·   {results[name]['grid']['confirmed']:,}")
        ax.set_xlabel('X, мм'); ax.set_ylabel('Y, мм')
    fig.colorbar(im,ax=[ax for ax in axs.flat if ax.get_visible()],label='Высота от стола, мм',shrink=.8,extend='both')
    fig.savefig(root/'heights.png',dpi=145);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scan',type=Path); p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--maps',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    base=json.loads((args.maps[0]/'area.json').read_text(encoding='utf8'))
    masks,xy,origin,pitch=evaluation_masks(args.scan,args.masks,base)
    results={}; grids={}
    for path in args.maps:
        name=path.name
        if name in results:
            raise ValueError(f'Duplicate output label {name}')
        results[name],grids[name]=evaluate(path,masks,xy,origin,pitch,base)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
    common={}
    latest=args.maps[-1].name
    for other,other_grid in grids.items():
        if other==latest:
            continue
        comparison={}
        for key in ('top','bridge','left','right','table','groove'):
            mask=masks[key]&np.isfinite(other_grid)&np.isfinite(grids[latest])
            target=(-12.65,-12.65) if key=='groove' else (0.,0.) if key=='table' else (9.85,10.)
            def errors(grid):
                val=grid[mask]
                return quant(abs(val-np.clip(val,*target)),95)
            comparison[key]={'common_cells':int(mask.sum()),'previous_p95_mm':errors(other_grid),
                             'last_p95_mm':errors(grids[latest])}
        common[other]=comparison
    (args.out/'paired.json').write_text(json.dumps(dict(last=latest,comparisons=common),indent=2,allow_nan=False),encoding='utf8')
    charts(results,grids,origin,pitch,args.out)


if __name__=='__main__':
    main()
