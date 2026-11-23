"""Rebuild saved frames offline into a NEW folder, preserving the original scan."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'z')]
from common.mosaic import Mosaic
from common.registration import register
from stereo import Rectifier


def rebuild(folder, out, ppm=None, fused=True):
    folder, out = Path(folder).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    meta=json.loads((folder/'mosaic.json').read_text(encoding='utf-8'))
    K=np.array(meta['camera_matrix'],float);D=np.array(meta['dist_coeffs'],float)
    model=meta.get('model')
    if model is None:
        path=ROOT/'calib'/meta['calibration']
        cal=json.loads(path.read_text(encoding='utf-8'))
        if not np.allclose(cal['camera_matrix'],K):raise ValueError('Calibration changed; cannot infer model')
        model=cal['model']
    ppm=meta['ppm'] if ppm is None else ppm
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError('ppm must be positive')
    m=Mosaic(meta['polygon'],ppm,blend_lighting=fused)
    records = meta['records']
    meta.pop('registration', None)
    meta.pop('tile_index', None)
    scale_map = None
    if fused:
        offsets, scale, meta['registration'] = register(folder, meta, K, D, model)
        grid_to_pixel = np.eye(2,3)*scale.spacing*ppm
        grid_to_pixel[:,2] = (scale.origin-m.origin)*ppm
        scale_map = cv2.warpAffine(scale.field, grid_to_pixel, m.size)
    else:
        offsets = np.zeros((len(records), 2))
    owner = np.zeros_like(m.region, dtype=np.uint16) if fused else None
    rect=None
    for i,r in enumerate(meta['records']):
        src=(folder/r['file']).resolve()
        raw=cv2.imread(str(src))
        if raw is None:raise ValueError(f'Cannot read {src}')
        if rect is None:rect=Rectifier(K,D,raw.shape[1::-1],model)
        H = np.array(r['H'], float)
        H[:2] += offsets[i, :, None]*H[2]
        m.add(raw,rect.valid,H,r['position']+offsets[i],raw=raw,K=K,D=D,model=model,diagnose=False,
              owner=owner,tile_id=i+1,scale_map=scale_map)
        r['file'] = os.path.relpath(src, out) if src.drive == out.drive else str(src)
        r['offset_mm'] = offsets[i].tolist()
        r['fused_position'] = (r['position']+offsets[i]).tolist()
    blend = 'single-source-detail-blended-lighting' if fused else 'central-source-single-resample'
    meta.update(ppm=ppm,model=model,blend=blend,source_scan=str(folder),
                coverage_fraction=float(np.count_nonzero(m.weight)/np.count_nonzero(m.region)))
    images = {'mosaic.png': m.image(), 'coverage.png': (m.weight>0).astype(np.uint8)*255}
    if fused:
        meta['tile_index'] = 'tile_index.png'
        images['tile_index.png'] = owner
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp)/'scan'
        stage.mkdir()
        for filename, image in images.items():
            if not cv2.imwrite(str(stage/filename), image):
                raise OSError(f'Cannot write {filename}')
        (stage/'mosaic.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
        stage.rename(out)
    return out, meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path)
    ap.add_argument('--ppm',type=float,default=None)
    ap.add_argument('--grbl-only', action='store_true', help='use recorded GRBL poses only')
    ap.add_argument('--out', type=Path, help='new output directory')
    args=ap.parse_args()
    out = args.out or args.folder/('sharp' if args.grbl_only else 'fused')
    folder, meta = rebuild(args.folder, out, args.ppm, not args.grbl_only)
    print(folder)
    if 'registration' in meta:
        print(json.dumps(meta['registration']['validation'], indent=2))


if __name__=='__main__':main()
