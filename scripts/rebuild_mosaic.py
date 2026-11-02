"""Rebuild saved frames offline into a NEW folder, preserving the original scan."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'z')]
from common.mosaic import Mosaic
from stereo import Rectifier


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path)
    ap.add_argument('--ppm',type=float,default=None)
    args=ap.parse_args()
    meta=json.loads((args.folder/'mosaic.json').read_text(encoding='utf-8'))
    out=args.folder/'sharp'
    if out.exists():raise SystemExit(f'Output already exists: {out}')
    out.mkdir()
    K=np.array(meta['camera_matrix'],float);D=np.array(meta['dist_coeffs'],float)
    model=meta.get('model')
    if model is None:
        path=ROOT/'calib'/meta['calibration']
        cal=json.loads(path.read_text(encoding='utf-8'))
        if not np.allclose(cal['camera_matrix'],K):raise ValueError('Calibration changed; cannot infer model')
        model=cal['model']
    ppm=args.ppm or meta['ppm']
    m=Mosaic(meta['polygon'],ppm)
    rect=None
    for i,r in enumerate(meta['records']):
        src=(args.folder/r['file']).resolve()
        raw=cv2.imread(str(src))
        if raw is None:raise ValueError(f'Cannot read {src}')
        if rect is None:rect=Rectifier(K,D,raw.shape[1::-1],model)
        m.add(rect(raw),rect.valid,np.array(r['H']),r['position'],raw=raw,K=K,D=D,model=model,diagnose=False)
        r['file']=str(src)
        print(f'{i+1}/{len(meta["records"])}',flush=True)
    meta.update(ppm=ppm,model=model,blend='central-source-single-resample',source_scan=str(args.folder.resolve()))
    cv2.imwrite(str(out/'mosaic.png'),m.image())
    cv2.imwrite(str(out/'coverage.png'),(m.weight>0).astype(np.uint8)*255)
    (out/'mosaic.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(out.resolve())


if __name__=='__main__':main()
