"""Offline camera/machine disagreement report. Does not alter motion or calibration."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folders',nargs='+',type=Path)
    args=ap.parse_args()
    for folder in args.folders:
        m=json.loads((folder/'mosaic.json').read_text(encoding='utf-8'))
        edges=m.get('pair_errors',[])
        good=[e for e in edges if e.get('valid')]
        print(f'{folder.name}: complete={m["complete"]}, coverage={m["coverage_fraction"]:.1%}, accepted pairs={len(good)}/{len(edges)}')
        print('closure (command mm):',m.get('closure_command_mm','unavailable'))
        if not good:continue
        err=np.array([e['visual_minus_command_mm'] for e in good])
        lengths=np.array([np.linalg.norm(np.array(e['position'])-e['previous_position']) for e in good])
        print('median discrepancy XY (command mm):',np.median(err,axis=0).round(4))
        print('pair discrepancy RMS / p95 (command mm):',round(float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),4),
              round(float(np.percentile(np.linalg.norm(err,axis=1),95)),4))
        moving=lengths>1
        if moving.any():print('median relative discrepancy (%):',round(float(np.median(np.linalg.norm(err[moving],axis=1)/lengths[moving])*100),3))
        print('These are disagreements, not absolute accuracy. No compensation fitted or applied.')


if __name__=='__main__':main()
