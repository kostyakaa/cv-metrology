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
        if m.get('trial'):
            records = m['records']
            settled = [np.linalg.norm(r['trial_marker_error_mm']) for r in records if 'trial_marker_error_mm' in r]
            observed = 0
            for i, (a,b) in enumerate(zip(records,records[1:]),1):
                leg = b.get('motion_from_previous',{})
                grbl = np.subtract(b['position'],a['position'])
                # Count actual intermediate reports, excluding stationary samples at either endpoint.
                interior = [s for s in leg.get('status_samples',[]) if 'frame_s' in s
                            and np.linalg.norm(np.subtract(s['position'],a['position'])) > .1
                            and np.linalg.norm(np.subtract(s['position'],b['position'])) > .1]
                optical = leg.get('delta_mm')
                err = np.linalg.norm(np.subtract(optical,grbl)) if optical is not None else None
                good_leg = (bool(interior) and leg.get('valid') and leg.get('timing_consistent')
                            and err is not None and err <= .5)
                observed += int(bool(good_leg))
                print(f'Leg {i}: GRBL {grbl.round(3)} optical {np.round(optical,3) if optical is not None else "lost"}'
                      f' mm, intermediate reports {len(interior)}, within 0.5 mm: {bool(good_leg)}')
            print(f'Trial: verified motion legs {observed}/4, observed stops {len(settled)}/5')
            if settled: print(f'Max marker discrepancy: {max(settled):.3f} mm')
        reg=m.get('registration')
        if reg:
            pairs=reg['pairs']
            path=reg['optical_trajectory']
            good_steps=sum(a['segment']==b['segment'] for a,b in zip(path,path[1:]))
            print(f"Optical steps: {good_steps}/{len(path)-1}; fused pairs: {sum(e['accepted'] for e in pairs)}/{len(pairs)}")
            if reg.get('optical_positions'):
                components = {p['origin_frame'] for p in reg['optical_positions']}
                print(f'Camera-only XY: {len(components)} connected component(s); each origin is a scan frame')
            for name, stats in reg['validation'].items():
                if stats:
                    print(f"Held-out {name}: n={stats['count']} RMS={stats['rms_mm']:.3f} p95={stats['p95_mm']:.3f} mm")
        if not good:continue
        err=np.array([e['visual_minus_command_mm'] for e in good])
        lengths=np.array([np.linalg.norm(np.array(e['position'])-e['previous_position']) for e in good])
        print('median discrepancy XY (command mm):',np.median(err,axis=0).round(4))
        print('pair discrepancy RMS / p95 (command mm):',round(float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),4),
              round(float(np.percentile(np.linalg.norm(err,axis=1),95)),4))
        moving=lengths>1
        if moving.any():print('median relative discrepancy (%):',round(float(np.median(np.linalg.norm(err[moving],axis=1)/lengths[moving])*100),3))
        print('These sequential-pair statistics describe the original GRBL projection.')


if __name__=='__main__':main()
