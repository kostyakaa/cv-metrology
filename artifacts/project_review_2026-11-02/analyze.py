"""Read-only project audit; outputs stay beside this script. No camera/serial calls."""
from pathlib import Path
import sys, json, hashlib, ast, re
import numpy as np
import cv2
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from common.geometry import undistort_points, machine_motion
from common.markers import make_charuco_board
def read(p): return json.loads(p.read_text(encoding='utf-8'))
inventory=[]
for folder in ['common','scripts','z']:
    for p in sorted((ROOT/folder).glob('*.py')):
        s=p.read_text(encoding='utf-8'); tree=ast.parse(s)
        inventory.append(dict(path=p.relative_to(ROOT).as_posix(),lines=len(s.splitlines()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),functions=[dict(name=n.name,line=n.lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]))
calibs={p.name:read(p) for p in (ROOT/'calib').glob('calib*.json')}
c=calibs['calib_cam0_1920x1080_pinhole_rational.json']; mc=c['machine_check']
tf=read(ROOT/'calib/table_frame.json'); tp=read(ROOT/'calib/table_plane.json')
sx,sy=mc['real_per_cmd_x'],mc['real_per_cmd_y']
summary=dict(source_files=len(inventory),source_lines=sum(x['lines'] for x in inventory),opencv=cv2.__version__,calibration={k:dict(rms=v['rms'],reliable=v['reliable'],fx=v['camera_matrix'][0][0],views=len(v['per_view_rms'])) for k,v in calibs.items()},scale_bias_percent=[100*(sx-1),100*(sy-1)],command_over_true_percent=[100*(1/sx-1),100*(1/sy-1)],error_at_100mm=[100*(sx-1),100*(sy-1)],suggested_steps=[800/sx,800/sy],closure_norm_mm=float(np.linalg.norm(tf['closure_mm'])),closure_over_perimeter_percent=100*float(np.linalg.norm(tf['closure_mm']))/sum(tf['frame']['edges_mm'].values()),mm_per_pixel=tp['plane']['d']/c['camera_matrix'][0][0],backlash_mm=tf['backlash_mm'])
summary['scans']=[]
for p in sorted((ROOT/'scans').glob('*/scan.json')):
    m=read(p); r=read(p.with_name('result.json')); objs=r['objects']
    summary['scans'].append(dict(name=p.parent.name,frames=len(m['positions']),baseline=m['baseline'],reference=r['ref_kind'],offset=r['global_offset_mm'],objects=len(objs),near_29mm=sum(o['height_top']>29 for o in objs)))
log=(ROOT/'log.txt').read_text(encoding='utf-8')
summary['log_coverage_percent']=[int(x) for x in re.findall(r'определилось (\d+)%',log)]
summary['log_points']=[dict(h=float(a),sigma=float(b),reliable='НЕНАДЁЖНО' not in tail) for a,b,tail in re.findall(r'точка: ([\d.-]+) ±([\d.-]+) мм([^\n]*)',log)]
summary['depth_sensitivity']=[dict(baseline=B,sigma_disparity_px=sd,sigma_z_mm=tp['plane']['d']**2/(c['camera_matrix'][0][0]*B)*sd) for B in [6,12,24] for sd in [.05,.1,.2]]
# Re-detect saved calibration frames (single saved frame, not original averaged detections).
folder=ROOT/'calib/frames_cam0_1920x1080_20261019-153432'; mode=read(folder/'mode.json')
board=make_charuco_board(**{k:v for k,v in mode['board'].items() if k!='name'})
det=cv2.aruco.CharucoDetector(board); views=[]; offsets=[]; names=[]
for p in sorted(folder.glob('view_*.png')):
    gray=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    corners,ids,_,_=det.detectBoard(gray)
    if ids is None or len(ids)<8 or board.checkCharucoCornersCollinear(ids): continue
    obj,img=board.matchImagePoints(corners,ids)
    views.append((obj.reshape(-1,3),img.reshape(-1,2))); offsets.append(mode.get('machine_offsets',{}).get(p.name)); names.append(p.name)
K=np.array(c['camera_matrix']); D=np.array(c['dist_coeffs'])
summary['redetected_motion']=machine_motion(views,offsets,K,D,c['model'])
seen=[]; moves=[]; grid_names=[]
for (obj,img),off,name in zip(views,offsets,names):
    if off is None: continue
    und=undistort_points(img,K,D,c['model'])
    H,_=cv2.findHomography(und,obj[:,:2].astype(float),0)
    q=H@np.r_[K[:2,2],1.]; seen.append(q[:2]/q[2]);moves.append(off);grid_names.append(name)
X=np.c_[moves,np.ones(len(moves))];Y=np.array(seen)
coef=np.linalg.lstsq(X,Y,rcond=None)[0];res=Y-X@coef
loo=[]
for i in range(len(X)):
    keep=np.arange(len(X))!=i
    loo.append(Y[i]-X[i]@np.linalg.lstsq(X[keep],Y[keep],rcond=None)[0])
summary['grid_analysis']=dict(n=len(X),fit_rms_mm=float(np.sqrt(np.mean(np.sum(res**2,axis=1)))),leave_one_out_rms_mm=float(np.sqrt(np.mean(np.sum(np.array(loo)**2,axis=1)))),max_residual_mm=float(np.max(np.linalg.norm(res,axis=1))),note='K,D fixed from same dataset; LOO validates affine motion only, not independent accuracy.')
(OUT/'grid_residuals.json').write_text(json.dumps([dict(frame=n,command=o,board=q.tolist(),residual=r.tolist()) for n,o,q,r in zip(grid_names,moves,Y,res)],indent=2),encoding='utf-8')
(OUT/'source_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# Scientific figure: affine residuals, not an error map certified by an external reference.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,ax=plt.subplots(figsize=(8,5)); moves=np.array(moves)
q=ax.quiver(moves[:,0],moves[:,1],res[:,0],res[:,1],np.linalg.norm(res,axis=1)*1000,angles='xy',scale_units='xy',scale=.006,cmap='viridis')
ax.set(xlabel='Команда X, мм',ylabel='Команда Y, мм',title='Остатки аффинной модели по сохранённым кадрам')
ax.grid(alpha=.2);fig.colorbar(q,ax=ax,label='Норма остатка, мкм');fig.tight_layout();fig.savefig(OUT/'grid_residuals.png',dpi=160);plt.close(fig)
# Contact sheet for data inspection.
from PIL import Image, ImageDraw
sheet=Image.new('RGB',(960,3*220),'#15252d'); draw=ImageDraw.Draw(sheet)
for i,p in enumerate(sorted((ROOT/'scans').glob('*/heights.png'))):
    im=Image.open(p).convert('RGB');im.thumbnail((312,185));x=(i%3)*320;y=(i//3)*220
    sheet.paste(im,(x,y+24));draw.text((x+5,y+5),p.parent.name,fill='white')
sheet.save(OUT/'scan_contact_sheet.jpg')
print(json.dumps({k:v for k,v in summary.items() if k not in ['log_points','scans','redetected_motion']},ensure_ascii=False,indent=2))
