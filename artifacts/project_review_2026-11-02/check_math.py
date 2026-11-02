"""Offline mathematical checks, no hardware and no edits to project modules."""
from pathlib import Path
import sys,json
import numpy as np
import cv2
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'z'))
from stereo import triangulate,plane_homography,plane_sweep,points_from_heights,under_camera
from common.geometry import undistort_points
K=np.array([[1100.,0,960],[0,1100,540],[0,0,1]])
Os=[np.array([x,y,0.]) for x,y in [(0,0),(12,0),(0,12),(-12,-12)]]
X=np.array([3.,-5.,77.]); pixels=[(K@(X-o))[:2]/(X-o)[2] for o in Os]
results=[]
def check(name,error,tol):
    assert error<tol,(name,error)
    results.append(dict(name=name,error=float(error),tolerance=tol))
est,rms=triangulate(pixels,Os,K);check('triangulation_ideal_3d_mm',np.linalg.norm(est-X),1e-7)
est2,_=triangulate(pixels,[1.01*o for o in Os],K);check('baseline_scale_gauge_1pct',np.linalg.norm(est2-1.01*X),1e-7)
H=plane_homography(K,Os[1],np.array([0,0,1.]),77)
p=H@np.r_[pixels[0],1];check('plane_homography_px',np.linalg.norm(p[:2]/p[2]-pixels[1]),1e-7)
c=json.loads((ROOT/'calib/calib_cam0_1920x1080_pinhole_rational.json').read_text())
kc=np.array(c['camera_matrix']);dc=np.array(c['dist_coeffs'])
obj=np.array([[x,y,1.] for x in [-.5,0,.5] for y in [-.3,0,.3]])
pix=cv2.projectPoints(obj,np.zeros(3),np.zeros(3),kc,dc)[0]
rest=undistort_points(pix,kc,dc,c['model'],P=None)
check('undistortion_roundtrip_normalized',np.max(np.abs(rest-obj[:,:2])),1e-6)
V=np.array(c['machine_check']['motion']['V_cam']);q=under_camera(V,X);check('under_camera_xy_mm',np.linalg.norm((X-V@q)[:2]),1e-7)
# Plane-sweep synthetic scene: known depth 80, guess 82, h=2, two baselines.
rng=np.random.default_rng(7);ref=cv2.GaussianBlur(rng.uniform(0,255,(120,160)).astype('float32'),(3,3),.6)
ks=np.array([[160.,0,80],[0,160.,60],[0,0,1.]])
origins=[np.array([5.,0,0]),np.array([0,5.,0])]
others=[cv2.warpPerspective(ref,plane_homography(ks,o,np.array([0,0,1.]),80),(160,120)) for o in origins]
h,score,valid=plane_sweep(ref,others,origins,ks,np.array([0,0,1.]),82,np.arange(0,4.01,.25))
check('plane_sweep_median_height_mm',abs(float(np.nanmedian(h[20:-20,20:-20]))-2),.02)
(OUT/'math_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
