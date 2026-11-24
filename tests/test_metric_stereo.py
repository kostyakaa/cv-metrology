"""Offline regression tests with perspective images of known metric surfaces."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'z'))
from epipolar_height import axes, bin_pair
from fit_stereo_axes import residual
from network_height import adjacent_pairs, consensus, independent, local_peak, measure_pair
from pair_geometry import Capture, triangulate_batch
from pair_height import foreground_occlusions


def capture():
    cap=Capture.__new__(Capture)
    cap.K=np.array([[240.,0,160],[0,240.,120],[0,0,1.]])
    cap.n=np.array([.025,-.015,1.]);cap.n/=np.linalg.norm(cap.n)
    cap.d=74.
    cap.V=np.array([[-1.,.02],[.01,1.],[.015,-.004]])
    cap.O=np.array([[0.,0,0],[8.,.12,.1]])
    cap.pitch=1.; cap.origin=np.array([-16.,-12.]); cap.shape=(25,33)
    cap.xy=cap.origin+np.indices(cap.shape)[::-1].transpose(1,2,0)*cap.pitch
    cap.valid=np.ones((240,320),bool)
    return cap


def render_plane(cap,z):
    rng=np.random.default_rng(700)
    texture=cv2.GaussianBlur(rng.integers(10,246,(1024,1024),dtype=np.uint8),(3,3),.6)
    yy,xx=np.indices(cap.valid.shape)
    rays=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(cap.K).T
    images=[]
    for O in cap.O:
        xyz=O+((cap.d-z-O@cap.n)/(rays@cap.n))[...,None]*rays
        uv=(xyz[...,:2]*4+512).astype(np.float32)
        images.append(cv2.remap(texture,uv,None,cv2.INTER_LINEAR))
    return images


class GeometryTests(unittest.TestCase):
    def test_signed_world_machine_grid_and_two_rays(self):
        c=capture()
        xy=np.array([[7.,-5.],[-8.,6.]])
        z=np.array([-8.1,11.3])
        xyz=c.world(xy,z)
        np.testing.assert_allclose(c.to_xy(xyz),xy,atol=1e-9)
        np.testing.assert_allclose(c.d-xyz@c.n,z)
        p,q=[c.project(i,xyz) for i in (0,1)]
        np.testing.assert_allclose(triangulate_batch(p,q,*c.O,c.K),xyz,atol=1e-9)
        R=axes(c.O[1]-c.O[0])
        np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-12)
        rays=[(xyz-o)@R.T for o in c.O]
        np.testing.assert_allclose(rays[0][:,1]/rays[0][:,2],rays[1][:,1]/rays[1][:,2])
        uv=[np.column_stack((p0,np.ones(len(p0))))@np.linalg.inv(c.K).T for p0 in (p,q)]
        np.testing.assert_allclose(residual(np.zeros(3),*uv,np.repeat((c.O[1:]-c.O[:1]),2,axis=0),c.K),0,atol=1e-10)

    def test_rotation_in_scan_axes_restores_depth_without_rescaling_baselines(self):
        c=capture()
        turn=np.array([.008,-.004,.006])
        rotation=cv2.Rodrigues(turn)[0]
        nominal=np.array([[12.,0,.08],[0,12.,.12],[-10.,12.,-.06],[12.,12.,.09]])
        actual=nominal@rotation.T
        xyz=c.world(np.array([[-14.,3.],[12.,6.],[0.,-8.],[6.,9.]]),np.array([4.,7.,-5.,3.]))
        p=(xyz@c.K.T); p=p[:,:2]/p[:,2,None]
        q=(xyz-actual)@c.K.T; q=q[:,:2]/q[:,2,None]
        ray=lambda uv:np.column_stack((uv,np.ones(len(uv))))@np.linalg.inv(c.K).T
        e=residual(np.zeros(3),ray(p),ray(q),nominal,c.K)
        self.assertGreater(np.median(abs(e)),.1)
        np.testing.assert_allclose(residual(turn,ray(p),ray(q),nominal,c.K),0,atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(nominal,axis=1),np.linalg.norm(actual,axis=1))
        for i,offset in enumerate(actual):
            point=triangulate_batch(p[i:i+1],q[i:i+1],np.zeros(3),offset,c.K)
            np.testing.assert_allclose(point[0],xyz[i],atol=1e-8)

    def test_real_stereo_pixels_on_positive_and_negative_planes(self):
        c=capture()
        c.O[1,0]=13.
        for z in (-4.2,7.3):
            c.images=render_plane(c,z)
            rec=measure_pair(c,(0,1),12,18,True)
            measured=rec['height'][np.isfinite(rec['height'])]
            self.assertGreater(len(measured),30)
            self.assertLess(abs(np.median(measured)-z),.12)
            self.assertLess(np.percentile(abs(measured-z),95),.35)

    def test_shorter_baseline_has_larger_depth_error_in_pixels(self):
        c=capture()
        c.images=render_plane(c,-4.2)
        rec=measure_pair(c,(0,1),12,18,True)
        ok=np.isfinite(rec['height'])
        # Sensitivity is mm per rectified disparity pixel; the aperture sets the metric limit.
        error_px=abs(rec['height'][ok]+4.2)/rec['sensitivity'][ok]
        self.assertGreater(len(error_px),30)
        self.assertLess(np.percentile(error_px,95),.22)

    def test_no_depth_from_constant_or_repeated_texture(self):
        constant=np.full((24,120),100,np.uint8)
        periodic=np.tile(np.random.default_rng(1).integers(0,255,(24,5),dtype=np.uint8),(1,24))
        for a in (constant,periodic):
            self.assertIsNone(local_peak(a,a,75,12,np.arange(5,45)))

    def test_bin_rejects_mixed_step_and_preserves_real_sample(self):
        c=capture()
        z=[0.,0.,0.,10.,10.,10.]
        points=np.array([[.01*i,0,h,.8,.3,.1,.15,i] for i,h in enumerate(z)])
        grid,mixed=bin_pair(c,points,18,30)
        self.assertTrue(mixed[12,16])
        self.assertIn(grid[0,12,16],z)
        self.assertIn(grid[5,12,16],points[:,0].astype(np.float32))

    def test_occluding_foreground_does_not_supply_evidence(self):
        c=capture();c.n=np.array([0.,0,1.]);c.V=np.eye(3)[:,:2]
        c.K=np.array([[80.,0,40],[0,80.,40],[0,0,1.]])
        c.images=[np.zeros((80,80),np.uint8)]*2
        c.xy=np.array([[[4.,0.],[5.,0.]]])
        c.d=50.; z=np.array([[10.,np.nan]])
        # In view 0, x=4 at depth 40 and x=5 at depth 50 project to the same pixel.
        candidates=np.array([[[10.,0.]]])
        blocked=foreground_occlusions(c,z,np.isfinite(z),candidates,[(0,1)],.35)
        self.assertTrue(blocked[0,0,1]); self.assertFalse(blocked[0,0,0])
        self.assertFalse(np.isfinite(z[0,1]))
        xy=c.xy[None].copy()
        xy[0,0,1,0]=5.4
        # A different real sample in the same cell projects clear of the foreground.
        blocked=foreground_occlusions(c,z,np.isfinite(z),candidates,[(0,1)],.35,xy,np.isfinite(z)[None])
        self.assertFalse(blocked[0,0,1])

    def test_camera_is_counted_once_and_competing_modes_rejected(self):
        pairs=[(0,1),(0,3),(0,5),(2,3),(4,5)]
        def solve(values,p):
            z=np.array(values,dtype=float)[:,None,None]
            return consensus(z,np.ones_like(z),p)
        self.assertFalse(solve([4,4.1,4.05],pairs[:3])[-1].item())
        result=solve([4,4.1,4.05,4.02,4.07],pairs)
        self.assertTrue(result[-1].item())
        chosen=np.flatnonzero(result[1][:,0,0])
        self.assertEqual(len({v for i in chosen for v in pairs[i]}),6)
        p=[(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
        self.assertFalse(solve([1,1.1,1.05,7,7.05,7.1],p)[-1].item())
        self.assertEqual(len(independent([(0,1),(0,3),(2,1)],[0,1,2])),2)

    def test_actual_positions_define_extended_pairs(self):
        p=np.array([[x,y] for y in range(5) for x in (range(4,-1,-1) if y%2 else range(5))],float)*11.664
        self.assertEqual(len(adjacent_pairs(p)),40)
        self.assertGreater(len(adjacent_pairs(p,True)),40)
        self.assertNotIn((4,5),adjacent_pairs(np.array([[x,y] for y in range(5) for x in range(5)])))


if __name__=='__main__':
    unittest.main()
