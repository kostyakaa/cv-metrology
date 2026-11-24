import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

from common.odometry import TileFeatures, motion_leg, reduced
from common.registration import adjust, optical_coordinates, projected_points, ProjectionScale

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from rebuild_mosaic import rebuild


class OpticalMotionTests(unittest.TestCase):
    def setUp(self):
        self.gray = cv2.GaussianBlur(np.random.default_rng(17).integers(
            0, 256, (384, 640), np.uint8), (3, 3), .7)
        self.K = np.array([[450., 0, 320], [0, 450, 192], [0, 0, 1]])
        self.D = np.zeros(5)

    def frame(self, dx, dy):
        return cv2.warpAffine(self.gray, np.float32([[1, 0, dx], [0, 1, dy]]), (640, 384))

    @staticmethod
    def record(x, y):
        H = [[.1, 0, x], [0, .1, y], [0, 0, 1]]
        return dict(position=[x,y], H=H)

    def test_camera_step_ignores_grbl_delta_and_integrates_signed_flow(self):
        a, b = self.record(10, 20), self.record(-4, 40)
        sift = cv2.SIFT_create()
        match = cv2.BFMatcher(cv2.NORM_L2)
        fa = TileFeatures(self.gray, self.K, self.D, 'pinhole', a, sift)
        fb = TileFeatures(self.frame(16,-8), self.K, self.D, 'pinhole', b, sift)
        np.testing.assert_allclose(fa.displacement(fb, match)['delta_mm'], [-1.6,.8], atol=.03)
        samples = [(k, reduced(self.frame(4*k,-2*k))) for k in range(5)]
        flow = motion_leg(samples, a, b, self.K, self.D, 'pinhole')
        self.assertTrue(flow['valid'])
        np.testing.assert_allclose(flow['delta_mm'], [-1.6,.8], atol=.03)
        status = [dict(status_s=1.5, position=[9.4,20.3]),dict(status_s=2.5,position=[9,20.5])]
        aligned = motion_leg(samples,a,self.record(8.4,20.8),self.K,self.D,'pinhole',status,60)
        self.assertTrue(aligned['timing_consistent'])
        np.testing.assert_allclose(aligned['steps'][1]['visual_minus_grbl_mm'], 0, atol=.03)
        status[0]['position'] = [80,80]
        self.assertFalse(motion_leg(samples,a,b,self.K,self.D,'pinhole',status,60)['timing_consistent'])
        samples[2] = (2, np.zeros_like(samples[2][1]))
        self.assertFalse(motion_leg(samples, a, b, self.K, self.D, 'pinhole')['valid'])

    def test_graph_keeps_unobserved_tile_and_signed_displacement(self):
        poses = np.array([[0.,0.], [10.,1.], [40.,6.]])
        edges = [dict(i=0, j=1, delta_mm=[10.5,.8], scatter_mm=.03)]
        offsets = adjust(poses, edges, [dict(i=0,offset=[0,0])])
        np.testing.assert_allclose(offsets[2], 0)
        np.testing.assert_allclose((poses+offsets)[1]-(poses+offsets)[0], [10.5,.8],atol=.05)

    def test_scale_requires_cross_axis_support(self):
        pos = np.array([[0.,0.],[10.,0.],[0.,10.]])
        support = (np.indices((12,12)).reshape(2,-1).T*4).tolist()
        polygon = [[0,0],[50,0],[50,50],[0,50]]
        def edge(j):
            return dict(i=0, j=j, grbl_delta_mm=pos[j],
                        tile=dict(delta_mm=pos[j]*1.02,local_support_mm=support))
        scale = ProjectionScale(polygon,pos,[edge(1)]*5,[])
        self.assertAlmostEqual(scale.at([24,24]), 0.)
        scale = ProjectionScale(polygon,pos,[edge(1),edge(2)]*5,[])
        np.testing.assert_allclose(projected_points(pos[0],np.array([24.,24.]),scale),np.array([24,24])/1.02,atol=.01)

    def test_camera_graph_bridges_missing_step_and_keeps_separate_origins(self):
        pairs = [dict(i=0,j=1),dict(i=0,j=2,tile=dict(delta_mm=[9,7])),
                 dict(i=1,j=2,tile=dict(delta_mm=[2,6]))]
        coords = optical_coordinates([{} for _ in range(4)], pairs, [{} for _ in range(4)])
        self.assertEqual([c['origin_frame'] for c in coords], [0,0,0,3])
        np.testing.assert_allclose([c['xy_mm'] for c in coords],[[0,0],[7,1],[9,7],[0,0]],atol=1e-8)

    def test_conflicting_and_mistimed_motion_do_not_join_optical_tracks(self):
        pairs = [dict(i=0, j=1), dict(i=1, j=2, tile=dict(delta_mm=[15, 0]),
                                       reason='endpoint and motion disagree')]
        records = [{}, dict(motion_from_previous=dict(valid=True, timing_consistent=False, delta_mm=[20,0])),
                   dict(motion_from_previous=dict(valid=True, timing_consistent=True, delta_mm=[17,0]))]
        coords = optical_coordinates([{}, {}, {}], pairs, records)
        self.assertEqual([c['origin_frame'] for c in coords], [0, 1, 2])

    def test_offline_blank_scan_preserves_pixels_and_reports_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            img = np.full((80,100,3),160,np.uint8)
            records = []
            for i in range(2):
                name = f'{i}.png'
                cv2.imwrite(str(folder/name),img)
                records.append(dict(self.record(i,0),file=name))
            (folder/'mosaic.json').write_text(json.dumps(dict(
                polygon=[[0,0],[10,0],[10,7],[0,7]], ppm=10, camera_matrix=self.K.tolist(),
                dist_coeffs=self.D.tolist(),model='pinhole',records=records)))
            out, m = rebuild(folder,folder/'fused')
            self.assertEqual(m['registration']['optical_trajectory'][1]['segment'], 1)
            self.assertEqual(m['records'][1]['offset_mm'], [0,0])
            self.assertTrue((out/m['records'][0]['file']).is_file())
            owners=cv2.imread(str(out/'tile_index.png'),cv2.IMREAD_UNCHANGED)
            self.assertEqual(owners.dtype,np.uint16)
            grbl, _ = rebuild(folder,folder/'grbl',fused=False)
            for name in ('mosaic.png','coverage.png'):
                np.testing.assert_array_equal(cv2.imread(str(grbl/name)),cv2.imread(str(out/name)))
            with self.assertRaises(FileExistsError):
                rebuild(folder,out)

