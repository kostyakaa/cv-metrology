import argparse
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'z')]
from common.grbl import Grbl, JOG_CANCEL
from common.registration import navigation_quality
from focus_height import acquire, peak_position, stack_positions
from measure_height import Abort, add_height_args, replay_args
from pose_refinement import stereo_positions
from test_hardware_trial import Stage
import test_height_scan


class ZPeer(Stage):
    def __init__(self):
        super().__init__()
        self.z = 0.
        self.move_deltas = []

    def write(self, data):
        if data == b'?':
            self.commands.append(data)
            self.lines.append(f'<Idle|MPos:0.000,0.000,{self.z:.3f}>'.encode())
        elif data.startswith(b'$J='):
            m = re.fullmatch(rb'\$J=G21G91Z([-.\d]+)F(\d+)\n', data)
            assert m
            dz = float(m[1])
            self.move_deltas.append(dz)
            self.z += dz
            self.lines.append(b'ok')
        else:
            super().write(data)


class ExperimentalTests(unittest.TestCase):
    def test_invalid_navigation_and_holdout_no_absolute_guarantee(self):
        m = dict(complete=True, closure_command_mm=[4.657, 0], registration={'anchors': []})
        self.assertFalse(navigation_quality(m)['usable'])
        m.update(closure_command_mm=[0, 0], registration={'anchors': [dict(marker=i) for i in range(4)]})
        self.assertTrue(navigation_quality(m)['usable'])
        m['return_to_start_mm'] = [.7, 0]
        self.assertFalse(navigation_quality(m)['usable'])

    def test_focus_limits_and_rejected_flat_peak(self):
        np.testing.assert_allclose(stack_positions(2, 1, .25, -1, [1, 2], 30), [2,1.75,1.5,1.25,1])
        for span,step in [(2,.25), (1,0), (1,.001)]:
            with self.assertRaises(ValueError):
                stack_positions(2, span, step, -1, [1, 2], 30)
        self.assertFalse(peak_position([dict(z=i,score=1) for i in range(5)])['valid'])
        shiny = [dict(z=i, score=s, saturation=.2) for i,s in enumerate([1,2,5,2,1])]
        self.assertFalse(peak_position(shiny)['valid'])

    def test_xy_requires_fresh_status(self):
        controller = Grbl(link=ZPeer())
        with patch.object(controller, "status", side_effect=lambda: controller._parse_status("<Idle|WPos:2,3,4>")):
            self.assertEqual(controller.position(), (2., 3.))

    def focus_tool(self, folder):
        peer = ZPeer()
        controller = Grbl(link=peer)
        tool = Mock()
        tool.args = argparse.Namespace(focus_z_span=1., focus_z_step=.25, z_up_sign=1,
                                      z_limits=[0.,1.], z_feed=30, scan_frames=3, scan_settle=0., confirm=True)
        tool.scans_dir = Path(folder)
        tool.grbl = controller
        tool.focus_session = 'synthetic'
        tool.focus, tool.profile_id = 185, 'test'
        tool.K = np.array([[80,0,64],[0,80,64],[0,0,1]])
        tool.position.return_value = np.zeros(2)
        tool.confirm.return_value = True
        tool.tick.return_value = False
        pattern = np.random.default_rng(3).integers(0,255,(128,128,3),np.uint8)
        tool.cap.read.side_effect = lambda: (True, cv2.GaussianBlur(pattern,(0,0),.4+abs(peer.z-.5)*6))
        return tool, peer

    def test_motor_focus_saves_curve_and_returns(self):
        with tempfile.TemporaryDirectory() as folder:
            tool, peer = self.focus_tool(folder)
            acquire(tool)
            data = json.loads(next(Path(folder).glob('*/focus.json')).read_text(encoding='utf-8'))
            self.assertEqual(peer.move_deltas, [.25,.25,.25,.25,-1.])
            self.assertTrue(data['complete'] and data['peak']['valid'])
            self.assertAlmostEqual(data['peak']['z_peak_mm'],.5, places=3)

    def test_z_escape_retains_partial_stack_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            tool, peer = self.focus_tool(folder)
            tool.tick.side_effect = lambda *args: peer.z >= .25
            with self.assertRaises(SystemExit):
                acquire(tool)
            self.assertIn(JOG_CANCEL, peer.commands)
            self.assertEqual(peer.move_deltas,[.25])
            data = json.loads(next(Path(folder).glob('*/focus.json')).read_text(encoding='utf-8'))
            self.assertFalse(data['complete'])

    def test_area_plan_and_impossible_extent(self):
        tool = test_height_scan.PlanningTests().tool()
        tool.args.area_mm = [95,34]
        plan = tool.plan_scan(np.array([140.,46.]))
        self.assertTrue(25 < len(plan) <= 49)
        self.assertEqual(tool.scan_plan['area_mm'],[95,34])
        tool.args.area_mm = [1000,1000]
        with self.assertRaises(Abort):
            tool.plan_scan(np.array([140.,46.]))

    def test_area_registers_signed_heights_and_leaves_holes(self):
        from area_height import reconstruct
        args = add_height_args(argparse.ArgumentParser()).parse_args(
            ['--area-mm','4','4','--area-margin','0','--cell-mm','1','--references','2'])
        xy = [[10., 15.], [11., 15.]]
        meta = dict(ref_index=0, positions=xy, V_cam=np.eye(3)[:, :2], camera_matrix=np.eye(3),
                    dist_coeffs=[], model='pinhole', guess=dict(n=[0.,0.,1.], d=80., kind='table'))
        def tile(x):
            return argparse.Namespace(n_t=np.array([0.,0.,1.]), d_t=80., ref_kind='table',
                                      X=np.array([[[x,0.,92.65], [x+1.,0.,70.]]]), valid=np.ones((1,2),bool))
        with tempfile.TemporaryDirectory() as temp, patch('area_height.ScanResult', return_value=tile(-1.)):
            out = Path(temp)/'area'
            result = reconstruct([None,None], meta, args, out, first_scan=tile(0.))
            self.assertEqual((result['observed'],result['confirmed']), (2,2))
            with np.load(out/'area.npz') as grid:
                np.testing.assert_allclose(grid['xy_origin'], [8.,13.])
                np.testing.assert_allclose(grid['height_mm'][2,2:4], [-12.65,10.], atol=.0001)
                self.assertTrue(np.isnan(grid['height_mm'][0,0]))

    def test_replay_uses_saved_threshold_and_explicit_roi(self):
        ap = add_height_args(argparse.ArgumentParser())
        meta = {'height_settings': {'below':18, 'table_roi':[[1,1,10,10]]}}
        self.assertEqual(replay_args(ap, meta, []).below,18)
        args = replay_args(add_height_args(argparse.ArgumentParser()), meta, ['--table-roi','2','2','8','8'])
        self.assertEqual(args.table_roi,[[2,2,8,8]])
        meta['height_settings'].update(pose_tag_height=1., references=7)
        args = replay_args(add_height_args(argparse.ArgumentParser()), meta, ['--no-pose-refinement'])
        self.assertIsNone(args.pose_tag_height)
        self.assertEqual(args.references, 7)

    def test_relative_pose_correction_requires_visual_evidence(self):
        pos = np.array([[x,y] for y in range(3) for x in range(3)],float)*3
        actual = pos.copy()
        actual[:, 0] += np.array([0,.14,-.12,.1,-.05,.17,-.12,.09,.16])
        K = np.array([[200,0,100],[0,200,100],[0,0,1.]])
        meta = dict(positions=pos, ref_index=0, V_cam=np.eye(3)[:,:2],
                    camera_matrix=K, dist_coeffs=np.zeros(5), model='pinhole', guess=dict(n=[0,0,1],d=40))
        frames = [np.zeros((200,200),np.uint8) for _ in pos]
        corners = np.array([[90,90],[110,90],[110,110],[90,110]],float)
        found = [{4:corners-5*p} for p in actual]
        args = argparse.Namespace(pose_tag_height=0.)
        with patch('pose_refinement.detect_precise',side_effect=found):
            fitted = stereo_positions(frames,meta,args)
        self.assertTrue(meta['pose_fit']['applied'])
        self.assertLess(np.linalg.norm(fitted-actual),np.linalg.norm(pos-actual))
        del meta['pose_fit']
        with patch('pose_refinement.detect_precise',return_value={}):
            np.testing.assert_array_equal(stereo_positions(frames,meta,args), pos)
