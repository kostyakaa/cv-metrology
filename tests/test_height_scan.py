"""Offline tests: no camera or serial connection is constructed."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "z"))
from measure_height import HeightTool, Abort, add_height_args
from stereo import find_objects, plane_homography, plane_sweep
from survey_markers import add_survey_args


class PlanningTests(unittest.TestCase):
    def tool(self):
        t = HeightTool.__new__(HeightTool)
        t.args = add_height_args(argparse.ArgumentParser()).parse_args([])
        t.backlash_comp = 0.2
        t.size = (1920, 1080)
        t.K = np.array([[1100., 0, 960], [0, 1100, 540], [0, 0, 1]])
        t.V = np.array([[1., 0], [0, 1], [0, 0]])
        t.rect = argparse.Namespace(valid=np.ones((1080, 1920), bool))
        t.expected = lambda: {0: [0, 0], 1: [280, 0], 2: [280, 92], 3: [0, 92]}
        t.plane_guess = lambda c: ((np.array([0., 0, 1]), 70., "test"), None)
        t.checked = []
        t.check_target = lambda p: t.checked.append(np.array(p))
        return t

    def test_center_and_edges(self):
        for start in ([140, 46], [0, 0], [280, 92], [0, 46]):
            t = self.tool()
            c = np.array(start, float)
            positions = np.array(t.plan_scan(c))
            self.assertEqual(len(np.unique(positions, axis=0)), 25)
            self.assertTrue(np.any(np.linalg.norm(positions - c, axis=1) < 1e-8))
            self.assertGreaterEqual(t.scan_plan['min_pair_overlap'], t.args.min_overlap)
            self.assertTrue(np.all(np.array(t.checked) >= [-5, -5]))
            self.assertTrue(np.all(np.array(t.checked) <= [285, 97]))
            self.assertLessEqual(t.scan_plan['actual_step_mm'], 16)
            if start == [140, 46]:
                self.assertTrue(np.all(np.ptp(positions, axis=0) > 24))

    def test_reject_before_motion(self):
        for key, value in [('grid', 4), ('baseline', float('nan')), ('max_height', 70),
                           ('min_overlap', 0), ('map_views', 2), ('map_views', 25),
                           ('scan_frames', 0), ('sweep_step', 0)]:
            t = self.tool()
            setattr(t.args, key, value)
            with self.assertRaises(Abort):
                t.plan_scan(np.array([140., 46]))
        t = self.tool()
        t.expected = lambda: {}
        with self.assertRaises(Abort):
            t.plan_scan(np.array([140., 46]))


class ReconstructionTests(unittest.TestCase):
    def test_capture_only_saves_raw_frames_without_reconstruction(self):
        tool = HeightTool.__new__(HeightTool)
        tool.args = add_height_args(add_survey_args(argparse.ArgumentParser())).parse_args(["--capture-only"])
        tool.require_grbl = Mock()
        tool.A = np.eye(2)
        tool.position = Mock(return_value=np.zeros(2))
        tool.plan_scan = Mock(return_value=[np.zeros(2)])
        tool.scan_plan = dict(actual_step_mm=12, min_pair_overlap=.5)
        tool.confirm = Mock(return_value=True)
        tool.backlash_comp = .2
        tool.move = Mock()
        tool.move_approach = Mock()
        tool.capture = Mock(return_value=np.full((6, 8), 42.5, np.float32))
        tool.last_capture = dict(received_s=[1.0])
        tool.grbl = argparse.Namespace(reported_position=[0., 0., 42.], reported_frame="MPos")
        tool.plane_guess = Mock(return_value=((np.array([0., 0., 1.]), 84., "test"), None))
        tool.K = np.eye(3)
        tool.D = np.zeros(5)
        tool.V = np.eye(3)[:, :2]
        tool.calib_name, tool.calib_model, tool.profile_id, tool.focus = "camera.json", "pinhole", "test", 185
        tool.say = Mock()
        tool.scan = object()
        with tempfile.TemporaryDirectory() as folder, patch("measure_height.ScanResult") as reconstruct:
            tool.scans_dir = Path(folder)
            self.assertIsNone(tool.scan_here())
            reconstruct.assert_not_called()
            self.assertIsNone(tool.scan)
            meta = json.loads((tool.scan_folder / "scan.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["captures"][0]["status_xyz"], [0., 0., 42.])
            image = cv2.imread(str(tool.scan_folder / "frame_00.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(image.dtype, np.uint16)
            self.assertEqual(image[0, 0], 10880)

    def test_supported_surface_ignores_isolated_high_spikes(self):
        h = np.full((45,81),np.nan,dtype=np.float32)
        h[7:36,7:70:3] = 10.0
        h[7:36,8:70:3] = 10.05
        h[21,24:65:5] = 21
        [part] = find_objects(h,np.isfinite(h), min_area_px=100)
        self.assertAlmostEqual(part['height_top'],10,delta=.06)
        self.assertLess(part['height_max'],11)

    def test_view_count_above_uint8_limit(self):
        rng = np.random.default_rng(11)
        texture = rng.uniform(0, 255, (60, 88)).astype(np.float32)
        ref = texture[:, :80]
        other = texture[:, 4:84]
        K = np.array([[64., 0, 40], [0, 64, 30], [0, 0, 1]])
        # Shift is exactly 4 px at depth 64 (height 6).
        h, _, valid = plane_sweep(ref, [other] * 260, [np.array([4., 0, 0])] * 260,
                                  K, np.array([0., 0, 1]), 70, [2., 6., 10.],
                                  min_views=256, workers=1)
        self.assertTrue(valid[30, 40])
        self.assertAlmostEqual(float(h[30, 40]), 6, delta=.7)

    def test_known_plane_more_views_and_larger_baseline(self):
        rng = np.random.default_rng(42)
        texture = cv2.GaussianBlur(rng.uniform(0, 255, (600, 800)).astype(np.float32), (3, 3), 0)
        K = np.array([[180., 0, 160], [0, 180, 90], [0, 0, 1]])
        n = np.array([0., 0, 1])
        # Render a large textured plane at height 8 mm above a table 70 mm away.
        xx, yy = np.meshgrid(np.arange(320), np.arange(180))
        def render(o):
            x = (xx + 200 + 180 * o[0] / 62).astype(np.float32)
            y = (yy + 200 + 180 * o[1] / 62).astype(np.float32)
            return cv2.remap(texture, x, y, cv2.INTER_LINEAR) + rng.normal(0, 1, xx.shape).astype(np.float32)
        ref = render([0, 0, 0])
        for grid, step, minimum in [(3, 12, 2), (5, 16, 4)]:
            coords = (np.arange(grid) - grid // 2) * step
            Os = [np.array([x, y, 0.]) for y in coords for x in coords if x or y]
            h, score, valid = plane_sweep(ref, [render(o) for o in Os], Os, K, n, 70,
                                         np.arange(5, 11.01, .25), min_views=minimum)
            core = np.zeros_like(valid)
            core[40:-40, 60:-60] = True
            self.assertGreater(np.mean(valid[core]), .95)
            error = np.sqrt(np.mean((h[valid & core] - 8) ** 2))
            self.assertLess(error, .1)
            print(f'{grid}x{grid}, step {step}: synthetic flat-plane RMS {error:.4f} mm')
        # Four-view threshold must not accept a pixel seen in only three views.
        h, _, valid = plane_sweep(ref, [ref] * 3, [np.zeros(3)] * 3, K, n, 70,
                                  np.arange(5, 11.01, .25), min_views=4)
        self.assertFalse(valid.any())


if __name__ == '__main__':
    unittest.main()
