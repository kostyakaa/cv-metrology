"""Exercise c/k/p on generated camera frames and an in-memory GRBL protocol peer."""
import argparse
from collections import deque
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from scan_plane import Abort, PlaneScanner, add_height_args, add_survey_args, apply_calibration_focus
from common.grbl import Grbl, GrblAborted, GrblError, JOG_CANCEL
from survey_markers import CenteringNotConverged
from common.markers import get_dictionary


class Stage:
    def __init__(self):
        self.lines = deque([b'Grbl 1.1f'])
        self.xy = np.zeros(2)
        self.stops = deque()
        self.commands = []

    def readline(self):
        return self.lines.popleft()

    def write(self, data):
        self.commands.append(data)
        if data == b'?':
            if self.stops:
                self.xy = self.stops.popleft()
            state = 'Jog' if self.stops else 'Idle'
            self.lines.append(f'<{state}|MPos:{self.xy[0]:.3f},{self.xy[1]:.3f},0>'.encode())
        elif data == JOG_CANCEL:
            self.stops.clear()
        elif data == b'$$\n':
            self.lines.extend([b'$3=1',b'$100=800',b'$101=800',b'ok'])
        else:
            match = re.fullmatch(rb'\$J=G21G91X([-.\d]+)Y([-.\d]+)F(\d+)\n',data)
            assert match, data
            dx,dy,feed = map(float, match.groups())
            assert 0 < feed <= 120 and max(abs(dx),abs(dy)) > 0
            target = self.xy+[dx,dy]
            assert np.max(np.abs(target)) <= 8
            self.stops.extend(np.linspace(self.xy,target,8 if max(abs(dx),abs(dy)) > 5 else 2)[1:])
            self.lines.append(b'ok')


class Camera:
    def __init__(self, stage):
        self.stage = stage
        self.direction = 1
        self.background = np.random.default_rng(3).integers(90,245,(360,480),np.uint8)
        self.background[97:263,157:323] = 255
        self.background[105:255,165:315] = cv2.aruco.generateImageMarker(get_dictionary(),0,150)

    def read(self):
        M = np.eye(2,3)
        M[:,2] = -10*self.stage.xy*self.direction
        gray = cv2.warpAffine(self.background,M,(480,360),borderMode=cv2.BORDER_REFLECT)
        return True, cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    def grab(self):
        return True


class HardwareTrialTests(unittest.TestCase):
    def test_calibration_focus_understands_directshow_manual_flag(self):
        for backend, autofocus, actual, accepted in [
            ('DSHOW', 2, 185, True), ('V4L2', 0, 185, True),
            ('DSHOW', 1, 185, False), ('DSHOW', 3, 185, False),
            ('DSHOW', 2, 170, False), ('DSHOW', -1, 185, False), ('V4L2', 2, 185, False),
        ]:
            with self.subTest(backend=backend, autofocus=autofocus, actual=actual):
                cap = Mock()
                cap.getBackendName.return_value = backend
                cap.get.side_effect = {cv2.CAP_PROP_FOCUS: actual, cv2.CAP_PROP_AUTOFOCUS: autofocus}.__getitem__
                cap.read.return_value = True, np.zeros((2, 2, 3), np.uint8)
                with patch('common.camera.time.sleep'), redirect_stdout(io.StringIO()):
                    if accepted:
                        apply_calibration_focus(cap, 185)
                    else:
                        with self.assertRaises(Abort):
                            apply_calibration_focus(cap, 185)
                cap.set.assert_any_call(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set.assert_any_call(cv2.CAP_PROP_FOCUS, 185)

    def test_reset_invalidates_subsequent_jogs(self):
        stage = Stage()
        grbl = Grbl(link=stage)
        stage.lines.append(b'Grbl 1.1f')
        with self.assertRaises(GrblError):
            grbl.status()
        with self.assertRaises(GrblError):
            grbl.jog(1,0,120)
        self.assertFalse(any(c.startswith(b'$J=') for c in stage.commands))

    def test_cancel_on_preview_failure_and_escape(self):
        for tick,exc in ((Mock(side_effect=RuntimeError('frame lost')),RuntimeError),(lambda: True,GrblAborted)):
            stage = Stage()
            grbl = Grbl(link=stage)
            with self.assertRaises(exc):
                grbl.jog(6,0,120,tick)
            self.assertIn(JOG_CANCEL, stage.commands)
            self.assertFalse(stage.stops)

    def run_trial(self, wrong_direction=False, closure_failed=False):
        parser = argparse.ArgumentParser()
        add_survey_args(parser)
        add_height_args(parser)
        args = parser.parse_args([])
        args.trial,args.tile_step,args.ppm = True,20,10
        args.frames,args.scan_frames = 1,1
        args.scan_settle,args.settle = 0,0
        V = [[1.,0.],[0.,1.],[0.,0.]]
        K = [[400.,0.,239.5],[0.,400.,179.5],[0.,0.,1.]]
        settings = {'3':1.,'100':800.,'101':800.}
        layout = {'0':[0,0],'1':[100,0],'2':[100,70],'3':[0,70]}
        if closure_failed:
            layout = {'0':[0,0],'1':[6,0],'2':[6,6],'3':[0,6]}
        calib = dict(camera_matrix=K,dist_coeffs=[0]*5,model='pinhole',
                     mode=dict(grbl_settings=settings,focus=185),
                     machine_check=dict(motion=dict(V_cam=V)))
        stage = Stage()
        camera = Camera(stage)
        with tempfile.TemporaryDirectory() as tmp, patch('scan_plane.ROOT',Path(tmp)), redirect_stdout(io.StringIO()):
            root = Path(tmp)
            frame_path,plane_path = root/'frame.json',root/'plane.json'
            frame_path.write_text(json.dumps(dict(grbl_settings=settings,frame=dict(markers_rel_home_machine_mm=layout))))
            plane_path.write_text(json.dumps(dict(calib='camera.json',home_id=0,grbl_settings=settings,
                                  markers_rel_home_machine_mm=layout,plane=dict(n=[0,0,1],d=40))))
            ui = Mock()
            ui.show.return_value = -1
            factory = Mock(side_effect=lambda: Grbl(link=stage))
            tool = PlaneScanner(camera,args,ui,calib,root/'camera.json',grbl_factory=factory,
                                frame_path=frame_path,plane_path=plane_path)
            tool.confirm = lambda msg: True
            tool.render = lambda img,found: img
            tool.show_mosaic = Mock()
            with patch.object(tool,'measure',return_value=None), self.assertRaises(Abort):
                tool.connect()
            factory.assert_not_called()
            tool.connect()
            with self.assertRaises(Abort):
                tool.move(9,0,check=False)
            tool.calibrate_axes()
            if wrong_direction:
                camera.direction = -1
                with self.assertRaisesRegex(Abort,'расходится с GRBL'):
                    tool.scan_plane()
                [partial] = root.glob('mosaics/*/mosaic.json')
                m = json.loads(partial.read_text())
                self.assertFalse(m['complete'])
                self.assertIn('расходится',m['stop_reason'])
                self.assertEqual(len(m['records']),2)
                self.assertAlmostEqual(np.linalg.norm(stage.xy),6,places=1)
                return
            if closure_failed:
                args.trial = False
                args.feed = 120
                with patch.object(tool, 'center_on', side_effect=CenteringNotConverged('0.059 > 0.05')):
                    tool.scan_plane()
                self.assertEqual(tool.mosaic_folder.name, 'fused')
                self.assertTrue(tool.mosaic_meta['capture_complete'])
                self.assertFalse(tool.mosaic_meta['complete'])
                self.assertFalse(tool.mosaic_meta['navigation_quality']['usable'])
                self.assertIn('0.059', tool.mosaic_meta['stop_reason'])
                self.assertGreaterEqual(len(tool.mosaic_meta['records']), 4)
                tool.show_mosaic.assert_called_once()
                with patch('scan_plane.view') as display:
                    PlaneScanner.show_mosaic(tool)
                display.assert_called_once_with(tool.mosaic_folder, None)
                return
            tool.scan_plane()
            folder = tool.mosaic_folder
            self.assertEqual(folder.name,'fused')
            m = json.loads((folder/'mosaic.json').read_text())
            self.assertTrue(m['trial'] and m['complete'])
            self.assertEqual(len(m['records']),5)
            for r in m['records'][1:]:
                leg = r['motion_from_previous']
                self.assertTrue(leg['valid'] and leg['timing_consistent'])
                self.assertLess(np.linalg.norm(r['trial_marker_error_mm']),.1)
                self.assertTrue(any('frame_s' in s for s in leg['status_samples']))
            self.assertLess(np.linalg.norm(m['closure_command_mm']),.1)
            np.testing.assert_allclose(stage.xy,0,atol=.1)

    def test_full_trial_without_devices(self):
        self.run_trial()

    def test_wrong_direction_stops_at_first_corner_without_return(self):
        self.run_trial(wrong_direction=True)

    def test_full_scan_rebuilds_for_viewing_when_final_centering_does_not_converge(self):
        self.run_trial(closure_failed=True)
