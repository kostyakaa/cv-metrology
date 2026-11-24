"""--open: загрузка сохранённой мозаики, привязка к домашней метке, контроль $3/$100/$101."""
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'z'), str(ROOT/'scripts')]
from scan_plane import PlaneScanner, load_mosaic, main, view  # noqa: E402
from survey_markers import Abort  # noqa: E402

K = np.array([[1101.93, 0, 938.45], [0, 1100.45, 568.11], [0, 0, 1]])
D = np.array([-0.29761, 0.09217, -0.0002, 0.00015, -0.01277])


def make_session(root, settings=None, complete=True):
    d = Path(root)/'sess'
    d.mkdir()
    cv2.imwrite(str(d/'mosaic.png'), np.zeros((10, 10, 3), np.uint8))
    cv2.imwrite(str(d/'coverage.png'), np.zeros((10, 10), np.uint8))
    meta = dict(version=1, home_id=0, ppm=10, complete=complete,
                polygon=[[0, 0], [280, 0], [280, 100], [0, 100]],
                camera_matrix=K.tolist(), dist_coeffs=D.tolist(), focus=185,
                records=[dict(file='frame_0000.png', position=[0, 0], H=np.eye(3).tolist())])
    if settings is not None:
        meta['grbl_settings'] = settings
    (d/'mosaic.json').write_text(json.dumps(meta), encoding='utf-8')
    return d


class LoadMosaicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_folder_and_json_path(self):
        d = make_session(self.tmp.name, settings={'3': 0, '100': 800, '101': 800})
        folder, meta = load_mosaic(d, K, D)
        self.assertEqual(folder, d)
        self.assertEqual(meta['grbl_settings']['100'], 800)
        folder2, _ = load_mosaic(d/'mosaic.json', K, D)
        self.assertEqual(folder2, d)

    def test_reject_other_calibration(self):
        d = make_session(self.tmp.name)
        K2 = K.copy()
        K2[0, 0] *= 1.01
        with self.assertRaises(Abort):
            load_mosaic(d, K2, D)
        with self.assertRaises(Abort):
            load_mosaic(d, K, D*0.5)

    def test_reject_broken_folder(self):
        d = make_session(self.tmp.name)
        (d/'mosaic.png').unlink()
        with self.assertRaises(Abort):
            load_mosaic(d, K, D)
        with self.assertRaises(Abort):
            load_mosaic(Path(self.tmp.name)/'nope', K, D)

    def test_click_undoes_source_tile_offset_and_scale_before_fake_motion(self):
        d = make_session(self.tmp.name)
        _, meta = load_mosaic(d)
        grid = dict(origin=[0,0],spacing=1,values=[[.02,.02],[.02,.02]])
        meta.update(polygon=[[0,0],[1,0],[1,1],[0,1]], registration={'method':'test','scale_grid':grid},
                    tile_index='tile_index.png')
        meta['records'][0]['offset_mm'] = [.2,-.1]
        (d/'mosaic.json').write_text(json.dumps(meta),encoding='utf-8')
        cv2.imwrite(str(d/'coverage.png'), np.ones((10,10), np.uint8)*255)
        cv2.imwrite(str(d/'tile_index.png'), np.ones((10,10), np.uint16))
        tool = Mock()
        tool.mosaic_anchor.return_value = np.array([3.,4.])
        tool.position.return_value = np.zeros(2)
        tool.backlash_comp = 0.
        tool.args.feed = 300
        def click(name, callback):
            callback(cv2.EVENT_LBUTTONDOWN,6,5,0,None)
            callback(cv2.EVENT_LBUTTONDOWN,7,5,0,None)
        with patch.multiple(cv2, namedWindow=Mock(), imshow=Mock(), destroyWindow=Mock(),
                            setMouseCallback=click, waitKey=Mock(side_effect=[ord('g'),27]),
                            getWindowProperty=Mock(return_value=1)):
            view(d,tool)
        np.testing.assert_allclose(tool.check_target.call_args.args[0],np.array([3.,4.])+np.array([.5,.6])/1.02)


def bare_tool(**attrs):
    t = object.__new__(PlaneScanner)
    t.mosaic_folder, t.mosaic_meta, t.mosaic_home_id = None, None, 0
    t.messages, t.markers, t.layout = [], {}, {}
    for k, v in attrs.items():
        setattr(t, k, v)
    return t


class AnchorTests(unittest.TestCase):
    def test_anchor_from_layout_and_recorded_marker(self):
        # записана ID0 на (10,-5), раскладка задаёт ID1 на +280 по X: дом ID1 -> (290,-5)
        t = bare_tool(mosaic_folder=Path('x'), mosaic_meta={}, mosaic_home_id=1,
                      markers={0: np.array([10., -5.])}, layout={0: [0, 0], 1: [280., 0]})
        np.testing.assert_allclose(t.mosaic_anchor(), [290., -5.])

    def test_anchor_none_without_binding_or_mosaic(self):
        self.assertIsNone(bare_tool(mosaic_folder=Path('x')).mosaic_anchor())
        self.assertIsNone(bare_tool(markers={0: np.array([1., 1.])}).mosaic_anchor())


class WorkflowTests(unittest.TestCase):
    def test_visual_refinement_moves_in_measured_direction_and_verifies(self):
        tool = bare_tool(mosaic_meta={'focus':185},focus=185,args=Mock(slow_feed=300),
                         backlash_comp=.1,K=np.eye(3),V=np.eye(3)[:,:2])
        xy = np.array([4.,5.])
        tool.mosaic_anchor = lambda: np.zeros(2)
        tool.position = lambda: xy.copy()
        tool.table_plane_here = lambda _: ((np.array([0,0,1]),1.),None)
        image = np.zeros((200,200,3),np.uint8)
        tool.capture_color = Mock(return_value=image)
        tool.rect = lambda frame: frame
        tool.check_target = Mock()
        def move(dx,dy,feed):
            self.assertLessEqual(feed,120)
            xy[:] += [dx,dy]
        tool.move = Mock(side_effect=move)
        with patch('scan_plane.target_displacement',side_effect=[np.array([.4,-.3]),np.array([.01,0.])]):
            tool.refine_mosaic_target(image,np.ones(image.shape[:2],np.uint8),(100,100),16)
        self.assertEqual(tool.move.call_count,2)
        self.assertEqual(tool.capture_color.call_count,2)
        np.testing.assert_allclose(xy,[4.4,4.7])

    def test_normal_mode_routes_four_corner_scan_jog_and_heights(self):
        cap, tool = Mock(), Mock()
        cap.get.return_value = 0.
        cap.getBackendName.return_value = "DSHOW"
        tool.grbl = None
        tool.focus = 185
        tool.seen_marker = False
        tool.ui.closed = False
        tool.ui.show.side_effect = map(ord, 'ckrtp1wasdbgjvq')
        frame = np.zeros((10,10,3), np.uint8)
        with (
            patch.object(sys,'argv',['scan_plane.py','--port','COM4']),
            patch('scan_plane.open_camera',return_value=cap),
            patch('scan_plane.apply_calibration_focus',return_value=185),
            patch('scan_plane.read_frame',return_value=frame),
            patch('scan_plane.find_calibration',return_value=({'mode':{'focus':185}},Path('calib.json'))),
            patch('scan_plane.PlaneScanner',return_value=tool),
            patch('scan_plane.detect',return_value={}),
            patch('scan_plane.WindowUI'),
            patch.object(cv2,'destroyAllWindows'),
            patch('builtins.print'),
        ):
            main()
        for action in ('connect','calibrate_axes','run_survey','measure_table_plane','scan_plane',
                       'scan_heights','go_selected','scan_focus_z','show_mosaic'):
            getattr(tool,action).assert_called_once_with()
        self.assertEqual([call.args[0] for call in tool.jog_key.call_args_list],list(map(ord,'wasd')))
        self.assertEqual(tool.jog_step,.1)

    def test_auto_keeps_confirmation_for_navigation_and_stops_after_bad_closure(self):
        settings = {3: 1, 100: 800, 101: 800}
        for yes, closure_ok in ((False, True), (True, True), (True, False)):
            with self.subTest(yes=yes, closure_ok=closure_ok):
                args = Mock(confirm=True, yes=yes)
                tool = bare_tool(args=args, grbl=Mock(), calib_settings=settings, layout_settings=settings,
                                 V=np.eye(3)[:, :2], V_calib=np.eye(3)[:, :2])
                tool.layout = {0: [0., 0.], 1: [20., 0.], 2: [20., 10.], 3: [0., 10.]}
                tool.grbl.settings.return_value = settings
                tool.require_grbl = Mock()
                tool.require_marker = Mock(return_value={'id': 0})
                tool.confirm = Mock(return_value=True)
                stages = []

                def stage(key):
                    stages.append(key)
                    self.assertFalse(args.confirm)
                    if key == 'r' and not closure_ok:
                        raise Abort('Замыкание 0.600 мм')

                tool.calibrate_axes = Mock(side_effect=lambda: stage('k'))
                tool.run_survey = Mock(side_effect=lambda **kwargs: stage('r'))
                tool.measure_table_plane = Mock(side_effect=lambda **kwargs: stage('t'))
                tool.scan_plane = Mock(side_effect=lambda **kwargs: stage('p'))
                tool.show_mosaic = Mock(side_effect=lambda: self.assertTrue(args.confirm))
                if closure_ok:
                    tool.run_auto()
                    self.assertEqual(stages, list('krtp'))
                    tool.run_survey.assert_called_once_with(max_closure=.5)
                    tool.measure_table_plane.assert_called_once_with(max_closure=.5)
                    tool.scan_plane.assert_called_once_with(open_view=False)
                    tool.show_mosaic.assert_called_once()
                else:
                    with self.assertRaisesRegex(Abort, 'Замыкание'):
                        tool.run_auto()
                    self.assertEqual(stages, list('kr'))
                    tool.show_mosaic.assert_not_called()
                self.assertEqual(tool.confirm.call_count, int(not yes))
                self.assertTrue(args.confirm)

    def test_auto_requires_saved_layout_before_any_axis_motion(self):
        tool = bare_tool(args=Mock(yes=True), grbl=Mock())
        tool.require_grbl = Mock()
        tool.calibrate_axes = Mock()
        with self.assertRaisesRegex(Abort, 'сохранённые четыре метки'):
            tool.run_auto()
        tool.calibrate_axes.assert_not_called()

    def test_bad_closure_rolls_back_session_markers(self):
        old = {0: np.array([0., 0.]), 1: np.array([20., 0.]), 2: np.array([20., 12.]), 3: np.array([0., 12.])}
        tool = bare_tool(A=np.eye(2), markers=old.copy(), closure=np.array([.01, 0.]), depth_pass=False)
        tool.require_grbl = tool.say = tool.save_frame = Mock()
        tool.expected = lambda: tool.markers.copy()
        tool.center_on = lambda: (0, np.array([.1, 0.]))
        tool.travel_to = lambda mid: (mid, old[mid] + (1. if mid == 0 else .1))
        with self.assertRaisesRegex(Abort, 'Замыкание'):
            tool.run_survey(max_closure=.5)
        for mid in old:
            np.testing.assert_array_equal(tool.markers[mid], old[mid])
        np.testing.assert_array_equal(tool.closure, [.01, 0])
        self.assertIsNone(tool.last_centered)

    def test_center_requires_visible_marker_before_backlash_move(self):
        tool = bare_tool(A=np.eye(2), last_dir=np.zeros(2))
        tool.require_grbl = Mock()
        tool.require_marker = Mock(side_effect=Abort('no marker'))
        tool.move_approach = Mock()
        with self.assertRaisesRegex(Abort, 'no marker'):
            tool.center_on(0)
        tool.move_approach.assert_not_called()

    def test_shifted_corner_stops_before_record_or_final_servo(self):
        tool = bare_tool(A=np.eye(2)*13, target=np.zeros(2), size=(1920,1080))
        target = np.array([295., 0.])
        approach = np.array([283., 0.])
        tool.position = Mock(side_effect=[np.zeros(2), approach])
        tool.require_grbl = Mock()
        tool.travel_to_point = Mock()
        tool.expected = lambda: {3: target}
        tool.measure = lambda mid: {'center': np.array([-1.7*13, 0.])}
        tool.center_on = Mock()
        with self.assertRaisesRegex(Abort, 'ID3'):
            tool.travel_to(3)
        np.testing.assert_allclose(tool.travel_to_point.call_args.args[0], approach)
        tool.center_on.assert_not_called()
        tool.markers = {3: target.copy()}
        with self.assertRaisesRegex(Abort, 'ID3'):
            tool.record(3, approach + [1.7, 0.])
        np.testing.assert_array_equal(tool.markers[3], target)

    def test_trial_mosaic_reopened_for_inspection(self):
        tool = bare_tool(mosaic_folder=Path('trial'), mosaic_meta={'trial':True},args=Mock(trial=False))
        with patch('scan_plane.view') as display:
            tool.show_mosaic()
        display.assert_called_once_with(Path('trial'),None)

class MosaicSettingsTests(unittest.TestCase):
    def setUp(self):
        self.t = bare_tool(mosaic_folder=Path('x'), mosaic_meta={'grbl_settings': {'3': 0, '100': 800, '101': 800}})

    def test_unchanged_settings_keep_mosaic(self):
        self.t.check_mosaic_settings({3: 0, 100: 800, 101: 800})
        self.assertIsNotNone(self.t.mosaic_folder)

    def test_changed_settings_disable_mosaic(self):
        with self.assertRaises(Abort):
            self.t.check_mosaic_settings({3: 3, 100: 800, 101: 800})
        self.assertIsNone(self.t.mosaic_folder)  # отвязана: g больше не поедет

    def test_missing_saved_settings_only_warns(self):
        t = bare_tool(mosaic_folder=Path('x'), mosaic_meta={})
        t.check_mosaic_settings({3: 3, 100: 900, 101: 900})
        self.assertIsNotNone(t.mosaic_folder)
        self.assertTrue(any('не записаны' in m for m in t.messages))


if __name__ == '__main__':
    unittest.main()
