"""--open: загрузка сохранённой мозаики, привязка к домашней метке, контроль $3/$100/$101."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'z'), str(ROOT/'scripts')]
from scan_plane import PlaneScanner, load_mosaic  # noqa: E402
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
