import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"z"))
from area_consensus import robust_consensus


class AreaConsensusTests(unittest.TestCase):
    def solve(self, heights, visible):
        layers = np.asarray(heights,float)[:,None,None]
        points = np.zeros((*layers.shape,3))
        points[...,2] = 90 - layers
        cameras = [(np.eye(3),np.array([[z]])) for z in visible]
        out = robust_consensus(layers,points,cameras,np.zeros((len(cameras),3)),np.array([0,0,1]),
                               90,.35,np.nanmedian(layers,axis=0),np.zeros((1,1),bool))
        return tuple(a.item() for a in out)

    def test_strict_majority_rejects_outlier(self):
        height, method, votes, rejected = self.solve([9.98,10.04,10.08,14.2],[np.nan]*4)
        self.assertEqual((method,votes,rejected),(2,3,1))
        self.assertAlmostEqual(height,10.04)

    def test_equal_or_overlapping_groups_remain_conflicting(self):
        for values in ([10,10.1,14,14.1],[10,10.2,10.25,10.45]):
            self.assertEqual(self.solve(values,[np.nan]*4)[1],0)

    def test_single_negative_height_needs_two_other_views(self):
        z = [-12.65,np.nan,np.nan,np.nan]
        self.assertEqual(self.solve(z,[-12.65,-12.63,0,np.nan])[1],0)
        height, method, votes, rejected = self.solve(z,[-12.65,-12.63,0,-12.67])
        self.assertEqual((method,votes,rejected),(3,3,0))
        self.assertAlmostEqual(height,-12.65)

    def test_two_reprojection_hypotheses_do_not_create_a_false_height(self):
        z = [10,10,14,14]
        self.assertEqual(self.solve(z,[10,10,14,14])[1],0)
