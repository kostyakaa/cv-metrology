import unittest
import cv2
import numpy as np
from common.mosaic import pixel_to_machine, overlap_error, Mosaic


class MosaicTests(unittest.TestCase):
    def test_overlap_does_not_average_misaligned_edges(self):
        m=Mosaic([[0,0],[10,0],[10,10],[0,10]],10)
        a=np.zeros((101,101,3),np.uint8);a[:,50:]=255
        b=np.zeros_like(a);b[:,55:]=255
        for image in (a,b):
            m.add(image,np.ones((101,101),bool),np.diag([.1,.1,1]),[0,0],diagnose=False)
        self.assertTrue(np.all(np.isin(m.image()[50,:,0],[0,255])))

    def test_single_sample_matches_identity(self):
        m=Mosaic([[0,0],[10,0],[10,10],[0,10]],10)
        a=np.random.default_rng(1).integers(0,256,(101,101,3),dtype=np.uint8)
        K=np.array([[100.,0,50],[0,100,50],[0,0,1]])
        m.add(a,np.ones((101,101),bool),np.diag([.1,.1,1]),[0,0],raw=a,K=K,D=np.zeros(5),diagnose=False)
        np.testing.assert_array_equal(m.image(),a)

    def test_projection_and_sign(self):
        K=np.array([[100.,0,50],[0,100,50],[0,0,1]])
        V=np.array([[1.,0],[0,-1],[0,0]])
        H=pixel_to_machine(K,V,np.array([0.,0,1]),10,np.array([7.,8]))
        p=cv2.perspectiveTransform(np.array([[[50.,50.],[60.,60.]]]),H)[0]
        np.testing.assert_allclose(p,[[7,8],[8,7]])

    def test_known_error_and_no_texture(self):
        rng=np.random.default_rng(3)
        a=rng.integers(0,256,(240,320,3),dtype=np.uint8)
        a=cv2.GaussianBlur(a,(5,5),1)
        b=cv2.warpAffine(a,np.float32([[1,0,2],[0,1,-1]]),(320,240))
        result=overlap_error(a,b,np.ones((240,320),bool))
        self.assertTrue(result['valid'])
        np.testing.assert_allclose(result['shift_px'],[2,-1],atol=.15)
        self.assertFalse(overlap_error(a*0,a*0,np.ones((240,320),bool))['valid'])

    def test_mosaic_coordinates_and_coverage(self):
        m=Mosaic([[0,0],[10,0],[10,10],[0,10]],10)
        m.add(np.full((101,101,3),100,np.uint8),np.ones((101,101),bool),
              np.diag([.1,.1,1]),[0,0])
        np.testing.assert_allclose(m.xy([50,80]),[5,8])
        self.assertEqual(int(m.image()[50,50,0]),100)


if __name__=='__main__':unittest.main()
