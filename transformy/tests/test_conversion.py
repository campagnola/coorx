import unittest
import numpy as np
from transformy import linear

try:
    import vispy.scene
    HAVE_VISPY = True
except ImportError:
    HAVE_VISPY = False

try:
    import pyqtgraph
    HAVE_PYQTGRAPH = True
except ImportError:
    HAVE_PYQTGRAPH = False


class VispyConversion(unittest.TestCase):
    def test_vispy_conversion(self):
        if not HAVE_VISPY:
            self.skipTest("vispy could not be imported")
        
        pts = np.random.normal(size=(100, 3))
        
        v_sttr = vispy.scene.STTransform(translate=(2, 3), scale=(5.6, -9))
        t_sttr = linear.STTransform.convert_from(v_sttr, 'vispy')
        v_sttr2 = t_sttr.convert_to('vispy')

        assert np.allclose(v_sttr.map(pts)[:,:3], t_sttr.map(pts))
        assert np.allclose(v_sttr2.map(pts)[:,:3], t_sttr.map(pts))


class PyqtgraphyConversion(unittest.TestCase):
    def test_pyqtgraph_conversion(self):
        if not HAVE_PYQTGRAPH:
            self.skipTest("pyqtgraph could not be imported")
        
        pts2 = np.random.normal(size=(100, 2))
        
        qtr1 = pyqtgraph.QtGui.QTransform()
        qtr1.scale(1.1, -9)
        qtr1.translate(3, -30)
        qtr1.rotate(32)
        t_atr = linear.STTransform.convert_from(qtr1, 'pyqtgraph')
        qtr2 = t_atr.convert_to('pyqtgraph')

        assert np.allclose(pyqtgraph.transformCoordinates(qtr1, pts2, transpose=True), t_atr.map(pts2))
        assert np.allclose(pyqtgraph.transformCoordinates(qtr2, pts2, transpose=True), t_atr.map(pts2))
        
        
        pts3 = np.random.normal(size=(100, 3))
        
        qtr1 = pyqtgraph.Transform3D()
        qtr1.scale(1.1, -9, 1e4)
        qtr1.translate(3, -30, 0)
        qtr1.rotate(32, 0, 1, 2)
        t_atr = linear.AffineTransform.convert_from(qtr1, 'pyqtgraph')
        print(t_atr)
        qtr2 = t_atr.convert_to('pyqtgraph')

        assert np.allclose(pyqtgraph.transformCoordinates(qtr1, pts3, transpose=True), t_atr.map(pts3))
        assert np.allclose(pyqtgraph.transformCoordinates(qtr2, pts3, transpose=True), t_atr.map(pts3))        