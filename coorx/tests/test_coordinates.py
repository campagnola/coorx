import unittest
import numpy as np
import coorx as coorx
from coorx import Point, PointArray


def check_point(pt, arr, sys):
    assert isinstance(pt.coordinates, np.ndarray)
    assert pt.ndim == arr.ndim
    assert pt.dtype == arr.dtype
    assert len(pt) == len(arr)
    assert pt.shape == arr.shape
    assert np.all(pt.coordinates == arr)
    assert pt.system.name == sys

    
class PointTests(unittest.TestCase):
    def test_init(self):
        # list / tuple / array of int
        d1 = [1, 2, 3]
        for data in [d1, tuple(d1), np.array(d1)]:
            pt = Point(data, 'sys3')
            check_point(pt, np.asarray(data), 'sys3')

        # list / tuple / array of float
        d2 = [1., 2., 3.]
        for data in [d2, tuple(d2), np.array(d2)]:
            pt = Point(data, 'sys3')
            check_point(pt, np.asarray(data), 'sys3')

        # sys already defined as 3d
        with self.assertRaisesRegex(TypeError, r'is 3D \(expected 4D\)'):
            Point([1, 2, 3, 4], 'sys3')
        # point must be 1d
        with self.assertRaisesRegex(TypeError, 'must be 1D'):
            Point([[1, 2, 3, 4]], None)

        # PointArray from list of int
        data = [[1,2], [3,4], [5,6]]
        arr = PointArray(data, 'sys2')
        check_point(arr, np.array(data), 'sys2')

        # PointArray from 1d list of Point
        data = [Point([1.,2.], 'sys2'), Point([3.,4.], 'sys2'), Point([5.,6.], 'sys2')]
        arr = PointArray(data, 'sys2')
        check_point(arr, np.array(data), 'sys2')

        # PointArray from 2d list of Point
        data = [
            [Point([1.,2.], 'sys2'), Point([3.,4.], 'sys2')], 
            [Point([5.,6.], 'sys2'), Point([7.,8.], 'sys2')],
        ]
        arr = PointArray(data, 'sys2')
        check_point(arr, np.array(data), 'sys2')
        # check that system is carried from Point
        arr = PointArray(data)
        check_point(arr, np.array(data), 'sys2')
        with self.assertRaisesRegex(TypeError, "does not match source coordinates"):
            arr = PointArray(data, 'wrong_system')
            check_point(arr, np.array(data), 'sys2')

        # PointArray from 2x2 array of Point
        data = np.empty((2, 2), dtype=object)
        data[:] = [
            [Point([1.,2.], 'sys2'), Point([3.,4.], 'sys2')], 
            [Point([5.,6.], 'sys2'), Point([7.,8.], 'sys2')],
        ]
        arr = PointArray(data, 'sys2')
        check_point(arr, np.array(data.tolist()), 'sys2')
        # check that system is carried from Point
        arr = PointArray(data)
        check_point(arr, np.array(data.tolist()), 'sys2')
        
