import unittest

import numpy as np
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
            pt = Point(data, "sys3")
            check_point(pt, np.asarray(data), "sys3")

        # list / tuple / array of float
        d2 = [1.0, 2.0, 3.0]
        for data in [d2, tuple(d2), np.array(d2)]:
            pt = Point(data, "sys3")
            check_point(pt, np.asarray(data), "sys3")

        # sys already defined as 3d
        with self.assertRaisesRegex(TypeError, r"is 3D \(expected 4D\)"):
            Point([1, 2, 3, 4], "sys3")
        # point must be 1d
        with self.assertRaisesRegex(TypeError, "must be 1D"):
            Point([[1, 2, 3, 4]], None)

        # PointArray from list of int
        data = [[1, 2], [3, 4], [5, 6]]
        arr = PointArray(data, "sys2")
        check_point(arr, np.array(data), "sys2")

        # PointArray from 1d list of Point
        data = [Point([1.0, 2.0], "sys2"), Point([3.0, 4.0], "sys2"), Point([5.0, 6.0], "sys2")]
        arr = PointArray(data, "sys2")
        check_point(arr, np.array(data), "sys2")

        # PointArray from 2d list of Point
        data = [
            [Point([1.0, 2.0], "sys2"), Point([3.0, 4.0], "sys2")],
            [Point([5.0, 6.0], "sys2"), Point([7.0, 8.0], "sys2")],
        ]
        arr = PointArray(data, "sys2")
        check_point(arr, np.array(data), "sys2")
        # check that system is carried from Point
        arr = PointArray(data)
        check_point(arr, np.array(data), "sys2")
        with self.assertRaisesRegex(TypeError, "does not match source coordinates"):
            arr = PointArray(data, "wrong_system")
            check_point(arr, np.array(data), "sys2")

        # PointArray from 2x2 array of Point
        data = np.empty((2, 2), dtype=object)
        data[:] = [
            [Point([1.0, 2.0], "sys2"), Point([3.0, 4.0], "sys2")],
            [Point([5.0, 6.0], "sys2"), Point([7.0, 8.0], "sys2")],
        ]
        arr = PointArray(data, "sys2")
        check_point(arr, np.array(data.tolist()), "sys2")
        # check that system is carried from Point
        arr = PointArray(data)
        check_point(arr, np.array(data.tolist()), "sys2")

    def test_numpy_ops(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        arr = PointArray(data, "sys2")
        assert np.all(arr + 1 == PointArray(data + 1, "sys2"))
        assert np.all(arr * 2 == PointArray(data * 2, "sys2"))
        assert np.all(arr + arr == PointArray(data + data, "sys2"))
        assert np.all(arr * arr == PointArray(data * data, "sys2"))
        assert np.all(np.sum(arr, axis=0) == PointArray(np.sum(data, axis=0), "sys2"))
        assert np.all(np.mean(arr, axis=0) == PointArray(np.mean(data, axis=0), "sys2"))
        assert np.all(np.std(arr, axis=0) == PointArray(np.std(data, axis=0), "sys2"))
        assert np.all(np.var(arr, axis=0) == PointArray(np.var(data, axis=0), "sys2"))
        assert np.all(np.min(arr, axis=0) == PointArray(np.min(data, axis=0), "sys2"))
        assert np.all(np.max(arr, axis=0) == PointArray(np.max(data, axis=0), "sys2"))
