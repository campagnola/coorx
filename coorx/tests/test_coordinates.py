import pickle
import unittest

import numpy as np
import pytest

from coorx import Point, PointArray, Vector, VectorArray
from coorx.systems import CoordinateSystemGraph, CoordinateSystem, get_coordinate_system


def check_point(pt, arr, sys_name):
    """Helper to check Point/PointArray properties."""
    assert isinstance(pt.coordinates, np.ndarray)
    assert pt.ndim == arr.ndim
    assert pt.dtype == arr.dtype
    assert len(pt) == len(arr)
    assert pt.shape == arr.shape  # Shape of the coordinate array
    assert np.allclose(pt.coordinates, arr)  # Use allclose for float comparison
    assert isinstance(pt.system, CoordinateSystem)
    assert pt.system.name == sys_name


def check_vector(vec, p1_arr, p2_arr, sys_name):
    """Helper to check Vector/VectorArray properties."""
    assert isinstance(vec, (Vector, VectorArray))
    assert isinstance(vec.system, CoordinateSystem)
    assert vec.system.name == sys_name
    assert vec.shape == p1_arr.shape  # Shape of the underlying point arrays
    assert np.allclose(vec.p1.coordinates, p1_arr)
    assert np.allclose(vec.p2.coordinates, p2_arr)
    assert np.allclose(vec.displacement, p2_arr - p1_arr)
    if isinstance(vec, Vector):
        assert isinstance(vec.p1, Point)
        assert isinstance(vec.p2, Point)
    else:
        assert isinstance(vec.p1, PointArray)
        assert isinstance(vec.p2, PointArray)


# Use a fixture for a clean graph per test function if using pytest
# For unittest, manage graph cleanup if necessary (though default graph might be ok)
@pytest.fixture(autouse=True)
def clean_graph():
    # Ensure tests use a clean default graph instance
    CoordinateSystemGraph._graphs = {}
    yield
    CoordinateSystemGraph._graphs = {}


class PointTests(unittest.TestCase):
    def test_init(self):
        # Ensure default graph is clean
        CoordinateSystemGraph.get_graph(create=True)  # Creates 'default' if needed
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

        # System ndim check during creation
        sys3 = get_coordinate_system("sys3", ndim=3, create=True)
        assert sys3.ndim == 3
        with self.assertRaisesRegex(TypeError, "expected 4D"):
            Point([1, 2, 3, 4], "sys3")  # Now checks against existing system

        # Point must be 1d coordinate array
        with self.assertRaisesRegex(TypeError, "must be 1D"):
            Point([[1, 2, 3]], "sys3")

        # PointArray from list of int - system created implicitly
        data = [[1, 2], [3, 4], [5, 6]]
        sys2 = get_coordinate_system("sys2", ndim=2, create=True)
        arr = PointArray(data, "sys2")
        check_point(arr, np.array(data), "sys2")

        # PointArray from 1d list of Point
        pt1 = Point([1.0, 2.0], "sys2")
        pt2 = Point([3.0, 4.0], "sys2")
        pt3 = Point([5.0, 6.0], "sys2")
        data = [pt1, pt2, pt3]
        arr = PointArray(data, "sys2")
        # np.array(data) extracts coordinates correctly due to __array__
        check_point(arr, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "sys2")

        # PointArray from 2d list of Point
        data = [
            [Point([1.0, 2.0], "sys2"), Point([3.0, 4.0], "sys2")],
            [Point([5.0, 6.0], "sys2"), Point([7.0, 8.0], "sys2")],
        ]
        arr = PointArray(data, "sys2")
        check_point(arr, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).reshape(2, 2, 2), "sys2")
        # check that system is carried from Point
        arr = PointArray(data)
        check_point(arr, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).reshape(2, 2, 2), "sys2")
        # check conflicting system
        with self.assertRaisesRegex(TypeError, "System wrong_system does not match source coordinates sys2"):
            arr = PointArray(data, "wrong_system")

        # PointArray from 2x2 array of Point
        data = np.empty((2, 2), dtype=object)
        data[:] = [
            [Point([1.0, 2.0], "sys2"), Point([3.0, 4.0], "sys2")],
            [Point([5.0, 6.0], "sys2"), Point([7.0, 8.0], "sys2")],
        ]
        arr = PointArray(data, "sys2")
        check_point(arr, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).reshape(2, 2, 2), "sys2")
        # check that system is carried from Point
        arr = PointArray(data)
        check_point(arr, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).reshape(2, 2, 2), "sys2")
        # check conflicting system
        with self.assertRaisesRegex(TypeError, "System wrong_system does not match source coordinates sys2"):
            arr = PointArray(data, "wrong_system")

    def test_pickle(self):
        pt = Point([1, 2], "sys2")
        pt2 = pickle.loads(pickle.dumps(pt))
        assert pt == pt2
        assert pt.system is pt2.system  # Should reuse system instance

        arr = PointArray([[1, 2], [3, 4]], "sys2")
        arr2 = pickle.loads(pickle.dumps(arr))
        assert arr == arr2
        assert arr.system is arr2.system

    def test_equality(self):
        p1 = Point([1, 2], "sys2")
        p2 = Point([1, 2], "sys2")
        p3 = Point([1, 3], "sys2")
        p4 = Point([1, 2], "other_sys")
        a1 = PointArray([[1, 2]], "sys2")

        assert p1 == p2
        assert p1 != p3
        assert p1 != p4  # Different system
        assert p1 != a1  # Different type
        assert p1 != [1, 2]  # Different type


# ==== Vector Tests ====


class VectorTests(unittest.TestCase):
    def test_vector_init(self):
        p1 = Point([1, 2], "cartesian")
        p2 = Point([4, 6], "cartesian")
        v = Vector(p1, p2)
        check_vector(v, np.array([1, 2]), np.array([4, 6]), "cartesian")
        assert v.shape == (2,)  # Shape of the coordinate array for a single point
        assert v.ndim == 1  # ndim of the coordinate array
        assert len(v) == 2  # Length of the coordinate array

        # Mismatched systems
        p3 = Point([0, 0], "polar")
        with self.assertRaisesRegex(ValueError, "must share the same coordinate system"):
            Vector(p1, p3)

        # Wrong types
        with self.assertRaisesRegex(TypeError, "must be Point instances"):
            Vector(p1, [4, 6])
        with self.assertRaisesRegex(TypeError, "must be Point instances"):
            Vector(PointArray([[1, 2]], "cartesian"), p2)

    def test_vector_array_init(self):
        pa1 = PointArray([[1, 2], [0, 0]], "cartesian")
        pa2 = PointArray([[4, 6], [1, 1]], "cartesian")
        va = VectorArray(pa1, pa2)
        check_vector(va, np.array([[1, 2], [0, 0]]), np.array([[4, 6], [1, 1]]), "cartesian")
        assert va.shape == (2, 2)  # Shape of the point array
        assert va.ndim == 2  # ndim of the point array structure
        assert len(va) == 2  # Number of vectors

        # Mismatched systems
        pa3 = PointArray([[0, 0], [0, 0]], "polar")
        with self.assertRaisesRegex(ValueError, "must share the same coordinate system"):
            VectorArray(pa1, pa3)

        # Mismatched shapes
        pa4 = PointArray([[0, 0]], "cartesian")
        va_mismatched = VectorArray(pa1, pa4)
        check_vector(va_mismatched, np.array([[1, 2], [0, 0]]), np.array([[0, 0], [0, 0]]), "cartesian")

        # Wrong types
        with self.assertRaisesRegex(TypeError, "must be PointArray or Point instances"):
            VectorArray(pa1, [[4, 6], [1, 1]])

        # Can init with Points (they are PointArrays)
        p1 = Point([1, 2], "cartesian")
        p2 = Point([4, 6], "cartesian")
        va_from_points = VectorArray(p1, p2)  # Technically allowed, but Vector is preferred
        check_vector(va_from_points, np.array([1, 2]), np.array([4, 6]), "cartesian")

    def test_point_subtraction(self):
        p1 = Point([1, 2], "cartesian")
        p2 = Point([4, 6], "cartesian")
        p3 = Point([0, 0], "polar")

        # Point - Point
        v = p2 - p1
        assert isinstance(v, Vector)
        check_vector(v, p1.coordinates, p2.coordinates, "cartesian")

        # Mismatched systems
        with self.assertRaisesRegex(ValueError, "does not match"):
            p2 - p3

        # PointArray - PointArray
        pa1 = PointArray([[1, 2], [0, 0]], "cartesian")
        pa2 = PointArray([[4, 6], [1, 1]], "cartesian")
        va = pa2 - pa1
        assert isinstance(va, VectorArray)
        assert not isinstance(va, Vector)  # Ensure it's not the specific subclass
        check_vector(va, pa1.coordinates, pa2.coordinates, "cartesian")

        # PointArray - Point (broadcasts conceptually) -> VectorArray
        va2 = pa1 - p1
        assert isinstance(va2, VectorArray)
        # Expected p1 coords broadcasted to match pa1 shape for subtraction endpoint
        expected_p1_arr = np.array([[1, 2], [1, 2]])
        check_vector(va2, expected_p1_arr, pa1.coordinates, "cartesian")

        # Point - PointArray -> VectorArray
        va3 = p2 - pa1
        assert isinstance(va3, VectorArray)
        expected_p2_arr = np.array([[4, 6], [4, 6]])
        check_vector(va3, pa1.coordinates, expected_p2_arr, "cartesian")

    def test_vector_addition(self):
        p1 = Point([1, 2], "cartesian")
        p2 = Point([4, 6], "cartesian")
        p3 = Point([0, 5], "cartesian")
        p4 = Point([1, 0], "cartesian")
        p_polar = Point([0, 0], "polar")

        v1 = Vector(p1, p2)  # Displacement [3, 4]
        v2 = Vector(p3, p4)  # Displacement [1, -5]

        # Vector + Vector
        v_sum = v1 + v2
        assert isinstance(v_sum, Vector)
        # Expected displacement: [3, 4] + [1, -5] = [4, -1]
        # Expected endpoints: starts at v1.p1, ends at v1.p1 + combined_disp
        # p1 = [1, 2], combined_disp = [4, -1] -> p2_new = [5, 1]
        check_vector(v_sum, p1.coordinates, np.array([5, 1]), "cartesian")
        assert np.allclose(v_sum.displacement, [4, -1])

        # Mismatched systems
        v_polar = Vector(p_polar, p_polar)
        with self.assertRaisesRegex(ValueError, "does not match"):
            v1 + v_polar

        # VectorArray + VectorArray
        pa1 = PointArray([[1, 2], [0, 0]], "cartesian")
        pa2 = PointArray([[4, 6], [1, 1]], "cartesian")  # Disp: [[3, 4], [1, 1]]
        pa3 = PointArray([[0, 5], [-1, 0]], "cartesian")
        pa4 = PointArray([[1, 0], [-1, 2]], "cartesian")  # Disp: [[1, -5], [0, 2]]
        va1 = VectorArray(pa1, pa2)
        va2 = VectorArray(pa3, pa4)

        va_sum = va1 + va2
        assert isinstance(va_sum, VectorArray)
        # Expected displacement: [[3, 4], [1, 1]] + [[1, -5], [0, 2]] = [[4, -1], [1, 3]]
        # Expected endpoints: starts at va1.p1, ends at va1.p1 + combined_disp
        # p1 = [[1, 2], [0, 0]], combined_disp = [[4, -1], [1, 3]] -> p2_new = [[5, 1], [1, 3]]
        check_vector(va_sum, pa1.coordinates, np.array([[5, 1], [1, 3]]), "cartesian")
        assert np.allclose(va_sum.displacement, [[4, -1], [1, 3]])

        # Vector + VectorArray
        vsum_va = v1 + va1
        assert isinstance(vsum_va, VectorArray)
        # Disp v1: [3, 4]. Disp va1: [[3, 4], [1, 1]]
        # Combined: [[6, 8], [4, 5]] (broadcasting v1)
        # Starts at v1.p1 broadcasted: [[1, 2], [1, 2]]
        # Ends at start + combined: [[7, 10], [5, 7]]
        check_vector(vsum_va, np.array([[1, 2], [1, 2]]), np.array([[7, 10], [5, 7]]), "cartesian")

        # VectorArray + Vector
        vasum_v = va1 + v1
        assert isinstance(vasum_v, VectorArray)
        # Combined: [[6, 8], [4, 5]]
        # Starts at va1.p1: [[1, 2], [0, 0]]
        # Ends at start + combined: [[7, 10], [4, 5]]
        check_vector(vasum_v, np.array([[1, 2], [0, 0]]), np.array([[7, 10], [4, 5]]), "cartesian")

    def test_point_vector_addition(self):
        p1 = Point([1, 2], "cartesian")
        p2 = Point([4, 6], "cartesian")
        v1 = Vector(p1, p2)  # Displacement [3, 4]
        p_other = Point([-1, -1], "cartesian")
        p_polar = Point([0, 0], "polar")

        # Point + Vector
        p_new = p_other + v1
        assert isinstance(p_new, Point)
        # Expected coords: [-1, -1] + [3, 4] = [2, 3]
        check_point(p_new, np.array([2, 3]), "cartesian")

        # Vector + Point (using __radd__)
        p_new_r = v1 + p_other
        assert isinstance(p_new_r, Point)
        check_point(p_new_r, np.array([2, 3]), "cartesian")

        # Mismatched systems
        with self.assertRaisesRegex(ValueError, "does not match"):
            p_polar + v1
        with self.assertRaisesRegex(ValueError, "does not match"):
            v1 + p_polar

        # PointArray + VectorArray
        pa1 = PointArray([[1, 2], [0, 0]], "cartesian")
        pa2 = PointArray([[4, 6], [1, 1]], "cartesian")  # Disp: [[3, 4], [1, 1]]
        va1 = VectorArray(pa1, pa2)
        pa_other = PointArray([[10, 20], [30, 40]], "cartesian")

        pa_new = pa_other + va1
        assert isinstance(pa_new, PointArray)
        # Expected coords: [[10, 20], [30, 40]] + [[3, 4], [1, 1]] = [[13, 24], [31, 41]]
        check_point(pa_new, np.array([[13, 24], [31, 41]]), "cartesian")

        # VectorArray + PointArray
        pa_new_r = va1 + pa_other
        assert isinstance(pa_new_r, PointArray)
        check_point(pa_new_r, np.array([[13, 24], [31, 41]]), "cartesian")

        # PointArray + Vector
        pa_new_v = pa_other + v1  # v1 disp [3, 4] broadcasted
        assert isinstance(pa_new_v, PointArray)
        # Expected: [[10, 20], [30, 40]] + [[3, 4], [3, 4]] = [[13, 24], [33, 44]]
        check_point(pa_new_v, np.array([[13, 24], [33, 44]]), "cartesian")

        # Vector + PointArray
        pa_new_v_r = v1 + pa_other
        assert isinstance(pa_new_v_r, PointArray)
        check_point(pa_new_v_r, np.array([[13, 24], [33, 44]]), "cartesian")

        # Point + VectorArray
        p_new_va = p1 + va1  # p1 [1, 2] broadcasted
        assert isinstance(p_new_va, PointArray)
        # Expected: [[1, 2], [1, 2]] + [[3, 4], [1, 1]] = [[4, 6], [2, 3]]
        check_point(p_new_va, np.array([[4, 6], [2, 3]]), "cartesian")

        # VectorArray + Point
        p_new_va_r = va1 + p1
        assert isinstance(p_new_va_r, PointArray)
        check_point(p_new_va_r, np.array([[4, 6], [2, 3]]), "cartesian")

    def test_point_vector_subtraction(self):
        # Setup common points and vectors
        p_start = Point([10, 20], "cartesian")
        v_disp_p1 = Point([1, 2], "cartesian")
        v_disp_p2 = Point([4, 6], "cartesian")  # Displacement [3, 4]
        vector = Vector(v_disp_p1, v_disp_p2)

        pa_start = PointArray([[10, 20], [30, 40]], "cartesian")
        va_disp_p1 = PointArray([[1, 2], [0, 0]], "cartesian")
        va_disp_p2 = PointArray([[4, 6], [1, 1]], "cartesian")  # Displacements [[3,4], [1,1]]
        vector_array = VectorArray(va_disp_p1, va_disp_p2)

        p_polar = Point([0, 0], "polar")
        v_polar_disp_p1 = Point([0, 0], "polar")
        v_polar_disp_p2 = Point([1, 1], "polar")
        vector_polar = Vector(v_polar_disp_p1, v_polar_disp_p2)

        # Case 1: Point - Vector
        result_pv = p_start - vector
        assert isinstance(result_pv, Point)
        # Expected coords: [10, 20] - [3, 4] = [7, 16]
        check_point(result_pv, np.array([7, 16]), "cartesian")

        # Case 2: Point - VectorArray
        result_pva = p_start - vector_array
        assert isinstance(result_pva, PointArray)
        # Expected coords: [[10, 20], [10, 20]] - [[3, 4], [1, 1]] = [[7, 16], [9, 19]]
        check_point(result_pva, np.array([[7, 16], [9, 19]]), "cartesian")

        # Case 3: PointArray - Vector
        result_pav = pa_start - vector
        assert isinstance(result_pav, PointArray)
        # Expected coords: [[10, 20], [30, 40]] - [[3, 4], [3, 4]] = [[7, 16], [27, 36]]
        check_point(result_pav, np.array([[7, 16], [27, 36]]), "cartesian")

        # Case 4: PointArray - VectorArray
        result_pava = pa_start - vector_array
        assert isinstance(result_pava, PointArray)
        # Expected coords: [[10, 20], [30, 40]] - [[3, 4], [1, 1]] = [[7, 16], [29, 39]]
        check_point(result_pava, np.array([[7, 16], [29, 39]]), "cartesian")

        # Error handling: Mismatched systems
        with self.assertRaisesRegex(ValueError, "does not match this PointArray's system"):
            p_start - vector_polar
        with self.assertRaisesRegex(ValueError, "does not match this PointArray's system"):
            pa_start - vector_polar

        # Error handling: Type errors (subtracting non-vector from point)
        # This should be caught by PointArray.__sub__ if the operand is not PointArray or VectorArray
        with self.assertRaisesRegex(TypeError, "Unsupported operand type"):  # Or similar, depending on implementation
            p_start - np.array([1, 2])
        with self.assertRaisesRegex(TypeError, "Unsupported operand type"):
            pa_start - np.array([[1, 2]])

    def test_vector_pickle(self):
        p1 = Point([1, 2], "cartesian")
        p2 = Point([4, 6], "cartesian")
        v = Vector(p1, p2)
        v2 = pickle.loads(pickle.dumps(v))
        assert v == v2
        assert v.system is v2.system  # System should be reused

        pa1 = PointArray([[1, 2], [0, 0]], "cartesian")
        pa2 = PointArray([[4, 6], [1, 1]], "cartesian")
        va = VectorArray(pa1, pa2)
        va2 = pickle.loads(pickle.dumps(va))
        assert va == va2
        assert va.system is va2.system

    def test_vector_equality(self):
        p1 = Point([1, 2], "sys")
        p2 = Point([3, 4], "sys")
        p3 = Point([1, 2], "sys")  # Same coords as p1
        p4 = Point([3, 5], "sys")  # Different coords from p2
        p5 = Point([1, 2], "other_sys")
        p6 = Point([3, 4], "other_sys")

        v1 = Vector(p1, p2)
        v2 = Vector(p1, p2)  # Identical
        v3 = Vector(p3, p2)  # Same start point value, different object
        v4 = Vector(p1, p4)  # Different end point
        v5 = Vector(p5, p6)  # Different system

        va1 = VectorArray(PointArray([p1.coordinates]), PointArray([p2.coordinates]))

        assert v1 == v2
        assert v1 == v3  # Equality based on endpoint values
        assert v1 != v4
        assert v1 != v5  # Different system via endpoints
        assert v1 != va1  # Different type
        assert v1 != [2, 2]  # Different type (displacement)

    def test_vector_ndarray_init(self):
        """Test initializing Vector/VectorArray with ndarray and system."""
        # Simple 1D vector from displacement array
        disp = np.array([3, 4])
        v = Vector(disp, "cartesian")
        assert isinstance(v, Vector)

        # Verify p1 is at origin and p2 is at the displacement
        check_vector(v, np.zeros_like(disp), disp, "cartesian")

        # Check internal structure
        assert isinstance(v.p1, PointArray)
        assert isinstance(v.p2, PointArray)
        assert np.allclose(v.p1.coordinates, np.zeros_like(disp))
        assert np.allclose(v.p2.coordinates, disp)
        assert np.allclose(v.displacement, disp)

        # Multi-dimensional displacement array
        multi_disp = np.array([[1, 2], [3, 4], [5, 6]])
        va = VectorArray(multi_disp, "cartesian")
        assert isinstance(va, VectorArray)

        # Verify points and displacement
        zeros = np.zeros_like(multi_disp)
        check_vector(va, zeros, multi_disp, "cartesian")
        assert np.allclose(va.displacement, multi_disp)

        # Test with float dtype
        float_disp = np.array([1.5, 2.5])
        vf = VectorArray(float_disp, "cartesian")
        assert vf.dtype == float_disp.dtype
        check_vector(vf, np.zeros_like(float_disp), float_disp, "cartesian")

        # Test with 3D coordinate space
        disp3d = np.array([1, 2, 3])
        v3d = VectorArray(disp3d, "cartesian3d")
        check_vector(v3d, np.zeros_like(disp3d), disp3d, "cartesian3d")

        # Edge case: empty array
        empty_disp = np.array([[]])
        with pytest.raises(ValueError):  # Should fail gracefully
            VectorArray(empty_disp, "cartesian")
