import json
import math
import pickle
import unittest

import numpy as np

import coorx
from coorx import LogTransform


try:
    import itk

    HAVE_ITK = True
except ImportError:
    HAVE_ITK = False


try:
    import vispy

    HAVE_VISPY = True
except ImportError:
    HAVE_VISPY = False


try:
    import pyqtgraph as pg

    HAVE_PG = True
except ImportError:
    HAVE_PG = False


NT = coorx.NullTransform
TT = coorx.TTransform
XT = coorx.TransposeTransform
ST = coorx.STTransform
AT = coorx.AffineTransform
RT = coorx.AffineTransform
PT = coorx.PolarTransform
LT = coorx.LogTransform
CT = coorx.CompositeTransform
T3 = coorx.SRT3DTransform
ZT = coorx.AxisSelectionEmbeddedTransform
HT = coorx.HomogeneousEmbeddedTransform
DT = coorx.LensDistortionTransform
QT = coorx.PerspectiveTransform
BT = coorx.BilinearTransform
T2 = coorx.Homography2DTransform


def assert_composite_types(composite, types):
    assert list(map(type, composite.transforms)) == types


def assert_composite_objects(composite1, composite2):
    assert composite1.transforms == composite2.transforms


def test_pickling():
    points = np.random.normal(size=(10, 3))
    transforms = [
        NT(dims=(3, 3)),
        TT(dims=(3, 3), offset=(1, 2, 3)),
        ST(dims=(3, 3), scale=(2, 3, 4), offset=(1, 2, 3)),
        AT(dims=(3, 3)),
        PT(dims=(3, 3)),
        XT(axis_order=(2, 1, 0)),
        LT(dims=(3, 3), base=(10, None, 2)),
        CT([ST(dims=(3, 3), scale=(2, 2, 2)), TT(dims=(3, 3), offset=(1, 1, 1))]),
        T3(dims=(3, 3), scale=(2, 2, 2), offset=(1, 2, 3), angle=45, axis=(0, 0, 1)),
        ZT(axes=(0, 2), transform=ST(dims=(2, 2), scale=(3, 3), offset=(1, 1)), dims=(3, 3)),
        HT(transform=AT(dims=(4, 4)), dims=(3, 3)),
        QT(
            dims=(3, 3),
            affine={
                'type': 'AffineTransform',
                'matrix': [[1, 0, 0, 0], [0, 1, 0, 0], [0.001, 0.002, 1, 0], [0, 0, 0, 1]],
                'offset': [0, 0, 0, 0],
            },
        ),
        # DT(dims=(2, 2), coeff=(0.1, -0.05, 0.01, 0.0, 0.0)),
        # BT(dims=(2, 2)),
        # T2(dims=(2, 2), src_quad=[(0, 0), (1, 0), (1, 1), (0, 1)], dst_quad=[(0, 0), (2, 0), (1, 1), (0, 2)]),
    ]
    for tr in transforms:
        tr2 = pickle.loads(pickle.dumps(tr))
        assert tr == tr2
        mapped1 = tr.map(points)
        mapped2 = tr2.map(points)
        assert_equal_including_nans(mapped1, mapped2, f"Mapping differs for pickled {type(tr)}")

        alt_tr2 = coorx.create_transform(**json.loads(json.dumps(tr.save_state())))
        assert tr == alt_tr2
        mapped2b = alt_tr2.map(points)
        assert_equal_including_nans(
            mapped1, mapped2b, f"Mapping differs for create_transform'd {type(tr)}"
        )

        try:
            tr.inverse.map(tr.map(points))
        except (np.linalg.LinAlgError, NotImplementedError):
            continue  # non-invertible

        inv_tr2 = pickle.loads(pickle.dumps(tr.inverse))
        assert tr.inverse == inv_tr2
        inv_mapped1 = tr.inverse.map(mapped1)
        inv_mapped2 = inv_tr2.map(mapped2)
        assert_equal_including_nans(
            inv_mapped1, inv_mapped2, f"Inverse mapping differs for pickled {type(tr)}"
        )

        alt_inv_tr2 = coorx.create_transform(**eval(str(tr.inverse.save_state())))
        assert tr.inverse == alt_inv_tr2
        alt_inv_mapped2 = alt_inv_tr2.map(mapped2)
        assert_equal_including_nans(
            inv_mapped1,
            alt_inv_mapped2,
            f"Inverse mapping differs for create_transform'd {type(tr)}",
        )


def assert_equal_including_nans(a, b, msg):
    assert np.all(
        (np.isnan(a) == np.isnan(b)) & (np.isclose(a, b) | (np.isnan(a) & np.isnan(b)))
    ), msg
    mask = ~np.isnan(a)
    assert np.allclose(a[mask], b[mask]), msg


class TransformMultiplication(unittest.TestCase):
    def test_multiplication(self):
        n = NT(dims=(3, 3))
        t = TT(dims=(3, 3))
        s = ST(dims=(3, 3))
        a = AT(dims=(3, 3))
        p = PT(dims=(3, 3))
        x = XT(axis_order=(2, 1, 0))
        log_trans = LT(dims=(3, 3))
        c1 = CT([s, a, p, x])
        assert c1
        c2 = CT([s, a, s])

        assert isinstance(n * n, NT)
        assert isinstance(n * t, TT)
        assert isinstance(n * s, ST)
        assert isinstance(n * p, PT)
        assert isinstance(t * t, TT)
        assert isinstance(x * x, XT)
        assert isinstance(t * s, ST)
        assert isinstance(t * a, AT)
        assert isinstance(s * t, ST)
        assert isinstance(s * s, ST)
        assert isinstance(s * a, AT)
        assert isinstance(s * p, CT)
        assert isinstance(a * t, AT)
        assert isinstance(a * s, AT)
        assert isinstance(a * a, AT)
        assert isinstance(a * p, CT)
        assert isinstance(p * a, CT)
        assert isinstance(p * s, CT)
        assert isinstance(s * x, CT)
        assert isinstance(x * a, CT)
        assert isinstance(t * x, CT)
        assert isinstance(x.inverse * t, CT)
        assert_composite_types(p * a, [AT, PT])
        assert_composite_types(p * s, [ST, PT])
        assert_composite_types(s * p, [PT, ST])
        assert_composite_types(s * p * a, [AT, PT, ST])
        assert_composite_types(s * a * p, [PT, AT])
        assert_composite_types(p * s * a, [AT, ST, PT])
        assert_composite_types(s * p * s, [ST, PT, ST])
        assert_composite_types(s * a * p * x * s * a, [AT, ST, XT, PT, AT])
        assert_composite_types(c2 * a, [AT, ST, AT, ST])
        assert_composite_types(p * log_trans * s, [ST, LT, PT])


class VectorMapping(unittest.TestCase):
    @unittest.skipIf(not HAVE_PG, "pyqtgraph could not be imported")
    def test_vector_map(self):
        vector = pg.Vector(1, 2, 3)
        t = coorx.STTransform(scale=(2, 3, 4), offset=(5, 6, 7))
        mapped = t.map(vector)
        assert isinstance(mapped, pg.Vector)
        self.assertEqual(mapped.x(), 1 * 2 + 5)
        self.assertEqual(mapped.y(), 2 * 3 + 6)
        self.assertEqual(mapped.z(), 3 * 4 + 7)


class CompositeTransform(unittest.TestCase):
    def test_transform_composite(self):
        # Make dummy classes for easier distinguishing the transforms

        class DummyTrans(coorx.Transform):
            def _validate_dims(self, dims, **kwargs):
                return dims

        class TransA(DummyTrans):
            pass

        class TransB(DummyTrans):
            pass

        class TransC(DummyTrans):
            pass

        # Create test transforms
        a, b, c = TransA(), TransB(), TransC()

        # Test Composite creation
        assert coorx.CompositeTransform().transforms == []
        assert coorx.CompositeTransform([a]).transforms == [a]
        assert coorx.CompositeTransform([a, b]).transforms == [a, b]
        assert coorx.CompositeTransform([a, b, c, a]).transforms == [a, b, c, a]

        # Test composition by multiplication
        assert_composite_objects(a * b, coorx.CompositeTransform([b, a]))
        assert_composite_objects(a * b * c, coorx.CompositeTransform([c, b, a]))
        assert_composite_objects(a * b * c * a, coorx.CompositeTransform([a, c, b, a]))

        # Test adding/prepending to transform
        composite = coorx.CompositeTransform()
        composite.append(a)
        assert composite.transforms == [a]
        composite.append(b)
        assert composite.transforms == [a, b]
        composite.append(c)
        assert composite.transforms == [a, b, c]
        composite.prepend(b)
        assert composite.transforms == [b, a, b, c]
        composite.prepend(c)
        assert composite.transforms == [c, b, a, b, c]

        # Test simplifying
        t1 = coorx.STTransform(scale=(2, 3))
        t2 = coorx.STTransform(offset=(3, 4))
        t3 = coorx.STTransform(offset=(3, 4))
        # Create multiplied versions
        t123 = t1 * t2 * t3
        t321 = t3 * t2 * t1
        c123 = coorx.CompositeTransform([t3, t2, t1])
        c321 = coorx.CompositeTransform([t1, t2, t3])
        c123s = c123.simplified
        c321s = c321.simplified

        assert isinstance(t123, coorx.STTransform)
        assert isinstance(t321, coorx.STTransform)
        assert isinstance(c123s, coorx.CompositeTransform)
        assert isinstance(c321s, coorx.CompositeTransform)

        # Test Mapping
        t1 = coorx.STTransform(scale=(2, 3))
        t2 = coorx.STTransform(offset=(3, 4))
        composite12 = coorx.CompositeTransform([t1, t2])
        composite21 = coorx.CompositeTransform([t2, t1])

        assert composite12.transforms == [t1, t2]
        assert composite21.transforms == [t2, t1]

        m12 = (t2 * t1).map((1, 1)).tolist()
        m21 = (t1 * t2).map((1, 1)).tolist()
        m12_ = composite12.map((1, 1)).tolist()
        m21_ = composite21.map((1, 1)).tolist()

        assert m12 != m21
        assert m12 == m12_
        assert m21 == m21_

        # test pickle
        s = pickle.dumps(composite12)
        assert pickle.loads(s) == composite12

    def test_srt_composites(self):
        s = coorx.STTransform(scale=(2, 3, 1))
        t = coorx.TTransform(offset=(3, 4, 5))
        srt = coorx.SRT3DTransform(scale=(2, 3, 1), offset=(3, 4, 5), angle=90, axis=(0, 0, 1))

        assert isinstance(srt, coorx.SRT3DTransform)
        assert isinstance(s * t, coorx.STTransform)
        assert isinstance(t * s, coorx.STTransform)
        assert isinstance(s * srt, coorx.CompositeTransform)
        assert isinstance(srt * s, coorx.CompositeTransform)
        assert isinstance(t * srt, coorx.CompositeTransform)
        assert isinstance(srt * t, coorx.CompositeTransform)

    def test_inverse_composite(self):
        # Test inverse of composite
        t1 = coorx.STTransform(scale=(2, 3))
        t2 = coorx.STTransform(offset=(3, 4))
        composite = coorx.CompositeTransform([t1, t2])
        composite_inv = composite.inverse

        assert composite_inv.map(composite.map((1, 1))).tolist() == [1, 1]

    def test_map_order(self):
        t1 = coorx.SRT3DTransform(scale=(2, 3, 1), offset=(3, 4, 5), angle=90, axis=(0, 0, 1))
        t2 = coorx.TTransform(offset=(3, 4, 5))
        t3 = coorx.TTransform(offset=(13, 4.1, 7)).inverse
        composite = t1 * t2 * t3
        explicit = coorx.CompositeTransform([t3, t2, t1])
        assert composite.map((1, 1, 1)).tolist() == t1.map(t2.map(t3.map((1, 1, 1)))).tolist()
        assert composite.map((1, 1, 1)).tolist() == explicit.map((1, 1, 1)).tolist()

        composite = t3 * t2 * t1
        explicit = coorx.CompositeTransform([t1, t2, t3])
        assert composite.map((1, 1, 1)).tolist() == t3.map(t2.map(t1.map((1, 1, 1)))).tolist()
        assert composite.map((1, 1, 1)).tolist() == explicit.map((1, 1, 1)).tolist()

    def test_as_affine(self):
        t1 = coorx.STTransform(scale=(2, 3, 1))
        t2 = coorx.TTransform(offset=(3, 4, 0))
        t3 = coorx.AffineTransform(dims=(3, 3))
        t3.rotate(90, (0, 0, 1))
        t3 = t3.inverse
        t4 = coorx.NullTransform(dims=3)
        composite = coorx.CompositeTransform([t1, t2, t3, t4])
        affine = composite.as_affine()
        assert isinstance(affine, coorx.AffineTransform)

    def test_full_matrix(self):
        t1 = coorx.STTransform(scale=(2, 3, 5))
        t2 = coorx.STTransform(offset=(3, 4, 2))
        composite = coorx.CompositeTransform([t1, t2.inverse])
        assert np.allclose(composite.full_matrix, np.dot(t2.inverse.full_matrix, t1.full_matrix))

    def test_save_state(self):
        t1 = coorx.STTransform(scale=(2, 3, 5))
        t2 = coorx.STTransform(offset=(3, 4, 0))
        matrix = np.eye(3)
        matrix[:2, :2] = [[0, -0.1], [0.1, 0]]
        t3 = coorx.AffineTransform(matrix, offset=(1, 2, 3))
        data = np.random.normal(size=(10, 3))
        composite = coorx.CompositeTransform([t1, t2, t3])
        state = eval(str(composite.save_state()))
        assert state['type'] == 'CompositeTransform'

        composite2 = coorx.create_transform(**state)
        assert composite == composite2
        assert np.allclose(composite.map(data), composite2.map(data))

    @unittest.skipIf(not HAVE_VISPY, "vispy could not be imported")
    def test_as_vispy(self):
        from vispy.visuals.transforms import ChainTransform

        t1 = coorx.STTransform(scale=(2, 3, 5))
        t2 = coorx.STTransform(offset=(3, 4, 0))
        composite = coorx.CompositeTransform([t1, t2])
        as_vispy = composite.as_vispy()
        assert isinstance(as_vispy, ChainTransform)
        assert np.allclose(as_vispy.map((1, 1, 1))[:3], composite.map((1, 1, 1)))


class TTransform(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.transforms = [
            TT(dims=(3, 3)),
            TT([1]),
            TT([0, 0]),
            TT(np.array([1, 1e16])),
            TT((-100e-6, 12e8, 0)),
            TT(np.random.normal(size=10)),
        ]
        self.points = [
            np.random.normal(size=(10, 3)),
            10 ** np.random.normal(size=(10, 3)),
            -(10 ** np.random.normal(size=(10, 3))),
            [2, 3, 4],
            [10],
            (5, 6),
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        ]

    def test_t_transform(self):
        # Check that TTransform maps exactly like AffineTransform
        pts = np.random.normal(size=(10, 3))
        translate = (1e6, 0.2, 0)
        mapped = pts + translate
        imapped = pts - translate
        tt = coorx.TTransform(offset=translate)

        assert np.allclose(tt.map(pts), mapped)
        assert np.allclose(tt.inverse.map(pts), imapped)

        # test save/restore
        tt2 = coorx.TTransform(dims=(3, 3))
        tt2.__setstate__(tt.__getstate__())
        assert np.allclose(tt.map(pts), tt2.map(pts))

        tt3 = pickle.loads(pickle.dumps(tt))
        assert np.allclose(tt.map(pts), tt3.map(pts))

    @unittest.skipIf(not HAVE_ITK, "itk could not be imported")
    def test_itk_compat(self):
        itk_tr = itk.TranslationTransform[itk.D, 3].New()
        ttr = TT(dims=(3, 3))

        pts = 10 ** np.random.normal(size=(20, 3), scale=16)
        offsets = 10 ** np.random.normal(size=(20, 3), scale=16)

        for offset in offsets:
            ttr_pts = ttr.map(pts)
            for i in range(len(pts)):
                itk_tr_pt = np.array(itk_tr.TransformPoint(itk.Point[itk.D, 3](pts[i])))
                assert np.allclose(itk_tr_pt, ttr_pts[i])
            ttr.translate(offset)
            itk_tr.Translate(itk.Point[itk.D, 3](offset))
            assert np.allclose(ttr.offset, np.array(itk_tr.GetOffset()))

    def test_inverse(self):
        for tr in self.transforms:
            for pts in self.points:
                arr = np.array(pts)
                if arr.shape[-1] == tr.dims[0]:
                    pts2 = tr.map(pts)
                    assert pts2.shape[-1] == tr.dims[1]
                    pts3 = tr.imap(pts2)
                    assert np.allclose(arr, pts3)
                else:
                    with self.assertRaises(TypeError):
                        tr.map(pts)


class STTransform(unittest.TestCase):
    def test_st_transform(self):
        # Check that STTransform maps exactly like AffineTransform
        pts = np.random.normal(size=(10, 3))

        scale = (1, 7.5, -4e-8)
        translate = (1e6, 0.2, 0)
        st = coorx.STTransform(scale=scale, offset=translate)
        at = coorx.AffineTransform(dims=(3, 3))
        at.zoom(scale)
        at.translate(translate)

        assert np.allclose(st.map(pts), at.map(pts))
        assert np.allclose(st.inverse.map(pts), at.inverse.map(pts))

    def test_st_mapping(self):
        p1 = [[5.0, 7.0], [23.0, 8.0]]
        p2 = [[-1.3, -1.4], [1.1, 1.2]]

        t = coorx.STTransform(dims=(2, 2))
        t.set_mapping(p1, p2)

        assert np.allclose(t.map(p1)[:, : len(p2)], p2)


def check_matrix(t, m):
    dim = m.shape[0] - 1
    assert np.allclose(t.full_matrix, m)
    assert np.allclose(t.matrix, m[:dim, :dim])
    assert np.allclose(t.offset, m[:dim, dim])


class AffineTransform(unittest.TestCase):
    def test_modifiers(self):
        t = coorx.AffineTransform(dims=(3, 3))
        m = np.eye(4)
        check_matrix(t, m)

        t.translate(1)
        m[:3, 3] += 1
        check_matrix(t, m)

        t.zoom(2)
        m[:3] *= 2
        check_matrix(t, m)

        t.translate(3)
        m[:3, 3] += 3
        check_matrix(t, m)

        t.rotate(90, (0, 0, 1))
        m[:2] = [[0, 2, 0, 5], [-2, 0, 0, -5]]
        check_matrix(t, m)

        rm = coorx.AffineTransform(dims=(3, 3))
        rm.rotate(90, (0, 0, 1))
        t2 = (
            rm
            * coorx.TTransform([3, 3, 3])
            * coorx.STTransform(scale=[2, 2, 2])
            * coorx.TTransform([1, 1, 1])
        )
        assert t2 == t

    def test_4d(self):
        t = coorx.AffineTransform(dims=(4, 4))
        m = np.eye(5)
        check_matrix(t, m)

        t.translate([1, 2, 3, 4])
        m[:4, 4] += [1, 2, 3, 4]
        check_matrix(t, m)

        t.zoom([2, 3, 4, 5])
        for i in range(4):
            m[i] *= [2, 3, 4, 5][i]
        check_matrix(t, m)

        t.translate([5, 6, 7, 8])
        m[:4, 4] += [5, 6, 7, 8]
        check_matrix(t, m)

        inv = t.inverse
        a = np.random.normal(size=(10, 4))
        assert np.allclose(inv.map(t.map(a)), a)

    def test_changing_dims(self):
        matrix = [[1, 0, 1], [0, 1, 1]]
        with self.assertRaises(ValueError):
            t = AT(matrix=matrix, dims=3)
        with self.assertRaises(ValueError):
            t = AT(matrix=matrix, dims=(2, 2))
        with self.assertRaises(ValueError):
            t = AT(matrix=matrix, offset=[1, 2, 3])

        t = AT(matrix=matrix, offset=[1, 2])
        assert t.dims == (3, 2)

        with self.assertRaises(ValueError):
            t.matrix = np.eye(3)

        pts = np.random.normal(size=(10, 3))
        mapped = t.map(pts)
        assert mapped.shape == (10, 2)

        with self.assertRaises(np.linalg.LinAlgError):
            t.imap(mapped)

    def test_inverse(self):
        tr = AT(dims=(3, 3), matrix=np.eye(3) * 0.1, offset=[1, 2, 3])
        pts = np.random.normal(size=(10, 3))
        arr = np.array(pts)
        if arr.shape[-1] == tr.dims[0]:
            pts2 = tr.map(pts)
            assert pts2.shape[-1] == tr.dims[1]
            pts3 = tr.imap(pts2)
            assert np.allclose(arr, pts3)
        else:
            with self.assertRaises(TypeError):
                tr.map(pts)

    def test_uninvertible(self):
        tr = AT(dims=(2, 2), matrix=[[1, 1], [2, 2]], offset=[0, 0])
        pts = np.random.normal(size=(10, 2))
        with self.assertRaises(np.linalg.LinAlgError):
            tr.imap(pts)
        with self.assertRaises(np.linalg.LinAlgError):
            tr.inverse.map(pts)

    def x_test_affine_mapping(self):
        t = coorx.AffineTransform()
        p1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # test pure translation
        p2 = p1 + 5.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, : p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test pure scaling
        p2 = p1 * 5.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, : p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.zoom(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test scale + translate
        p2 = (p1 * 5.5) + 3.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, : p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.zoom(3.5)
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test scale + translate + rotate
        p2 = np.array([[10, 5, 3], [10, 15, 3], [30, 5, 3], [10, 5, 3.5]])
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, : p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.zoom(3.5)
        t2.rotate(90)
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

    def test_to_and_from_qmatrix4x4(self):
        """Test conversion to and from QMatrix4x4 objects."""
        # Skip test if no Qt is available
        try:
            from coorx.qt import import_qt_gui

            import_qt_gui()
        except ImportError:
            self.skipTest("Qt GUI module not available")

        # Create a test transform with rotation, scaling, and translation
        tr = coorx.AffineTransform(dims=(3, 3))
        tr.zoom((2, 3, 0.5))
        tr.rotate(45, (0, 0, 1))
        tr.translate((10, 20, 30))

        # Convert to QMatrix4x4 and back
        qmatrix = tr.to_qmatrix4x4()
        tr2 = coorx.AffineTransform.from_qmatrix4x4(qmatrix)

        # Check that full matrices are identical
        assert np.allclose(tr.full_matrix, tr2.full_matrix)

        # Check that transforms produce identical results
        pts = np.random.normal(size=(10, 3))
        assert np.allclose(tr.map(pts), tr2.map(pts))


class TransposeTransformTest(unittest.TestCase):
    def test_inverse_and_map(self):
        tt = XT(axis_order=(2, 1, 0))
        pts = np.random.normal(size=(10, 3))
        assert np.allclose(tt.inverse.map(tt.map(pts)), pts)
        assert np.allclose(tt.map(pts), pts[..., ::-1])


class LogTransformTest(unittest.TestCase):
    def test_log(self):
        lt = LogTransform((12, None))  # None is the identity value
        data = [(12, -6), (144, 13.2), (float("inf"), 21), (-7, 44), (0, 0)]
        output = lt.map(data)
        self.assertAlmostEqual(output[0][0], 1)
        self.assertAlmostEqual(output[1][0], 2)
        self.assertAlmostEqual(output[1][1], 13.2)  # None = identity, so unchanged
        self.assertTrue(math.isnan(output[2][0]))
        self.assertTrue(math.isnan(output[3][0]))
        self.assertTrue(math.isnan(output[4][0]))
        self.assertAlmostEqual(output[4][1], 0)  # None = identity, so unchanged


class SRT3DTransformTest(unittest.TestCase):
    def test_srt3d(self):
        pts = np.random.normal(size=(10, 3))

        tr = coorx.SRT3DTransform()
        aff = coorx.AffineTransform(dims=(3, 3))
        assert np.allclose(pts, tr.map(pts))

        scale = [10, 1, 0.1]
        tr.scale = scale
        aff.zoom(scale)
        assert np.allclose(aff.map(pts), tr.map(pts))

        angle = 30
        axis = np.array([1, 0.5, 0.3])
        tr.rotation = angle, axis
        aff.rotate(angle, axis)
        assert np.allclose(aff.map(pts), tr.map(pts))
        assert tr.rotation[0] == angle
        assert np.allclose(tr.rotation[1], axis)

        offset = [1e-6, -10, 1e6]
        tr.offset = offset
        aff.translate(offset)
        assert np.allclose(aff.map(pts), tr.map(pts))

    @unittest.expectedFailure
    def test_from_affine(self):
        # TODO this was previously part of test_srt3d, but it breaks trying to initialize tr2
        tr2 = coorx.SRT3DTransform(init=aff)
        assert np.allclose(tr.offset, tr2.offset)
        assert np.allclose(tr.scale, tr2.scale)
        assert tr.rotation[0] == tr2.rotation[0]
        assert np.allclose(tr.rotation[1], tr2.rotation[1])
        assert np.allclose(tr2.map(pts), tr.map(pts))

    def test_save(self):
        tr = coorx.SRT3DTransform(scale=(1, 2, 3), offset=(10, 5, 3), angle=120, axis=(1, 1, 2))
        s = eval(str(tr.save_state()))
        assert s["type"] == "SRT3DTransform"
        assert tuple(s["dims"]) == (3, 3)

    @unittest.skipIf(not HAVE_VISPY, "vispy could not be imported")
    def test_as_vispy(self):
        tr = coorx.SRT3DTransform(scale=(1, 2, 3), offset=(10, 5, 3), angle=120, axis=(1, 1, 2))
        vt = tr.as_vispy()
        assert np.allclose(vt.map((1, 1, 1))[:3], tr.map((1, 1, 1)))
        assert np.allclose(vt.map((1, 3, 5))[:3], tr.map((1, 3, 5)))

    @unittest.skipIf(not HAVE_PG, "pyqtgraph could not be imported")
    def test_to_and_from_pyqtgraph(self):
        axis = np.array((1, 1, 2))
        axis = axis / np.linalg.norm(axis)
        tr = coorx.SRT3DTransform(scale=(1, 2, 3), offset=(10, 5, 3), angle=120, axis=axis)
        tr2 = coorx.SRT3DTransform.from_pyqtgraph(tr.as_pyqtgraph())
        assert np.allclose(tr.full_matrix, tr2.full_matrix)
        pt = np.random.normal(size=(10, 3))
        assert np.allclose(tr.map(pt), tr2.map(pt))

    def test_composite(self):
        tr1 = coorx.SRT3DTransform(offset=(1, 2, 3))
        tr2 = coorx.SRT3DTransform(scale=(10, 10, 1))

        # test multiplication
        tr3 = tr1 * tr2  # scale, then offset
        assert isinstance(tr3, coorx.AffineTransform)
        assert np.allclose(tr3.map([0, 0, 0]), [1, 2, 3])
        assert np.allclose(tr3.map([1, 1, 1]), [11, 12, 4])

        # test composite
        tr4 = coorx.CompositeTransform([tr2, tr1])
        assert len(tr4.simplified.transforms) == 1
        assert isinstance(tr4.simplified.transforms[0], coorx.AffineTransform)
        assert np.all(tr3.matrix == tr4.simplified.transforms[0].matrix)
        assert np.all(tr3.offset == tr4.simplified.transforms[0].offset)
        assert np.all(tr3.full_matrix == tr4.simplified.transforms[0].full_matrix)
        assert np.allclose(tr4.map([2, -26.7, 0]), tr3.map([2, -26.7, 0]))

    def test_composite_setitem(self):
        tr1 = coorx.SRT3DTransform(offset=(1, 2, 3))
        tr2 = coorx.SRT3DTransform(scale=(10, 10, 1))
        tr3 = coorx.SRT3DTransform(scale=(1, 11, 111))

        tr4 = coorx.CompositeTransform([tr2, tr1])
        tr4[0] = tr3
        assert np.all(tr4.transforms[0].full_matrix == tr3.full_matrix)


class TransformInverse(unittest.TestCase):
    def test_inverse(self):
        m = np.random.normal(size=(3, 3))
        transforms = [
            NT(dims=3),
            ST(scale=(1e-4, 2e5, 1), offset=(10, -6e9, 0)),
            AT(m),
            RT(m),
        ]

        np.random.seed(0)
        N = 20
        x = np.random.normal(size=(N, 3))
        pw = np.random.normal(size=(N, 3), scale=3)
        pos = x * 10**pw

        for trn in transforms:
            assert np.allclose(pos, trn.inverse.map(trn.map(pos))[:, :3])

        # log transform only works on positive values
        # abs_pos = np.abs(pos)
        # tr = LT(base=(2, 4.5, 0))
        # assert np.allclose(abs_pos, tr.inverse.map(tr.map(abs_pos))[:,:3])


class BilinearTest(unittest.TestCase):
    def test_bilinear(self):
        # identity
        tr = self.check_mapping([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 0], [1, 0], [0, 1], [1, 1]])
        assert np.allclose(tr.matrix, np.eye(4)[1:3])

        tr = self.check_mapping([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 0], [1, 1], [0, 1], [1, 0]])

        tr = self.check_mapping([[0, 0], [1, 0], [0, 1], [1, 1]], [[1, 1], [3, 0], [0, 4], [7, 7]])

    def check_mapping(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)

        tr = coorx.linear.BilinearTransform()
        tr.set_mapping(a, b)
        c = tr.map(a)
        assert c.shape == b.shape
        assert np.allclose(c, b)

        assert np.allclose(tr.map(a.mean(axis=0)), b.mean(axis=0))

        tr_inv = tr.inverse
        d = tr_inv.map(b)
        assert d.shape == a.shape
        assert np.allclose(d, a)

        # Note: there are cases where a bilinear transform is not invertible
        # assert np.allclose(
        #     tr_inv.map(b.mean(axis=0)),
        #     a.mean(axis=0)
        # )

        assert np.all(tr_inv.map(b) == tr.imap(b))

        return tr
