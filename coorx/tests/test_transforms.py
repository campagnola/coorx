import unittest
import pickle
import numpy as np

try:
    import itk
    HAVE_ITK = True
except ImportError:
    HAVE_ITK = False

import coorx

NT = coorx.NullTransform
TT = coorx.TTransform
ST = coorx.STTransform
AT = coorx.AffineTransform
RT = coorx.AffineTransform
PT = coorx.PolarTransform
LT = coorx.LogTransform
CT = coorx.CompositeTransform


def assert_composite_types(composite, types):
    assert list(map(type, composite.transforms)) == types


def assert_composite_objects(composite1, composite2):
    assert composite1.transforms == composite2.transforms


class TransformMultiplication(unittest.TestCase):
    def test_multiplication(self):
        n = NT(dims=(3, 3))
        t = TT(dims=(3, 3))
        s = ST(dims=(3, 3))
        a = AT(dims=(3, 3))
        p = PT(dims=(3, 3))
        log_trans = LT(dims=(3, 3))
        c1 = CT([s, a, p])
        assert c1
        c2 = CT([s, a, s])

        assert isinstance(n * n, NT)
        assert isinstance(n * t, TT)
        assert isinstance(n * s, ST)
        assert isinstance(n * p, PT)
        assert isinstance(t * t, TT)
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
        assert_composite_types(p * a, [PT, AT])
        assert_composite_types(p * s, [PT, ST])
        assert_composite_types(s * p, [ST, PT])
        assert_composite_types(s * p * a, [ST, PT, AT])
        assert_composite_types(s * a * p, [AT, PT])
        assert_composite_types(p * s * a, [PT, ST, AT])
        assert_composite_types(s * p * s, [ST, PT, ST])
        assert_composite_types(s * a * p * s * a, [AT, PT, ST, AT])
        assert_composite_types(c2 * a, [ST, AT, ST, AT])
        assert_composite_types(p * log_trans * s, [PT, LT, ST])


class CompositeTransform(unittest.TestCase):
    def test_transform_composite(self):
        # Make dummy classes for easier distinguishing the transforms

        class DummyTrans(coorx.BaseTransform):
            pass

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
        assert coorx.CompositeTransform(a).transforms == [a]
        assert coorx.CompositeTransform(a, b).transforms == [a, b]
        assert coorx.CompositeTransform(a, b, c, a).transforms == [a, b, c, a]

        # Test composition by multiplication
        assert_composite_objects(a * b, coorx.CompositeTransform(a, b))
        assert_composite_objects(a * b * c, coorx.CompositeTransform(a, b, c))
        assert_composite_objects(a * b * c * a, coorx.CompositeTransform(a, b, c, a))

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
        t123 = t1*t2*t3
        t321 = t3*t2*t1
        c123 = coorx.CompositeTransform(t1, t2, t3)
        c321 = coorx.CompositeTransform(t3, t2, t1)
        c123s = c123.simplified
        c321s = c321.simplified
        #
        assert isinstance(t123, coorx.STTransform)  # or the test is useless
        assert isinstance(t321, coorx.STTransform)  # or the test is useless
        assert isinstance(c123s, coorx.CompositeTransform)  # or the test is useless
        assert isinstance(c321s, coorx.CompositeTransform)  # or the test is useless

        # Test Mapping
        t1 = coorx.STTransform(scale=(2, 3))
        t2 = coorx.STTransform(offset=(3, 4))
        composite1 = coorx.CompositeTransform(t1, t2)
        composite2 = coorx.CompositeTransform(t2, t1)
        #
        assert composite1.transforms == [t1, t2]  # or the test is useless
        assert composite2.transforms == [t2, t1]  # or the test is useless
        #
        m12 = (t2*t1).map((1, 1)).tolist()
        m21 = (t1*t2).map((1, 1)).tolist()
        m12_ = composite1.map((1, 1)).tolist()
        m21_ = composite2.map((1, 1)).tolist()
        #
        #print(m12, m21, m12_, m21_)
        assert m12 != m21
        assert m12 == m12_
        assert m21 == m21_

        # test pickle
        s = pickle.dumps(composite1)
        assert pickle.loads(s) == composite1

    def test_inverse_composite(self):
        # Test inverse of composite
        t1 = coorx.STTransform(scale=(2, 3))
        t2 = coorx.STTransform(offset=(3, 4))
        composite = coorx.CompositeTransform(t1, t2)
        composite_inv = composite.inverse

        assert composite_inv.map(composite.map((1, 1))).tolist() == [1, 1]

    def test_full_matrix(self):
        t1 = coorx.STTransform(scale=(2, 3, 5))
        t2 = coorx.STTransform(offset=(3, 4, 2))
        composite = coorx.CompositeTransform(t1, t2.inverse)
        assert np.allclose(composite.full_matrix, np.dot(t2.inverse.full_matrix, t1.full_matrix))

    def test_to_vispy(self):
        from vispy.visuals.transforms import ChainTransform

        t1 = coorx.STTransform(scale=(2, 3, 5))
        t2 = coorx.STTransform(offset=(3, 4, 0))
        composite = coorx.CompositeTransform(t1, t2)
        as_vispy = composite.to_vispy()
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
            10**np.random.normal(size=(10, 3)),
            -10**np.random.normal(size=(10, 3)),
            [2,3,4],
            [10],
            (5,6),
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        ]            

    def test_t_transform(self):
        # Check that TTransform maps exactly like AffineTransform
        pts = np.random.normal(size=(10, 3))
        
        translate = (1e6, 0.2, 0)
        tt = coorx.TTransform(offset=translate)
        at = coorx.AffineTransform(dims=(3, 3))
        at.translate(translate)
        
        assert np.allclose(tt.map(pts), at.map(pts))
        assert np.allclose(tt.inverse.map(pts), at.inverse.map(pts))    

        # test save/restore
        tt2 = coorx.TTransform(dims=(3, 3))
        tt2.__setstate__(tt.__getstate__())
        assert np.all(tt.map(pts) == tt2.map(pts))

    def test_itk_compat(self):
        if not HAVE_ITK:
            self.skipTest("itk could not be imported")
        
        itk_tr = itk.TranslationTransform[itk.D, 3].New()
        ttr = TT(dims=(3, 3))
        
        pts = 10**np.random.normal(size=(20, 3), scale=16)
        offsets = 10**np.random.normal(size=(20, 3), scale=16)
        
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
        at.scale(scale)
        at.translate(translate)
        
        assert np.allclose(st.map(pts), at.map(pts))
        assert np.allclose(st.inverse.map(pts), at.inverse.map(pts))    
        

    def test_st_mapping(self):
        p1 = [[5., 7.], [23., 8.]]
        p2 = [[-1.3, -1.4], [1.1, 1.2]]

        t = coorx.STTransform(dims=(2, 2))
        t.set_mapping(p1, p2)

        assert np.allclose(t.map(p1)[:, :len(p2)], p2)


class AffineTransform(unittest.TestCase):
    def test_modifiers(self):
        def check_matrix(t, m):
            assert np.allclose(t.full_matrix, m)
            assert np.allclose(t.matrix, m[:3, :3])
            assert np.allclose(t.offset, m[:3, 3])

        t = coorx.AffineTransform(dims=(3, 3))
        m = np.eye(4)
        check_matrix(t, m)
        
        t.translate(1)
        m[:3, 3] += 1
        check_matrix(t, m)

        t.scale(2)
        m[:3] *= 2
        check_matrix(t, m)

        t.translate(3)
        m[:3, 3] += 3
        check_matrix(t, m)

        t.rotate(90, (0, 0, 1))
        m[:2] = [
            [0, 2, 0, 5], 
            [-2, 0, 0, -5]
        ]
        check_matrix(t, m)

        rm = coorx.AffineTransform(dims=(3, 3))
        rm.rotate(90, (0, 0, 1))
        t2 = rm * coorx.TTransform([3, 3, 3]) * coorx.STTransform(scale=[2, 2, 2]) * coorx.TTransform([1, 1, 1])
        assert t2 == t


    def x_test_affine_mapping(self):
        t = coorx.AffineTransform()
        p1 = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

        # test pure translation
        p2 = p1 + 5.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, :p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test pure scaling
        p2 = p1 * 5.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, :p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.scale(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test scale + translate
        p2 = (p1 * 5.5) + 3.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, :p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.scale(3.5)
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test scale + translate + rotate
        p2 = np.array([[10, 5, 3],
                    [10, 15, 3],
                    [30, 5, 3],
                    [10, 5, 3.5]])
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, :p2.shape[1]], p2)
        t2 = coorx.AffineTransform()
        t2.scale(3.5)
        t2.rotate(90)
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)


class SRT3DTransformTest(unittest.TestCase):
    def test_srt3d(self):
        pts = np.random.normal(size=(10, 3))

        tr = coorx.SRT3DTransform()
        aff = coorx.AffineTransform(dims=(3, 3))
        assert np.allclose(pts, tr.map(pts))

        scale = [10, 1, 0.1]
        tr.set_scale(scale)
        aff.scale(scale)
        assert np.allclose(aff.map(pts), tr.map(pts))

        angle = 30
        axis = np.array([1, 0.5, 0.3])
        tr.set_rotation(angle, axis)
        aff.rotate(angle, axis)
        assert np.allclose(aff.map(pts), tr.map(pts))

        offset = [1e-6, -10, 1e6]
        tr.set_offset(offset)
        aff.translate(offset)
        assert np.allclose(aff.map(pts), tr.map(pts))

        tr2 = coorx.SRT3DTransform(init=aff)
        assert np.allclose(tr.params['offset'], tr2.params['offset'])
        assert np.allclose(tr.params['scale'], tr2.params['scale'])
        assert np.allclose(tr2.map(pts), tr.map(pts))

    def test_save(self):
        tr = coorx.SRT3DTransform(scale=(1, 2, 3), offset=(10, 5, 3), angle=120, axis=(1, 1, 2))
        s = tr.save_state()
        assert s['type'] == 'SRT3DTransform'
        assert s['dims'] == (3, 3)

    def test_to_vispy(self):
        tr = coorx.SRT3DTransform(scale=(1, 2, 3), offset=(10, 5, 3), angle=120, axis=(1, 1, 2))
        vt = tr.to_vispy()
        assert np.allclose(vt.map((1, 1, 1))[:3], tr.map((1, 1, 1)))
        assert np.allclose(vt.map((1, 3, 5))[:3], tr.map((1, 3, 5)))

    def test_composite(self):
        tr1 = coorx.SRT3DTransform(offset=(1, 2, 3))
        tr2 = coorx.SRT3DTransform(scale=(10, 10, 1))

        # test multiplication
        tr3 = tr1 * tr2  # scale, then offset
        assert isinstance(tr3, coorx.AffineTransform)
        assert np.allclose(tr3.map([0, 0, 0]), [1, 2, 3])
        assert np.allclose(tr3.map([1, 1, 1]), [11, 12, 4])

        # test composite
        tr4 = coorx.CompositeTransform(tr2, tr1)
        assert len(tr4.simplified.transforms) == 1
        assert isinstance(tr4.simplified.transforms[0], coorx.AffineTransform)
        assert np.all(tr3.matrix == tr4.simplified.transforms[0].matrix)
        assert np.all(tr3.offset == tr4.simplified.transforms[0].offset)
        assert np.all(tr3.full_matrix == tr4.simplified.transforms[0].full_matrix)
        assert np.allclose(tr4.map([2, -26.7, 0]), tr3.map([2, -26.7, 0]))

        
class TransformInverse(unittest.TestCase):
    def test_inverse(self):
        m = np.random.normal(size=(3, 3))
        transforms = [
            NT(),
            ST(scale=(1e-4, 2e5, 1), offset=(10, -6e9, 0)),
            AT(m),
            RT(m),
        ]

        np.random.seed(0)
        N = 20
        x = np.random.normal(size=(N, 3))
        pw = np.random.normal(size=(N, 3), scale=3)
        pos = x * 10 ** pw

        for trn in transforms:
            assert np.allclose(pos, trn.inverse.map(trn.map(pos))[:, :3])

        # log transform only works on positive values
        #abs_pos = np.abs(pos)
        #tr = LT(base=(2, 4.5, 0))
        #assert np.allclose(abs_pos, tr.inverse.map(tr.map(abs_pos))[:,:3])
