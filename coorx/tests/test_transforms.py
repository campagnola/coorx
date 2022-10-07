# -*- coding: utf-8 -*-
# Adapted from vispy
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.
import math
import unittest

import numpy as np
from coorx import LogTransform

try:
    import itk
    HAVE_ITK = True
except ImportError:
    HAVE_ITK = False

import coorx as tr

NT = tr.NullTransform
TT = tr.TTransform
ST = tr.STTransform
AT = tr.AffineTransform
RT = tr.AffineTransform
PT = tr.PolarTransform
LT = tr.LogTransform
CT = tr.CompositeTransform


def assert_composite_types(composite, types):
    assert list(map(type, composite.transforms)) == types


def assert_composite_objects(composite1, composite2):
    assert composite1.transforms == composite2.transforms


class TransformMultiplication(unittest.TestCase):
    def test_multiplication(self):
        n = NT()
        t = TT()
        s = ST()
        a = AT()
        p = PT()
        log_trans = LT()
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

        class DummyTrans(tr.BaseTransform):
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
        assert tr.CompositeTransform().transforms == []
        assert tr.CompositeTransform(a).transforms == [a]
        assert tr.CompositeTransform(a, b).transforms == [a, b]
        assert tr.CompositeTransform(a, b, c, a).transforms == [a, b, c, a]

        # Test composition by multiplication
        assert_composite_objects(a * b, tr.CompositeTransform(a, b))
        assert_composite_objects(a * b * c, tr.CompositeTransform(a, b, c))
        assert_composite_objects(a * b * c * a, tr.CompositeTransform(a, b, c, a))

        # Test adding/prepending to transform
        composite = tr.CompositeTransform()
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
        t1 = tr.STTransform(scale=(2, 3))
        t2 = tr.STTransform(offset=(3, 4))
        t3 = tr.STTransform(offset=(3, 4))
        # Create multiplied versions
        t123 = t1*t2*t3
        t321 = t3*t2*t1
        c123 = tr.CompositeTransform(t1, t2, t3)
        c321 = tr.CompositeTransform(t3, t2, t1)
        c123s = c123.simplified
        c321s = c321.simplified
        #
        assert isinstance(t123, tr.STTransform)  # or the test is useless
        assert isinstance(t321, tr.STTransform)  # or the test is useless
        assert isinstance(c123s, tr.CompositeTransform)  # or the test is useless
        assert isinstance(c321s, tr.CompositeTransform)  # or the test is useless

        # Test Mapping
        t1 = tr.STTransform(scale=(2, 3))
        t2 = tr.STTransform(offset=(3, 4))
        composite1 = tr.CompositeTransform(t1, t2)
        composite2 = tr.CompositeTransform(t2, t1)
        #
        assert composite1.transforms == [t1, t2]  # or the test is useless
        assert composite2.transforms == [t2, t1]  # or the test is useless
        #
        m12 = (t1*t2).map((1, 1)).tolist()
        m21 = (t2*t1).map((1, 1)).tolist()
        m12_ = composite1.map((1, 1)).tolist()
        m21_ = composite2.map((1, 1)).tolist()
        #
        #print(m12, m21, m12_, m21_)
        assert m12 != m21
        assert m12 == m12_
        assert m21 == m21_


class TTransform(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.transforms = [
            TT(),
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
        tt = tr.TTransform(offset=translate)
        at = tr.AffineTransform()
        at.translate(translate)
        
        assert np.allclose(tt.map(pts), at.map(pts))
        assert np.allclose(tt.inverse.map(pts), at.inverse.map(pts))    

        # test save/restore
        tt2 = tr.TTransform()
        tt2.__setstate__(tt.__getstate__())
        assert np.all(tt.map(pts) == tt2.map(pts))

    def test_itk_compat(self):
        if not HAVE_ITK:
            self.skipTest("itk could not be imported")
        
        itk_tr = itk.TranslationTransform[itk.D, 3].New()
        ttr = TT()
        
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
        st = tr.STTransform(scale=scale, offset=translate)
        at = tr.AffineTransform()
        at.scale(scale)
        at.translate(translate)
        
        assert np.allclose(st.map(pts), at.map(pts))
        assert np.allclose(st.inverse.map(pts), at.inverse.map(pts))    
        

    def test_st_mapping(self):
        p1 = [[5., 7.], [23., 8.]]
        p2 = [[-1.3, -1.4], [1.1, 1.2]]

        t = tr.STTransform(dims=(2, 2))
        t.set_mapping(p1, p2)

        assert np.allclose(t.map(p1)[:, :len(p2)], p2)


class AffineTransform(unittest.TestCase):
    def test_modifiers(self):
        def check_matrix(t, m):
            assert np.allclose(t.full_matrix, m)
            assert np.allclose(t.matrix, m[:3, :3])
            assert np.allclose(t.offset, m[:3, 3])

        t = tr.AffineTransform(dims=(3, 3))
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

        rm = tr.AffineTransform(dims=(3, 3))
        rm.rotate(90, (0, 0, 1))
        t2 = rm * tr.TTransform([3, 3, 3]) * tr.STTransform(scale=[2, 2, 2]) * tr.TTransform([1, 1, 1])
        assert t2 == t


    def x_test_affine_mapping(self):
        t = tr.AffineTransform()
        p1 = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

        # test pure translation
        p2 = p1 + 5.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, :p2.shape[1]], p2)
        t2 = tr.AffineTransform()
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test pure scaling
        p2 = p1 * 5.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, :p2.shape[1]], p2)
        t2 = tr.AffineTransform()
        t2.scale(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)

        # test scale + translate
        p2 = (p1 * 5.5) + 3.5
        t.set_mapping(p1, p2)
        assert np.allclose(t.map(p1)[:, :p2.shape[1]], p2)
        t2 = tr.AffineTransform()
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
        t2 = tr.AffineTransform()
        t2.scale(3.5)
        t2.rotate(90)
        t2.translate(5.5)
        assert np.allclose(t.full_matrix, t2.full_matrix)


class LogTransformTest(unittest.TestCase):
    def test_log(self):
        lt = LogTransform((12, 0))
        data = [(12, -6), (144, 13.2), (float("inf"), 21), (-7, 44), (0, 0)]
        output = lt.map(data)
        self.assertAlmostEqual(output[0][0], 1)
        self.assertAlmostEqual(output[1][0], 2)
        self.assertAlmostEqual(output[1][1], 13.2)
        self.assertTrue(math.isnan(output[2][0]))
        self.assertTrue(math.isnan(output[3][0]))
        self.assertTrue(math.isnan(output[4][0]))
        self.assertAlmostEqual(output[4][1], 0)


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
