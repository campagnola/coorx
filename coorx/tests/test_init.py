import unittest
import inspect
import numpy as np

import coorx as coorx


def assert_eq(a, b):
    """Test for equivalence of deep data structures
    """
    assert type(a) == type(b)
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert np.allclose(a, b)
    elif isinstance(a, dict):
        assert len(a) == len(b)
        for k,v in a.items():
            assert k in b
            assert_eq(b[k], v)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a, b):
            assert_eq(x, y)
    else:
        assert a == b

    
class InitTests(unittest.TestCase):
    def test_transform_types(self):
        types = list(set(
                    self.list_types(coorx.base_transform) +
                    self.list_types(coorx.linear) + 
                    self.list_types(coorx.nonlinear) +
                    self.list_types(coorx.composite) +
                    self.list_types(coorx.util)
                ))
        types.remove(coorx.Transform)
        
        found_types = coorx.transform_types()
        for typ in types:
            assert typ in found_types
            found_types.remove(typ)
        assert len(found_types) == 0

        class TestTransform(coorx.Transform):
            pass
        
        assert TestTransform in coorx.transform_types()
                    
    def list_types(self, mod):
        return [x for x in mod.__dict__.values() if inspect.isclass(x) and issubclass(x, coorx.Transform)]

    def test_create(self):
        for typ in coorx.transform_types():
            if typ in [coorx.InverseTransform, coorx.CompositeTransform, coorx.SimplifiedCompositeTransform]:
                continue

            kwargs = {
                coorx.nonlinear.LogTransform: {'base': [2, 3, 5], 'dims': (3, 3)},
                coorx.nonlinear.LensDistortionTransform: {'coeff': (0.1, 0.2, 0.3, 0.4, 0.5), 'dims': (2, 2)},
                coorx.util.AxisSelectionEmbeddedTransform: {'dims': (3, 3), 'axes': [0, 1], 'transform': coorx.NullTransform(dims=(2, 2))},
                coorx.util.HomogeneousEmbeddedTransform: {'dims': (3, 3), 'transform': coorx.NullTransform(dims=(4, 4))},
                coorx.linear.BilinearTransform: {'dims': (2, 2)},
                coorx.linear.TransposeTransform: {'axis_order': [1, 0, 2]},
                coorx.linear.Homography2DTransform: {'dims': (2, 2)},
            }
            default_kwargs = {'dims': (3, 3)}
            transform1 = typ(**kwargs.get(typ, default_kwargs))
            state = transform1.save_state()
            transform2 = coorx.create_transform(**state)
            assert type(transform1) is type(transform2)
            assert_eq(state, transform2.save_state())
