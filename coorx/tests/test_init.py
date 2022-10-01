import unittest
import inspect
import numpy as np
import coorx as tr


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
                    self.list_types(tr.base_transform) +
                    self.list_types(tr.linear) + 
                    self.list_types(tr.nonlinear) +
                    self.list_types(tr.composite)
                ))
        types.remove(tr.BaseTransform)
        
        found_types = tr.transform_types()
        for typ in types:
            assert typ in found_types
            found_types.remove(typ)
        assert len(found_types) == 0

        class TestTransform(tr.BaseTransform):
            pass
        
        assert TestTransform in tr.transform_types()
                    
    def list_types(self, mod):
        return [x for x in mod.__dict__.values() if inspect.isclass(x) and issubclass(x, tr.BaseTransform)]

    def test_create(self):
        for typ in tr.transform_types():
            if typ in [tr.InverseTransform, tr.CompositeTransform, tr.SimplifiedCompositeTransform]:
                continue
            transform1 = typ()
            state = transform1.save_state()
            transform2 = tr.create_transform(**state)
            assert type(transform1) is type(transform2)
            assert_eq(state, transform2.save_state())
