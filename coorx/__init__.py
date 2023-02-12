from .base_transform import BaseTransform, InverseTransform
from .linear import NullTransform, TTransform, STTransform, AffineTransform, SRT3DTransform
from .nonlinear import LogTransform, PolarTransform
from .composite import CompositeTransform, SimplifiedCompositeTransform
from .coordinates import Point, PointArray
from .util import AxisSelectionEmbeddedTransform

__version__ = '1.0.0'


def transform_types():
    typs = [BaseTransform]
    i = 0
    while i < len(typs):
        typs.extend(typs[i].__subclasses__())
        i += 1
    return typs[1:]


_cached_types = None
def create_transform(type, params, dims=None, systems=(None, None)):
    global _cached_types
    if _cached_types is None or type not in _cached_types:
        _cached_types = {tr.__name__: tr for tr in transform_types()}
        
    if type not in _cached_types:
        raise TypeError('Unknown transform type %r' % type)

    return _cached_types[type](dims=dims, from_cs=systems[0], to_cs=systems[1], **params)
