from ._util import DependentTransformError
from .base_transform import Transform, InverseTransform
from .composite import CompositeTransform, SimplifiedCompositeTransform
from .linear import NullTransform, TTransform, STTransform, AffineTransform, SRT3DTransform, TransposeTransform
from .nonlinear import LogTransform, PolarTransform
from .coordinates import Point, PointArray
from .util import AxisSelectionEmbeddedTransform


__version__ = "1.0.4"


def transform_types():
    typs = [Transform]
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
        raise TypeError(f"Unknown transform type {type!r}")

    return _cached_types[type](dims=dims, from_cs=systems[0], to_cs=systems[1], **params)
