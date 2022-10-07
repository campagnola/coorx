from ._util import arg_to_array, arg_to_vec, as_vec, TransformCache
from .base_transform import BaseTransform, InverseTransform
from .composite import CompositeTransform, SimplifiedCompositeTransform
from .linear import NullTransform, TTransform, STTransform, AffineTransform
from .nonlinear import LogTransform, PolarTransform

__all__ = [
    "AffineTransform",
    "BaseTransform",
    "CompositeTransform",
    "InverseTransform",
    "LogTransform",
    "NullTransform",
    "PolarTransform",
    "STTransform",
    "SimplifiedCompositeTransform",
    "TTransform",
    "TransformCache",
    "arg_to_array",
    "arg_to_vec",
    "as_vec",
    "create_transform",
    "transform_types",
]


__version__ = "1.0.0"


def transform_types():
    typs = [BaseTransform]
    i = 0
    while i < len(typs):
        typs.extend(typs[i].__subclasses__())
        i += 1
    return typs[1:]


_cached_types = None


def create_transform(typ, params):
    global _cached_types
    if _cached_types is None or typ not in _cached_types:
        _cached_types = {tr.__name__: tr for tr in transform_types()}

    if typ not in _cached_types:
        raise TypeError("Unknown transform type %r" % typ)

    return _cached_types[typ](**params)
