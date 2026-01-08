from .util import (
    AxisSelectionEmbeddedTransform,
    HomogeneousEmbeddedTransform,
)
from .base_transform import Transform, InverseTransform, DependentTransformError
from .composite import CompositeTransform, SimplifiedCompositeTransform
from .linear import (
    NullTransform,
    TTransform,
    STTransform,
    AffineTransform,
    SRT3DTransform,
    TransposeTransform,
    PerspectiveTransform,
    BilinearTransform,
    Homography2DTransform,
)
from .nonlinear import LogTransform, PolarTransform, LensDistortionTransform
from .coordinates import Point, PointArray, Vector, VectorArray
from .image import Image
from .systems import CoordinateSystem, CoordinateSystemGraph

__version__ = "2.0.0"


def transform_types():
    typs = [Transform]
    i = 0
    while i < len(typs):
        typs.extend(typs[i].__subclasses__())
        i += 1
    return typs[1:]


_cached_types = None


def create_transform(type, **kwargs):
    global _cached_types
    if _cached_types is None or type not in _cached_types:
        _cached_types = {tr.__name__: tr for tr in transform_types()}

    if type not in _cached_types:
        raise TypeError(f"Unknown transform type {type!r}")

    return _cached_types[type].from_state(kwargs)
