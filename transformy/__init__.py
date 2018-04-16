# -*- coding: utf-8 -*-
"""
Provides classes representing different transform types suitable for
use with visuals and scenes.

Adapted from vispy.visuals.transforms
Copyright (c) Vispy Development Team. All Rights Reserved.
Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.
"""

from .base_transform import BaseTransform, InverseTransform
from .linear import NullTransform, TTransform, STTransform, AffineTransform
from .nonlinear import LogTransform, PolarTransform
from .composite import CompositeTransform, SimplifiedCompositeTransform
from ._util import arg_to_array, arg_to_vec, as_vec, TransformCache


def transform_types():
    typs = [BaseTransform]
    i = 0
    while i < len(typs):
        typs.extend(typs[i].__subclasses__())
        i += 1
    return typs[1:]


_cached_types = None
def create_transform(type, params):
    global _cached_types
    if _cached_types is None or type not in _cached_types:
        _cached_types = {tr.__name__: tr for tr in transform_types()}
        
    if type not in _cached_types:
        raise TypeError('Unknown transform type %r' % type)

    return _cached_types[type](**params)
