# -*- coding: utf-8 -*-
"""
Provides classes representing different transform types suitable for
use with visuals and scenes.

Adapted from vispy.visuals.transforms
Copyright (c) Vispy Development Team. All Rights Reserved.
Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.
"""

from .base_transform import BaseTransform
from .linear import NullTransform, TTransform, STTransform, AffineTransform
from .nonlinear import LogTransform, PolarTransform
from .composite import CompositeTransform
from ._util import arg_to_array, arg_to_vec, as_vec, TransformCache


transform_types = {}
for o in list(globals().values()):
    try:
        if issubclass(o, BaseTransform) and o is not BaseTransform:
            name = o.__name__[:-len('Transform')].lower()
            transform_types[name] = o
    except TypeError:
        continue


def create_transform(type, *args, **kwargs):
    return transform_types[type](*args, **kwargs)
