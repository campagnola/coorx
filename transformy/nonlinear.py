# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import division

import numpy as np

from ._util import arg_to_array, arg_to_vec, as_vec
from .base_transform import BaseTransform


class LogTransform(BaseTransform):
    """ Transform perfoming logarithmic transformation on three axes.

    Maps (x, y, z) => (log(base.x, x), log(base.y, y), log(base.z, z))

    No transformation is applied for axes with base == 0.

    If base < 0, then the inverse function is applied: x => base.x ** x

    Parameters
    ----------
    base : array-like
        Base for the X, Y, Z axes.
    """
    Linear = False
    Orthogonal = True
    NonScaling = False
    Isometric = False

    def __init__(self, base=None, dims=None):
        if base is not None:
            if dims is not None:
                raise TypeError("Cannot specify both base and dims")
            base = np.asarray(base)
            if base.ndim != 1:
                raise TypeError("Base must be 1-D array-like")
            dims = (len(base), len(base))
        if dims is None:
            dims = (3, 3)
        
        super(LogTransform, self).__init__(dims)
        
        self._base = np.zeros(self.dims[0], dtype=np.float32)
        if base is not None:
            self.base = base

    @property
    def base(self):
        """
        *base* is a tuple (x, y, z) containing the log base that should be
        applied to each axis of the input vector. If any axis has a base == 0,
        then that axis is not affected.
        """
        return self._base.copy()

    @base.setter
    def base(self, s):
        self._base[:] = s

    @arg_to_array
    def map(self, coords, base=None):
        ret = np.empty(coords.shape, coords.dtype)
        if base is None:
            base = self.base
        for i in range(min(ret.shape[-1], 3)):
            if base[i] > 1.0:
                ret[..., i] = np.log(coords[..., i]) / np.log(base[i])
            elif base[i] < -1.0:
                ret[..., i] = -base[i] ** coords[..., i]
            else:
                ret[..., i] = coords[..., i]
        return ret

    @arg_to_array
    def imap(self, coords):
        return self.map(coords, -self.base)

    def __repr__(self):
        return "<LogTransform base=%s>" % (self.base)


class PolarTransform(BaseTransform):
    """Polar transform

    Maps (theta, r, z) to (x, y, z), where `x = r*cos(theta)`
    and `y = r*sin(theta)`.
    """
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=None):
        if dims is None:
            dims = (3, 3)
        super(PolarTransform, self).__init__(dims)

    @arg_to_array
    def map(self, coords):
        ret = np.empty(coords.shape, coords.dtype)
        ret[..., 0] = coords[..., 1] * np.cos(coords[..., 0])
        ret[..., 1] = coords[..., 1] * np.sin(coords[..., 0])
        for i in range(2, coords.shape[-1]):  # copy any further axes
            ret[..., i] = coords[..., i]
        return ret

    @arg_to_array
    def imap(self, coords):
        ret = np.empty(coords.shape, coords.dtype)
        ret[..., 0] = np.arctan2(coords[..., 0], coords[..., 1])
        ret[..., 1] = (coords[..., 0]**2 + coords[..., 1]**2) ** 0.5
        for i in range(2, coords.shape[-1]):  # copy any further axes
            ret[..., i] = coords[..., i]
        return ret


#class BilinearTransform(BaseTransform):
#    # TODO
#    pass


#class WarpTransform(BaseTransform):
#    """ Multiple bilinear transforms in a grid arrangement.
#    """
#    # TODO

