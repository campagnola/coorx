# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.

from __future__ import division

import numpy as np


def arg_to_array(func):
    """
    Decorator to convert argument to array.

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    func : function
        The decorated function.
    """
    def fn(self, arg, *args, **kwargs):
        """Function

        Parameters
        ----------
        arg : array-like
            Argument to convert.
        *args : tuple
            Arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        value : object
            The return value of the function.
        """
        return func(self, np.array(arg), *args, **kwargs)
    return fn


def as_vec(obj, n, default=0):
    """
    Convert `obj` to N-element vector (numpy array with shape[-1] == N)

    Parameters
    ----------
    obj : array-like
        Original object.
    default : array-like
        The defaults to use if the object does not have N entries.

    Returns
    -------
    obj : array-like
        The object promoted to have N elements.

    Notes
    -----
    `obj` will have at least two dimensions.

    If `obj` has < N elements, then new elements are added from `default`.
    """
    obj = np.atleast_2d(obj)
    # For multiple vectors, reshape to (..., N)
    if obj.shape[-1] < n:
        new = np.empty(obj.shape[:-1] + (n,), dtype=obj.dtype)
        if np.isscalar(default):
            new[..., obj.shape[-1]:] = default
        else:
            new[..., obj.shape[-1]:] = default[obj.shape[-1]:]
        new[..., :obj.shape[-1]] = obj
        obj = new
    elif obj.shape[-1] > n:
        raise TypeError("Array shape %s cannot be converted to vec size %d"
                        % (n, obj.shape))
    return obj


def arg_to_vec(func):
    """
    Decorator for converting argument to vec-N format suitable for NxN matrix
    multiplication. If N=3, for example::

        [x, y]      =>  [[x, y, 0]]

        [x, y, z]   =>  [[x, y, z]]

        [[x1, y1],      [[x1, y1, 0],
        [x2, y2],  =>   [x2, y2, 0],
        [x3, y3]]       [x3, y3, 0]]

    If 1D input is provided, then the return value will be flattened.
    Accepts input of any dimension.

    Alternatively, any class may define its own transform conversion interface
    by defining a _transform_in() method that returns an array with shape
    (.., N), and a _transform_out() method that accepts the same array shape
    and returns a new (mapped) object.

    """
    def arg_to_vec(self_, arg, *args, **kwargs):
        if isinstance(arg, (tuple, list, np.ndarray)):
            arg = np.asarray(arg)
            flatten = arg.ndim == 1
            arg = np.atleast_2d(arg)

            ret = func(self_, arg, *args, **kwargs)
            if flatten and ret is not None:
                return ret.flatten()
            return ret
        elif hasattr(arg, '_transform_in'):
            arr = arg._transform_in()
            ret = func(self_, arr, *args, **kwargs)
            return arg._transform_out(ret)
        else:
            raise TypeError("Cannot convert argument to vector: %s" % arg)
    arg_to_vec.__name__ = func.__name__ + '_arg_to_vec'
    arg_to_vec.__doc__ = func.__doc__
    return arg_to_vec


class TransformCache(object):
    """ Utility class for managing a cache of ChainTransforms.

    This is an LRU cache; items are removed if they are not accessed after
    *max_age* calls to roll().

    Notes
    -----
    This class is used by SceneCanvas to ensure that ChainTransform instances
    are re-used across calls to draw_visual(). SceneCanvas creates one
    TransformCache instance for each top-level visual drawn, and calls
    roll() on each cache before drawing, which removes from the cache any
    transforms that were not accessed during the last draw cycle.
    """
    def __init__(self, max_age=1):
        self._cache = {}  # maps {key: [age, transform]}
        self.max_age = max_age

    def get(self, path):
        """ Get a transform from the cache that maps along *path*, which must
        be a list of Transforms to apply in reverse order (last transform is
        applied first).

        Accessed items have their age reset to 0.
        """
        key = tuple(map(id, path))
        item = self._cache.get(key, None)
        if item is None:
            item = [0, self._create(path)]
            self._cache[key] = item
        item[0] = 0  # reset age for this item

        # make sure the chain is up to date
        #tr = item[1]
        #for i, node in enumerate(path[1:]):
        #    if tr.transforms[i] is not node.transform:
        #        tr[i] = node.transform

        return item[1]

    def _create(self, path):
        # import here to avoid import cycle
        from .chain import ChainTransform
        return ChainTransform(path)

    def roll(self):
        """ Increase the age of all items in the cache by 1. Items whose age
        is greater than self.max_age will be removed from the cache.
        """
        rem = []
        for key, item in self._cache.items():
            if item[0] > self.max_age:
                rem.append(key)
            item[0] += 1

        for key in rem:
            del self._cache[key]
