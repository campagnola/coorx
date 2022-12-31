# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.

from __future__ import division

import numpy as np




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
