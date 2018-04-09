#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Library of matrix transformation functions adapted from vispy.
"""

from __future__ import division

# Note: we use functions (e.g. sin) from math module because they're faster

import math
import numpy as np


def translate(offset, dtype=None):
    """Translate by an offset (x, y, z) .

    Parameters
    ----------
    offset : array-like, shape (3,)
        Translation in x, y, z.
    dtype : dtype | None
        Output type (if None, don't cast).

    Returns
    -------
    M : ndarray
        Transformation matrix describing the translation.
    """
    assert len(offset) == 3
    x, y, z = offset
    M = np.array([[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [x, y, z, 1.0]], dtype)
    return M


def scale(s, dtype=None):
    """Non-uniform scaling along the x, y, and z axes

    Parameters
    ----------
    s : array-like, shape (3,)
        Scaling in x, y, z.
    dtype : dtype | None
        Output type (if None, don't cast).

    Returns
    -------
    M : ndarray
        Transformation matrix describing the scaling.
    """
    assert len(s) == 3
    return np.array(np.diag(np.concatenate([s, (1.,)])), dtype)


def rotate(angle, axis, dtype=None):
    """The 3x3 rotation matrix for rotation about a vector.

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.
    """
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, .0],
                  [cx * y + z * s, cy * y + c, cz * y - x * s, 0.],
                  [cx * z - y * s, cy * z + x * s, cz * z + c, 0.],
                  [0., 0., 0., 1.]], dtype).T
    return M


def affine_map(points1, points2):
    """ Find a 3D transformation matrix that maps points1 onto points2.

    Arguments are specified as arrays of four 3D coordinates, shape (4, 3).
    """
    A = np.ones((4, 4))
    A[:, :3] = points1
    B = np.ones((4, 4))
    B[:, :3] = points2

    # solve 3 sets of linear equations to determine
    # transformation matrix elements
    matrix = np.eye(4)
    for i in range(3):
        # solve Ax = B; x is one row of the desired transformation matrix
        matrix[i] = np.linalg.solve(A, B[:, i])

    return matrix
