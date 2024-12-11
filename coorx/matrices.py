"""
Library of matrix transformation functions adapted from vispy.
"""

from __future__ import division

# Note: we use functions (e.g. sin) from math module because they're faster

import math
import numpy as np
import numpy.linalg


def translate(offset, dtype=None):
    """Return an N-dimensional translation matrix (of shape N+1,N+1)

    Parameters
    ----------
    offset : array-like
        Amount to translate each axis
    dtype : dtype | None
        Output type (if None, then same as *offset*).

    Returns
    -------
    M : ndarray
        Transformation matrix describing the translation.
    """
    if dtype is None:
        dtype = offset.dtype
    M = np.eye(len(offset) + 1, dtype=dtype)
    M[-1, :len(offset)] = offset
    return M


def scale(s, dtype=None):
    """Return an N-dimensional scaling matrix (of shape N+1,N+1)

    Parameters
    ----------
    scale : array-like, shape (3,)
        Scale factors for each axis
    dtype : dtype | None
        Output type (if None, then same as *scale*).

    Returns
    -------
    M : ndarray
        Transformation matrix describing the scaling.
    """
    if dtype is None:
        dtype = scale.dtype
    return np.array(np.diag(np.concatenate([s, (1.,)])), dtype)


def rotate2d(angle, dtype=None):
    """Return a 2D rotation matrix (shape 2,2)

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    dtype : dtype | None
        Output dtype
    """
    rad = np.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    M = np.array([
        [c,  s],
        [-s, c],
    ], dtype=dtype)
    return M


def rotate3d(angle, axis, dtype=None):
    """Return a 3D rotation matrix (shape 3,3) for rotation about a vector.

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.
    dtype : dtype | None
        Output dtype
    """
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array([
        [cx * x + c, cy * x - z * s, cz * x + y * s],
        [cx * y + z * s, cy * y + c, cz * y - x * s],
        [cx * z - y * s, cy * z + x * s, cz * z + c],
    ], dtype).T
    return M


def affine_map(points1, points2):
    """ Find an N-D affine transformation that maps points1 onto points2.

    Arguments are specified as arrays of coordinates, shape (N+1, N).
    """
    N = points1.shape[1]
    N1 = N + 1
    if points2.shape != (N1, N) or points1.shape != (N1, N):
        raise TypeError("Points must have shape (N+1, N)")
    A = np.ones((N1, N1))
    A[:, :N] = points1
    B = np.ones((N1, N1))
    B[:, :N] = points2

    # solve N sets of linear equations to determine
    # transformation matrix elements
    matrix = np.eye(N1)
    for i in range(N):
        # solve Ax = B; x is one row of the desired transformation matrix
        matrix[i] = np.linalg.solve(A, B[:, i])

    return matrix[:N]


def bilinear_map2d(points1, points2):
    """
    Find a bilinear transformation matrix (shape 2x4) that maps 2D points1 onto 2D points2.

    Parameters
    ----------
    points1 :  array-like (N, 2)
        Array of N+1 2D points
    points2 :  array-like (N, 2)
        Array of N+1 2D points
    
    To use this matrix to map a point [x,y]::
    
        mapped = np.dot(matrix, [x*y, x, y, 1])
    """
    ## A is 4 rows (points) x 4 columns (xy, x, y, 1)
    ## B is 4 rows (points) x 2 columns (x, y)
    A = np.array([[points1[i,0]*points1[i,1], points1[i,0], points1[i,1], 1] for i in range(4)])
    B = np.array([[points2[i,0], points2[i,1]] for i in range(4)])
    
    ## solve 2 sets of linear equations to determine transformation matrix elements
    matrix = np.zeros((2,4))
    for i in range(2):
        matrix[i] = numpy.linalg.solve(A, B[:,i])  ## solve Ax = B; x is one row of the desired transformation matrix
    
    return matrix


def frustum(left, right, bottom, top, znear, zfar):
    """Create view frustum

    Parameters
    ----------
    left : float
        Left coordinate of the field of view.
    right : float
        Right coordinate of the field of view.
    bottom : float
        Bottom coordinate of the field of view.
    top : float
        Top coordinate of the field of view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.

    Returns
    -------
    M : ndarray
        View frustum matrix (4x4).
    """
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)

    M = np.zeros((4, 4))
    M[0, 0] = +2.0 * znear / float(right - left)
    M[2, 0] = (right + left) / float(right - left)
    M[1, 1] = +2.0 * znear / float(top - bottom)
    M[2, 1] = (top + bottom) / float(top - bottom)
    M[2, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / float(zfar - znear)
    M[2, 3] = -1.0
    return M
