# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.

from __future__ import division

import numpy as np

from ._util import arg_to_vec, as_vec
from .base_transform import BaseTransform
from . import matrices


class NullTransform(BaseTransform):
    """ Transform having no effect on coordinates (identity transform).
    
    The default dimensionality is (3, 3), but this is ignored when mapping;
    all argments are returned unmodified.
    """
    Linear = True
    Orthogonal = True
    NonScaling = True
    Isometric = True

    def __init__(self, dims=3):
        BaseTransform.__init__(self, dims)

    def map(self, coords):
        """Return the input array unmodified.

        Parameters
        ----------
        coords : array-like
            Coordinates to map.
        """
        return coords

    def imap(self, coords):
        """Return the input array unmodified.

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map.
        """
        return coords

    def __mul__(self, tr):
        return tr

    def __rmul__(self, tr):
        return tr

    @property
    def params(self):
        return {}
    
    def set_params(self):
        return


class TTransform(BaseTransform):
    """ Transform performing only 3D translation.

    Parameters
    ----------
    offset : array-like
        Translation distances for X, Y, Z axes.
    dims : tuple
        (input, output) dimensions for transform.
    """
    Linear = True
    Orthogonal = True
    NonScaling = False
    Isometric = False

    def __init__(self, offset=None, dims=None):

        if offset is not None:
            offset = np.asarray(offset)
            if offset.ndim != 1:
                raise TypeError("offset must be 1-D array or similar")
            if dims is not None:
                raise TypeError("Cannot specify both offset and dims")
            d = len(offset)
            dims = (d, d)
        if dims is None:
            dims = (3, 3)
            
        try:
            dims = tuple(dims)
            assert len(dims) == 2
        except (TypeError, AssertionError):
            raise TypeError("dims must be length-2 tuple")

        super(TTransform, self).__init__(dims)
        
        if self.dims[0] != self.dims[1]:
            raise ValueError("Input and output dimensionality must be equal")
            
        self._offset = np.zeros(self.dims[0], dtype=np.float)
        if offset is not None:
            self.offset = offset

    @arg_to_vec
    def map(self, coords):
        """Return translated coordinates.

        Parameters
        ----------
        coords : array-like
            Coordinates to map. The length of the last array dimenstion
            must be equal to the input dimensionality of the transform.

        Returns
        -------
        coords : ndarray
            Mapped coordinates: coords + translation
        """
        if coords.shape[-1] != self.dims[0]:
            raise TypeError("Shape of last axis (%d) is not equal to input dimension of transform (%d)" % (coords.shape[-1], self.dims[0]))
        return coords + self.offset[np.newaxis, :]

    @arg_to_vec
    def imap(self, coords):
        """Return inverse-mapped coordinates.

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map. The length of the last array dimenstion
            must be equal to the input dimensionality of the transform.

        Returns
        -------
        coords : ndarray
            Mapped coordinates: coords - translation
        """
        if coords.shape[-1] != self.dims[1]:
            raise TypeError("Shape of last axis (%d) is not equal to output dimension of transform (%d)" % (coords.shape[-1], self.dims[1]))
        return coords - self.offset[np.newaxis, :]

    @property
    def offset(self):
        return self._offset.copy()

    @offset.setter
    def offset(self, t):
        t = np.asarray(t)
        if t.shape != (self.dims[0],):
            raise TypeError("Offset must have length equal to transform dimensionality (%d)" % self.dims[0])
        if np.all(t == self._offset):
            return
        
        self._offset[:] = t
        self.update()   # inform listeners there has been a change

    def translate(self, offset):
        """Change the translation of this transform by the amount given.

        Parameters
        ----------
        offset : array-like
            The values to be added to the current translation of the transform.
        """
        offset = np.asarray(offset)
        self.offset = self.offset + offset

    def as_affine(self):
        m = AffineTransform()
        m.translate(self.offset)
        return m

    def as_st(self):
        return STTransform(offset=self.offset, scale=(1,) * self.dims[0])

    def __mul__(self, tr):
        if isinstance(tr, TTransform):
            return TTransform(self.offset + tr.offset)
        elif isinstance(tr, STTransform):
            return self.as_st() * tr
        elif isinstance(tr, AffineTransform):
            return self.as_affine() * tr
        else:
            return super(STTransform, self).__mul__(tr)

    def __rmul__(self, tr):
        if isinstance(tr, STTransform):
            return tr * self.as_st()
        if isinstance(tr, AffineTransform):
            return tr * self.as_affine()
        return super(TTransform, self).__rmul__(tr)

    def __repr__(self):
        return ("<TTransform offset=%s at 0x%s>"
                % (self.offset, id(self)))

    @property
    def params(self):
        return {'offset': self.offset}
    
    def set_params(self, offset=None):
        if offset is not None:
            self.offset = offset


class STTransform(BaseTransform):
    """ Transform performing only scale and translate, in that order.

    Parameters
    ----------
    scale : array-like
        Scale factors for X, Y, Z axes.
    offset : array-like
        Translation distances for X, Y, Z axes.
    """
    Linear = True
    Orthogonal = True
    NonScaling = False
    Isometric = False

    def __init__(self, scale=None, offset=None, dims=None):

        if scale is not None or offset is not None:
            if dims is not None:
                raise TypeError("Cannot specify both dims and scale/offset")
        if scale is not None:
            dims = len(scale), len(scale)
        elif offset is not None:
            dims = len(offset), len(offset)

        if dims is None:
            dims = (3, 3)
            
        super(STTransform, self).__init__(dims)
        
        if self.dims[0] != self.dims[1]:
            raise ValueError("Input and output dimensionality must be equal")
            
        self._scale = np.ones(self.dims[0], dtype=np.float)
        self._offset = np.zeros(self.dims[0], dtype=np.float)

        self.set_params(scale, offset)

    @arg_to_vec
    def map(self, coords):
        """Return coordinates mapped by scale and translation.

        Parameters
        ----------
        coords : array-like
            Coordinates to map. The length of the last array dimenstion
            must be equal to the input dimensionality of the transform.

        Returns
        -------
        coords : ndarray
            Mapped coordinates: coords * scale + offset
        """
        if coords.shape[-1] != self.dims[0]:
            raise TypeError("Shape of last axis (%d) is not equal to input dimension of transform (%d)" % (coords.shape[-1], self.dims[0]))
        return coords * self.scale[None, :] + self.offset[None, :]

    @arg_to_vec
    def imap(self, coords):
        """Return coordinates inverse-mapped by translation and scale.

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map. The length of the last array dimenstion
            must be equal to the input dimensionality of the transform.

        Returns
        -------
        coords : ndarray
            Mapped coordinates: (coords - offset) / scale
        """
        if coords.shape[-1] != self.dims[1]:
            raise TypeError("Shape of last axis (%d) is not equal to output dimension of transform (%d)" % (coords.shape[-1], self.dims[1]))
        return (coords - self.offset[None, :]) / self.scale[None, :]

    @property
    def scale(self):
        return self._scale.copy()

    @scale.setter
    def scale(self, s):
        self.set_params(scale=s)

    @property
    def offset(self):
        return self._offset.copy()

    @offset.setter
    def offset(self, t):
        self.set_params(offset=t)

    @property
    def params(self):
        return {'scale': self.scale, 'offset': self.offset}

    def set_params(self, scale=None, offset=None):
        need_update = False

        if scale is not None:
            scale = np.asarray(scale)
            if scale.shape != (self.dims[0],):
                raise TypeError("Scale must have length equal to transform dimensionality (%d)" % self.dims[0])
            if not np.all(scale == self._scale):
                self._scale[:] = scale
                need_update = True

        if offset is not None and not np.all(offset == self._offset):
            offset = np.asarray(offset)
            if offset.shape != (self.dims[0],):
                raise TypeError("Offset must have length equal to transform dimensionality (%d)" % self.dims[0])
            if not np.all(offset == self._offset):
                self._offset[:] = offset
                need_update = True

        if need_update:
            self.update()   # inform listeners there has been a change

    def translate(self, offset):
        """Change the translation of this transform by the amount given.

        Parameters
        ----------
        offset : array-like
            The values to be added to the current translation of the transform.
        """
        offset = np.asarray(offset)
        self.offset = self.offset + offset

    def zoom(self, zoom, center=(0, 0, 0), mapped=True):
        """Update the transform such that its scale factor is changed, but
        the specified center point is left unchanged.

        Parameters
        ----------
        zoom : array-like
            Values to multiply the transform's current scale
            factors.
        center : array-like
            The center point around which the scaling will take place.
        mapped : bool
            Whether *center* is expressed in mapped coordinates (True) or
            unmapped coordinates (False).
        """
        zoom = as_vec(zoom, 3, default=1)
        center = as_vec(center, 3, default=0)
        scale = self.scale * zoom
        if mapped:
            trans = center - (center - self.offset) * zoom
        else:
            trans = self.scale * (1 - zoom) * center + self.offset
        self.set_params(scale=scale, offset=trans)

    def as_affine(self):
        m = AffineTransform()
        m.scale(self.scale)
        m.translate(self.offset)
        return m

    @classmethod
    def from_mapping(cls, x0, x1):
        """ Create an STTransform from the given mapping

        See `set_mapping` for details.

        Parameters
        ----------
        x0 : array-like
            Start.
        x1 : array-like
            End.

        Returns
        -------
        t : instance of STTransform
            The transform.
        """
        t = cls()
        t.set_mapping(x0, x1)
        return t

    def set_mapping(self, x0, x1):
        """Configure this transform such that it maps points x0 onto x1

        Parameters
        ----------
        x0 : array-like, shape (2, N)
            Two source points
        x1 : array-like, shape (2, N)
            Two destination points

        Examples
        --------
        For example, if we wish to map the corners of a rectangle::

            >>> p1 = [[0, 0], [200, 300]]

        onto a unit cube::

            >>> p2 = [[-1, -1], [1, 1]]

        then we can generate the transform as follows::

            >>> tr = STTransform()
            >>> tr.set_mapping(p1, p2)
            >>> assert tr.map(p1)[:,:2] == p2  # test

        """
        x0 = np.asarray(x0)
        x1 = np.asarray(x1)
        if (x0.ndim != 2 or x0.shape[0] != 2 or x1.ndim != 2 or 
                x1.shape[0] != 2):
            raise TypeError("set_mapping requires array inputs of shape "
                            "(2, N).")
        denom = x0[1] - x0[0]
        mask = denom == 0
        denom[mask] = 1.0
        s = (x1[1] - x1[0]) / denom
        s[mask] = 1.0
        t = x1[0] - s * x0[0]
        self.set_params(scale=s, offset=t)

    def __mul__(self, tr):
        if isinstance(tr, STTransform):
            s = self.scale * tr.scale
            t = self.offset + (tr.offset * self.scale)
            return STTransform(scale=s, offset=t)
        elif isinstance(tr, AffineTransform):
            return self.as_affine() * tr
        else:
            return super(STTransform, self).__mul__(tr)

    def __rmul__(self, tr):
        if isinstance(tr, AffineTransform):
            return tr * self.as_affine()
        return super(STTransform, self).__rmul__(tr)

    def __repr__(self):
        return ("<STTransform scale=%s offset=%s at 0x%s>"
                % (self.scale, self.offset, id(self)))


class AffineTransform(BaseTransform):
    """Affine transformation class

    Parameters
    ----------
    matrix : array-like | None
        Array to use for the transform. If None, then an identity transform is
        assumed. The shape of the matrix determines the (output, input)
        dimensions of the transform.
    offset : array-like | None
        The translation to apply in this affine transform.
    dims : tuple
        Optionally specifies the (input, output) dimensions of this transform.
        
    """
    Linear = True
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, matrix=None, offset=None, dims=None):
        if matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise TypeError("Matrix must be 2-dimensional and square; got %r" % matrix.shape)
            if dims is not None:
                raise TypeError("Cannot specify both matrix and dims")
            dims = matrix.shape[::-1]
        if dims is None:
            dims = (3, 3)
        
        super(AffineTransform, self).__init__(dims)
        
        self.reset()
        if matrix is not None:
            self.matrix = matrix
        if offset is not None:
            self.offset = offset

    @arg_to_vec
    def map(self, coords):
        """Map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to map. The length of the last array dimenstion
            must be equal to the input dimensionality of the transform.

        Returns
        -------
        coords : ndarray
            Mapped coordinates: (M * coords) + offset
        """
        if coords.shape[-1] != self.dims[0]:
            raise TypeError("Shape of last axis (%d) is not equal to input dimension of transform (%d)" % (coords.shape[-1], self.dims[0]))
        return np.dot(self.matrix, coords.T).T + self.offset[None, :]

    @arg_to_vec
    def imap(self, coords):
        """Inverse map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map. The length of the last array dimenstion
            must be equal to the input dimensionality of the transform.

        Returns
        -------
        coords : ndarray
            Mapped coordinates: M_inv * (coords - offset)
        """
        if coords.shape[-1] != self.dims[1]:
            raise TypeError("Shape of last axis (%d) is not equal to output dimension of transform (%d)" % (coords.shape[-1], self.dims[1]))
        return np.dot(self.inv_matrix, (coords + self.inv_offset[None, :]).T).T

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, m):
        self.set_params(matrix=m)

    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self, o):
        self.set_params(offset=o)

    @property
    def params(self):
        return {'matrix': self.matrix, 'offset': self.offset}

    def set_params(self, matrix=None, offset=None):
        need_update = False
        
        if matrix is not None:
            m = np.asarray(matrix)
            if m.shape[::-1] != self.dims:
                raise TypeError("Matrix shape must be %s" % (self.dims[::-1],))
            if np.any(m != self._matrix):
                self._matrix = m
                self._inv_matrix = None
                need_update = True

        if offset is not None:
            o = np.asarray(offset)
            if o.ndim != 1 or len(o) != self.dims[1]:
                raise Exception("Offset length must be the same as transform output dimension (%d)" % self.dims[1])
            if np.any(o != self._offset):
                self._offset = o
                self._inv_matrix = None
                need_update = True

        if need_update:
            self.update()

    @property
    def inv_matrix(self):
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self.matrix)
        return self._inv_matrix

    @property
    def inv_offset(self):
        return -self.offset

    @property
    def full_matrix(self):
        """Return a matrix of shape (N+1, N+1) that contains both self.matrix
        and self.offset::

            [[m11, m21, m31, o1],
             [m21, m22, m32, o2],
             [m31, m32, m33, o3],
             [  0,   0,   0,  1]]

        The full matrix can be multiplied by other similar matrices in order to compose affine 
        transforms together.
        """
        m = np.zeros((self.dims[1]+1, self.dims[0]+1))
        m[:-1, :-1] = self.matrix
        m[:-1, -1] = self.offset
        m[-1, -1] = 1
        return m

    def translate(self, pos):
        """
        Add to the offset.

        The translation is applied *after* the transformations already present.

        Parameters
        ----------
        pos : arrayndarray
            Position to translate by.
        """
        pos = np.asarray(pos)
        self.offset = self.offset + pos

    def scale(self, scale, center=None):
        """
        Scale the matrix about a given origin.

        The scaling is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        scale : array-like
            Scale factors along x, y and z axes.
        center : array-like or None
            The x, y and z coordinates to scale around. If None,
            (0, 0, 0) will be used.
        """
        scale_matrix = np.zeros(self.dims[::-1])
        for i in range(min(self.dims)):
            scale_matrix[i,i] = scale[i]

        if center is not None:
            raise NotImplementedError()
        self.matrix = np.dot(scale_matrix, self.matrix)

    def rotate(self, angle, axis):
        """
        Rotate the matrix by some angle about a given axis.

        The rotation is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        angle : float
            The angle of rotation in degrees.
        axis : array-like or None
            The x, y and z coordinates of the axis vector to rotate around.
        """
        self.matrix = np.dot(self.matrix, matrices.rotate(angle, axis))

    @classmethod
    def from_mapping(cls, x0, x1):
        """ Create an AffineTransform from the given mapping

        See `set_mapping` for details.

        Parameters
        ----------
        x0 : array-like
            Unmapped points.
        x1 : array-like
            Mapped points.

        Returns
        -------
        t : instance of STTransform
            The transform.
        """
        t = cls()
        t.set_mapping(x0, x1)
        return t

    def set_mapping(self, points1, points2):
        """Set this transform such that it maps points1 onto points2.

        Parameters
        ----------
        points1 : array-like, shape (4, N)
            Four starting (unmapped) coordinates.
        points2 : array-like, shape (4, N)
            Four ending (mapped) coordinates.
        """
        m = matrices.affine_map(points1, points2)
        self.set_params(matrix=m[:,:-1], offset=m[:, -1])

    def reset(self):
        """Reset this transform to have an identity matrix and no offset.
        """
        self._matrix = np.eye(max(self.dims))[:self.dims[1], :self.dims[0]]
        self._offset = np.zeros(self.dims[1])
        self.update()

    def __mul__(self, tr):
        if isinstance(tr, AffineTransform):
            m = np.dot(self.full_matrix, tr.full_matrix)
            return AffineTransform(matrix=m[:-1, :-1], offset=m[:-1, -1])
        else:
            return tr.__rmul__(self)

    def __repr__(self):
        s = "%s(offset=%r matrix=[" % (self.__class__.__name__, list(self.offset))
        indent = " "*len(s)
        for i in range(self.dims[0]):
            if i > 0:
                s += indent
            s += str(list(self.matrix[i]))
            if i == self.dims[0]-1:
                s += "]"
            else:
                s += ",\n"
        s +=  "at 0x%x)" % id(self)
        return s
