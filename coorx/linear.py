import numpy as np
import numpy.linalg
import scipy.optimize

from . import matrices
from ._types import Dims
from .base_transform import Transform
from .params import ArrayParameter, TupleParameter, FloatParameter, TransformParameter


class NullTransform(Transform):
    """Transform having no effect on coordinates (identity transform)."""

    Linear = True
    Orthogonal = True
    NonScaling = True
    Isometric = True
    Equidimensional = True

    def _map(self, coords):
        """Return the input array unmodified.

        Parameters
        ----------
        coords : array-like
            Coordinates to map.
        """
        return coords

    def _imap(self, coords):
        """Return the input array unmodified.

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map.
        """
        return coords

    def as_affine(self):
        return AffineTransform(
            matrix=np.eye(self.dims[0]),
            offset=np.zeros(self.dims[0]),
            from_cs=self.systems[0],
            to_cs=self.systems[1],
        )

    @property
    def full_matrix(self):
        return np.eye(self.dims[0] + 1)

    def __mul__(self, tr):
        from coorx import CompositeTransform

        if isinstance(tr, CompositeTransform):
            return tr.__rmul__(self)
        if isinstance(tr.inverse, CompositeTransform):
            return tr.inverse.__mul__(self.inverse).inverse
        self.validate_transform_for_mul(tr)
        return tr.copy(from_cs=tr.systems[0], to_cs=self.systems[1])

    def __rmul__(self, tr):
        from coorx import CompositeTransform

        if isinstance(tr, CompositeTransform):
            return tr.__mul__(self)
        if isinstance(tr.inverse, CompositeTransform):
            return tr.inverse.__rmul__(self.inverse).inverse
        tr.validate_transform_for_mul(self)
        return tr.copy(from_cs=self.systems[0], to_cs=tr.systems[1])


class TransposeTransform(Transform):
    """Transposes columns of input coordinates (as opposed to dimensions)."""

    Linear = True
    Orthogonal = True
    NonScaling = False
    Isometric = True
    Equidimensional = True

    parameter_spec = [
        TupleParameter("axis_order", length='dims0'),
    ]

    def __init__(self, axis_order: None | tuple[int, ...] = None, *args, **kwargs):
        super().__init__(*args, **kwargs, axis_order=axis_order)

    @property
    def axis_order(self):
        return self._state['axis_order']

    @axis_order.setter
    def axis_order(self, order):
        self.set_params(axis_order=order)

    def _map(self, coords):
        """Return the input array with columns swapped."""
        return coords[..., self.axis_order]

    def _imap(self, coords):
        """Return the input array columns inversely swapped."""
        return coords[..., np.argsort(self.axis_order)]

    @property
    def params(self):
        return {'axis_order': self.axis_order}

    def as_affine(self):
        return AffineTransform(
            matrix=np.eye(self.dims[0])[:, self.axis_order],
            offset=np.zeros(self.dims[0]),
            from_cs=self.systems[0],
            to_cs=self.systems[1],
        )

    def __rmul__(self, tr):
        if isinstance(tr, TransposeTransform):
            tr.validate_transform_for_mul(self)
            return TransposeTransform(
                axis_order=tuple(self._map(np.array(tr.axis_order))),
                from_cs=self.systems[0],
                to_cs=tr.systems[1],
            )
        return super().__rmul__(tr)


class TTransform(Transform):
    """Transform performing only translation.

    Input/output dimensionality of this transform may be set by the length of the offset parameter.

    Parameters
    ----------
    offset : array-like
        Translation distances.
    dims : tuple
        (input, output) dimensions for transform.
    """

    Linear = True
    Orthogonal = True
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        ArrayParameter(
            "offset",
            dtype=float,
            shape=('dims0',),
            default=0,
        ),
    ]

    def __init__(self, offset=None, dims: Dims = None, **kwargs):
        super().__init__(dims, offset=offset, **kwargs)

    def _map(self, coords):
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
        return coords + self._state["offset"][np.newaxis, :]

    def _imap(self, coords):
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
        return coords - self._state["offset"][np.newaxis, :]

    @property
    def offset(self):
        return self._state['offset'].copy()

    @offset.setter
    def offset(self, t):
        self.set_params(offset=t)

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
        m = AffineTransform(dims=self.dims, from_cs=self.systems[0], to_cs=self.systems[1])
        m.translate(self.offset)
        return m

    def as_st(self):
        return STTransform(
            offset=self.offset,
            scale=(1,) * self.dims[0],
            from_cs=self.systems[0],
            to_cs=self.systems[1],
        )

    def __mul__(self, tr):
        self.validate_transform_for_mul(tr)
        if isinstance(tr, TTransform):
            return TTransform(self.offset + tr.offset, from_cs=tr.systems[0], to_cs=self.systems[1])
        elif isinstance(tr, STTransform):
            return self.as_st() * tr
        elif isinstance(tr, AffineTransform):
            return self.as_affine() * tr
        else:
            return super().__mul__(tr)

    def __rmul__(self, tr):
        tr.validate_transform_for_mul(self)
        if isinstance(tr, STTransform):
            return tr * self.as_st()
        if isinstance(tr, AffineTransform):
            return tr * self.as_affine()
        return super().__rmul__(tr)

    def __repr__(self):
        return f"<TTransform offset={self.offset} at 0x{id(self)}>"


class STTransform(Transform):
    """Transform performing only scale and translate, in that order.

    Input/output dimensionality of this transform may be set by the length of the scale or offset parameters.

    Parameters
    ----------
    scale : array-like
        Scale factors.
    offset : array-like
        Translation distances.
    """

    Linear = True
    Orthogonal = True
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        ArrayParameter("scale", dtype=float, shape=('dims0',), default=1),
        ArrayParameter("offset", dtype=float, shape=('dims0',), default=0),
    ]

    def __init__(self, scale=None, offset=None, dims=None, **kwargs):
        super().__init__(dims, **kwargs, scale=scale, offset=offset)

    def _map(self, coords):
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
        return coords * self.scale[None, :] + self.offset[None, :]

    def _imap(self, coords):
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
        return (coords - self.offset[None, :]) / self.scale[None, :]

    @property
    def scale(self):
        return self._state["scale"].copy()

    @scale.setter
    def scale(self, s):
        self.set_params(scale=s)

    @property
    def offset(self):
        return self._state["offset"].copy()

    @offset.setter
    def offset(self, t):
        self.set_params(offset=t)

    def translate(self, offset):
        """Change the translation of this transform by the amount given.

        Parameters
        ----------
        offset : array-like
            The values to be added to the current translation of the transform.
        """
        offset = np.asarray(offset)
        self.offset = self.offset + offset

    def zoom(self, zoom, center=None, mapped=True):
        """Update the transform such that its scale factor is changed, but
        the specified center point is left unchanged.

        Parameters
        ----------
        zoom : array-like
            Values to multiply the transform's current scale
            factors.
        center : array-like
            The center point around which the scaling will take place. Defaults to the origin.
        mapped : bool
            Whether *center* is expressed in mapped coordinates (True) or
            unmapped coordinates (False).
        """
        zoom = np.asarray(zoom)
        if center is None:
            center = np.zeros(self.dims[0])
        center = np.asarray(center)
        assert zoom.shape == center.shape == (self.dims[0],)
        scale = self.scale * zoom
        if mapped:
            trans = center - (center - self.offset) * zoom
        else:
            trans = self.scale * (1 - zoom) * center + self.offset
        self.set_params(scale=scale, offset=trans)

    def as_affine(self):
        m = AffineTransform(dims=self.dims, from_cs=self.systems[0], to_cs=self.systems[1])
        m.zoom(self.scale)
        m.translate(self.offset)
        return m

    def as_vispy(self):
        if self.dims != (3, 3):
            raise NotImplementedError("as_vispy is only implemented for 3D transforms")
        from vispy.visuals.transforms import STTransform as VispySTTransform

        return VispySTTransform(scale=self.scale, translate=self.offset)

    @classmethod
    def from_mapping(cls, x0, x1):
        """Create an STTransform from the given mapping

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
        if x0.ndim != 2 or x0.shape[0] != 2 or x1.ndim != 2 or x1.shape[0] != 2:
            raise TypeError("set_mapping requires array inputs of shape " "(2, N).")
        denom = x0[1] - x0[0]
        mask = denom == 0
        denom[mask] = 1.0
        s = (x1[1] - x1[0]) / denom
        s[mask] = 1.0
        t = x1[0] - s * x0[0]
        self.set_params(scale=s, offset=t)

    def __mul__(self, tr):
        self.validate_transform_for_mul(tr)
        if isinstance(tr, STTransform):
            s = self.scale * tr.scale
            t = self.offset + (tr.offset * self.scale)
            return STTransform(scale=s, offset=t, from_cs=tr.systems[0], to_cs=self.systems[1])
        elif isinstance(tr, AffineTransform):
            return self.as_affine() * tr
        else:
            return super().__mul__(tr)

    def __rmul__(self, tr):
        tr.validate_transform_for_mul(self)
        if isinstance(tr, AffineTransform):
            return tr * self.as_affine()
        return super().__rmul__(tr)

    def __repr__(self):
        return f"<STTransform scale={self.scale} offset={self.offset} at 0x{id(self)}>"


class AffineTransform(Transform):
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
    Equidimensional = False

    parameter_spec = [
        ArrayParameter(
            "matrix",
            dtype=float,
            shape=('dims1', 'dims0'),
            default=lambda shape: np.eye(max(shape))[: shape[1], : shape[0]],
        ),
        ArrayParameter("offset", dtype=float, shape=('dims1',), default=0),
    ]

    def __init__(self, matrix=None, offset=None, dims: Dims = None, **kwargs):
        super().__init__(dims, **kwargs, matrix=matrix, offset=offset)

    def _init_with_no_state(self):
        self._inv_matrix = None
        super()._init_with_no_state()

    def _map(self, coords):
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
            raise TypeError(
                "Shape of last axis (%d) is not equal to input dimension of transform (%d)"
                % (coords.shape[-1], self.dims[0])
            )
        return np.dot(self._state["matrix"], coords.T).T + self._state["offset"][None, :]

    def _imap(self, coords):
        """Inverse map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map. The length of the last array dimension
            must be equal to the input dimensionality of the transform.

        Returns
        -------
        coords : ndarray
            Mapped coordinates: M_inv * (coords - offset)
        """
        if coords.shape[-1] != self.dims[1]:
            raise TypeError(
                "Shape of last axis (%d) is not equal to output dimension of transform (%d)"
                % (coords.shape[-1], self.dims[1])
            )
        return np.dot(self.inv_matrix, (coords + self.inv_offset[None, :]).T).T

    @property
    def matrix(self):
        return self._state["matrix"].copy()

    @matrix.setter
    def matrix(self, m):
        self.set_params(matrix=m)

    @property
    def offset(self):
        return self._state["offset"].copy()

    @offset.setter
    def offset(self, o):
        self.set_params(offset=o)

    @property
    def inv_matrix(self):
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self._state["matrix"])
        return self._inv_matrix

    @property
    def inv_offset(self):
        return -self._state["offset"]

    def as_affine(self):
        return AffineTransform(
            matrix=self.matrix,
            offset=self.offset,
            from_cs=self.systems[0],
            to_cs=self.systems[1],
        )

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
        m = np.zeros((self.dims[1] + 1, self.dims[0] + 1))
        m[:-1, :-1] = self._state["matrix"]
        m[:-1, -1] = self._state["offset"]
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

    def zoom(self, scale, center=None):
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
        if np.isscalar(scale):
            scale = (scale,) * self.dims[0]
        scale_matrix = np.zeros(self.dims[::-1])
        for i in range(min(self.dims)):
            scale_matrix[i, i] = scale[i]

        if center is not None:
            raise NotImplementedError()
        self.matrix = np.dot(scale_matrix, self.matrix)
        self.offset = np.dot(scale_matrix, self.offset)

    def rotate(self, angle, axis=None):
        """
        Rotate the matrix by some angle about a given axis.

        The rotation is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        angle : float
            The angle of rotation in degrees.
        axis : array-like or None
            The x, y and z coordinates of the axis vector to rotate around (only for 3D).
        """
        if self.dims == (2, 2):
            rm = matrices.rotate2d(angle)
        elif self.dims == (3, 3):
            rm = matrices.rotate3d(angle, axis)
        else:
            raise TypeError("Rotation only supported for 2D and 3D affine transforms")
        self.matrix = np.dot(rm, self.matrix)
        self.offset = np.dot(rm, self.offset)

    def set_mapping(self, points1, points2):
        """Set to a transformation matrix that maps points1 onto points2.

        Parameters
        ----------
        points1 : array-like, shape (4, 3)
            Four starting coordinates.
        points2 : array-like, shape (4, 3)
            Four ending coordinates.
        """
        m = matrices.affine_map(points1, points2)
        self.set_params(matrix=m[:, :-1], offset=m[:, -1])

    def __mul__(self, tr):
        self.validate_transform_for_mul(tr)
        if isinstance(tr, AffineTransform):
            m = np.dot(self.full_matrix, tr.full_matrix)
            return AffineTransform(
                matrix=m[:-1, :-1], offset=m[:-1, -1], from_cs=tr.systems[0], to_cs=self.systems[1]
            )
        return tr.__rmul__(self)

    def __eq__(self, tr):
        if not isinstance(tr, AffineTransform):
            # todo: we can assess equality for some others like TTransform and STTransform
            return False
        return np.all(self.full_matrix == tr.full_matrix)

    @classmethod
    def from_qmatrix4x4(cls, matrix, *init_args, **init_kwargs):
        """Create an AffineTransform from a QMatrix4x4."""
        # Extract the matrix data and reshape it to 4x4
        data = matrix.data()
        # QMatrix4x4 data() returns 16 values in column-major order
        matrix_array = np.array(data).reshape(4, 4, order='F')  # Fortran order for column-major

        # Create an AffineTransform from the matrix
        return cls.from_matrix(matrix_array, *init_args, **init_kwargs)

    @classmethod
    def from_matrix(cls, matrix, *init_args, **init_kwargs):
        offset = matrix[:-1, -1]
        matrix = matrix[:-1, :-1]
        return cls(matrix=matrix, offset=offset, *init_args, **init_kwargs)

    def copy(self, from_cs=None, to_cs=None):
        return AffineTransform(
            matrix=self.matrix,
            offset=self.offset,
            from_cs=from_cs or self.systems[0],
            to_cs=to_cs or self.systems[1],
        )


class SRT2DTransform:
    def __init__(self, **kwds):
        raise NotImplementedError()


class SRT3DTransform(Transform):
    """3D transform implemented as 4x4 affine that can always be represented as a combination of 3 matrices: scale * rotate * translate
    This transform has no shear; angles are always preserved.

    Parameters
    ----------
    offset : array-like | None
        Translation distances. Default is (0, 0, 0).
    scale : array-like | None
        Scale factors. Default is (1, 1, 1).
    angle : float | None
        Rotation angle in degrees. Default is 0.
    axis : array-like | None
        Rotation axis. Default is (0, 0, 1).
    """

    Linear = True
    Orthogonal = True
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        ArrayParameter("offset", dtype=float, shape=(3,), default=0),
        ArrayParameter("scale", dtype=float, shape=(3,), default=1),
        ArrayParameter("axis", dtype=float, shape=(3,), default=lambda shape: np.array((0, 0, 1))),
        FloatParameter("angle", default=0.0),
    ]

    def __init__(self, offset=None, scale=None, angle=None, axis=None, init=None, **kwds):
        kwds.setdefault('dims', (3, 3))
        if kwds['dims'] != (3, 3):
            raise ValueError("SRT3DTransform must have dims (3, 3)")
        super().__init__(**kwds, offset=offset, scale=scale, angle=angle, axis=axis)

        # TODO the following looks like untested code
        # if all(p is None for p in (offset, scale, angle, axis)):
        #     if init is not None:
        #         if isinstance(init, SRT3DTransform):
        #             self.set_state(**init._state)
        #         elif isinstance(init, SRT2DTransform):
        #             self.set_state(
        #                 offset=tuple(init._state["offset"]) + (0,),
        #                 scale=tuple(init._state["scale"]) + (1,),
        #                 angle=init._state["angle"],
        #                 axis=(0, 0, 1),
        #             )
        #         elif isinstance(init, AffineTransform):
        #             self.set_from_affine(init)
        #         else:
        #             raise TypeError("Cannot build SRTTransform3D from argument type:", type(init))
        # else:
        #     assert init is None
        #     self.set_params(offset, scale, angle, axis)

    def _init_with_no_state(self):
        super()._init_with_no_state()
        self._affine = None

    @property
    def scale(self):
        return self._state["scale"].copy()

    @scale.setter
    def scale(self, s):
        self.set_params(scale=s)

    @property
    def rotation(self):
        """Return (angle, axis) of rotation"""
        return self._state["angle"], self._state["axis"].copy()

    @rotation.setter
    def rotation(self, angle_axis):
        """Set the transformation rotation to angle (in degrees)"""
        if np.isscalar(angle_axis):
            angle = angle_axis
            axis = (0.0, 0.0, 1.0)
        else:
            angle, axis = angle_axis

        self.set_params(angle=angle, axis=axis)

    @property
    def offset(self):
        return self._state["offset"].copy()

    @offset.setter
    def offset(self, o):
        self.set_params(offset=o)

    def translate(self, offset):
        """Adjust the translation of this transform"""
        self.offset = self._state["offset"] + offset

    def zoom(self, zoom, center=None, mapped=True):
        """Update the transform such that its scale factor is changed, but
        the specified center point is left unchanged.

        Parameters
        ----------
        zoom : array-like
            Values to multiply the transform's current scale
            factors.
        center : array-like
            The center point around which the scaling will take place. Defaults to the origin.
        mapped : bool
            Whether *center* is expressed in mapped coordinates (True) or
            unmapped coordinates (False).
        """
        zoom = np.asarray(zoom)
        if center is None:
            center = np.zeros(self.dims[0])
        center = np.asarray(center)
        assert zoom.shape == center.shape == (self.dims[0],)
        scale = self.scale * zoom
        if mapped:
            trans = center - (center - self.offset) * zoom
        else:
            trans = self.scale * (1 - zoom) * center + self.offset
        self.set_params(scale=scale, offset=trans)

    def rotate(self, angle, axis):
        """Adjust the rotation of this transform"""
        axis = np.asarray(axis)
        origAxis = self._state["axis"]
        if np.all(axis == origAxis):
            self.rotation = self._state["angle"] + angle, axis
        else:
            m = AffineTransform(dims=self.dims)
            m.translate(self._state["offset"])
            m.rotate(self._state["angle"], self._state["axis"])
            m.rotate(angle, axis)
            m.zoom(self._state["scale"])
            self.set_from_affine(m)

    def set_from_affine(self, tr):
        """
        Set this transform based on the elements of *m*
        The input matrix must be affine AND have no shear,
        otherwise the conversion will most likely fail.
        """
        assert tr.dims == (3, 3)

        # scale is vector-length of first three matrix columns
        m = tr.matrix.T
        scale = (m**2).sum(axis=0) ** 0.5

        # see whether there is an inversion
        z = np.cross(m[0], m[1])
        if np.dot(z, m[2]) < 0:
            scale[1] *= -1  ## doesn't really matter which axis we invert

        ## rotation axis is the eigenvector with eigenvalue=1
        r = m / scale[np.newaxis, :]
        try:
            evals, evecs = numpy.linalg.eig(r)
        except Exception:
            print(f"Rotation matrix: {r}")
            print(f"Scale: {scale}")
            print(f"Original matrix: {m}")
            raise
        eigIndex = np.argwhere(np.abs(evals - 1) < 1e-6)
        if len(eigIndex) < 1:
            print(f"eigenvalues: {evals}")
            print(f"eigenvectors: {evecs}")
            print(f"index: {eigIndex}, {evals - 1}")
            raise ValueError("Could not determine rotation axis.")
        axis = evecs[:, eigIndex[0, 0]].real
        axis /= ((axis**2).sum()) ** 0.5

        # trace(r) == 2 cos(angle) + 1, so:
        cos = (r.trace() - 1) * 0.5  # this only gets us abs(angle)

        # The off-diagonal values can be used to correct the angle ambiguity,
        # but we need to figure out which element to use:
        axisInd = np.argmax(np.abs(axis))
        rInd, sign = [((1, 2), -1), ((0, 2), 1), ((0, 1), -1)][axisInd]

        # Then we have r-r.T = sin(angle) * 2 * sign * axis[axisInd];
        # solve for sin(angle)
        sin = (r - r.T)[rInd] / (2.0 * sign * axis[axisInd])

        # finally, we get the complete angle from arctan(sin/cos)
        angle = np.arctan2(sin, cos) * 180 / np.pi
        if angle == 0:
            axis = (0, 0, 1)

        self.set_params(offset=tr.offset, scale=scale, angle=angle, axis=axis)

    @property
    def full_matrix(self):
        return self._get_affine().full_matrix

    def as2D(self):
        """Return an SRT2DTransform representing the x,y portion of this transform (if possible)"""
        return SRT2DTransform(init=self)

    @property
    def params(self):
        return {
            "offset": tuple(self._state["offset"]),
            "scale": tuple(self._state["scale"]),
            "angle": self._state["angle"],
            "axis": tuple(self._state["axis"]),
        }

    def _map(self, arr):
        return self._get_affine()._map(arr)

    def _imap(self, arr):
        return self._get_affine()._imap(arr)

    def set_mapping(self, points1, points2):
        """Set to a transformation that maps points1 onto points2.

        Parameters
        ----------
        points1 : ndarray, shape (N, 3)
            Input coordinates.
        points2 : ndarray, shape (N, 3)
            Output coordinates.
        """
        aff = AffineTransform(dims=(3, 3))
        aff.set_mapping(points1, points2)
        self.set_from_affine(aff)
        return

        params = self.params
        # params_flat = []
        # param_len = {}
        # for k,v in params.items():
        #     if np.isscalar(v):
        #         params_flat.append(v)
        #         param_len[k] = 0
        #     else:
        #         params_flat.extend(v)
        #         param_len[k] = len(v)
        params_flat = (
            list(params["offset"]) + [params["scale"][0]] + list(params["axis"]) + [params["angle"]]
        )
        params_flat = (
            list(params["offset"])
            + list(params["scale"])
            + list(params["axis"])
            + [params["angle"]]
        )
        x0 = np.array(params_flat)

        def unflatten_params(x):
            # return {'offset': x[:3], 'scale': (x[3], x[3], x[3]), 'axis': x[4:7], 'angle': x[7]}
            return {"offset": x[:3], "scale": x[3:6], "axis": x[6:9], "angle": x[9]}
            # params = {}
            # i = 0
            # for k,l in param_len.items():
            #     if l == 0:
            #         params[k] = x[i]
            #         i += 1
            #     else:
            #         params[k] = x[i:i+l]
            #         i += l
            # return params

        def err_func(tr, points1, points2):
            mapped = tr.map(points1)
            err = (mapped - points2).flatten()
            err = (err**2).mean() ** 0.5
            return err

        def flat_err_func(x, points1, points2):
            params = unflatten_params(x)
            tr = SRT3DTransform(**params)
            return err_func(tr, points1, points2)

        # result = scipy.optimize.leastsq(err_func, x0, args=(points1, points2))
        # params = unflatten_params(result[0])
        result = scipy.optimize.minimize(
            flat_err_func,
            x0,
            args=(points1, points2),
            # method=None,
            # method='CG',
            # method=None,
            # method=None,
        )
        params = unflatten_params(result.x)

        self.set_params(**params)
        return err_func(self, points1, points2)

    def as_affine(self):
        affine = AffineTransform(dims=(3, 3), from_cs=self.systems[0], to_cs=self.systems[1])
        affine.zoom(self._state["scale"])
        affine.rotate(self._state["angle"], self._state["axis"])
        affine.translate(self._state["offset"])
        # TODO figure out if this can be generalized to all meta data
        return affine

    def _get_affine(self):
        if getattr(self, "_affine", None) is None:
            self._affine = self.as_affine()
        return self._affine

    def _update(self, source_event=None):
        self._affine = None
        super()._update(source_event=source_event)

    def __repr__(self):
        return (
            f'<SRT3DTransform offset={self._state["offset"]} scale={self._state["scale"]}'
            f' angle={self._state["angle"]} axis={self._state["axis"]} at 0x{id(self)}>'
        )

    def __mul__(self, tr):
        self.validate_transform_for_mul(tr)
        if isinstance(tr, SRT3DTransform):
            return self._get_affine() * tr._get_affine()
        elif isinstance(tr, AffineTransform):
            return self._get_affine() * tr
        else:
            return tr.__rmul__(self)

    @classmethod
    def from_pyqtgraph(cls, pg_transform, *init_args, **init_kwargs):
        """Create an SRT3DTransform from a pyqtgraph Transform3D instance"""
        from pyqtgraph import SRTTransform3D, Transform3D

        if not isinstance(pg_transform, (SRTTransform3D, Transform3D)):
            raise TypeError("Input must be a SRTTransform3D instance")
        offset = pg_transform.getTranslation()
        scale = pg_transform.getScale()
        angle, axis = pg_transform.getRotation()
        angle = -angle  # pyqtgraph uses left-handed rotations
        return cls(*init_args, **init_kwargs, offset=offset, scale=scale, angle=angle, axis=axis)


class PerspectiveTransform(Transform):
    """3D perspective or orthographic matrix transform using homogeneous coordinates.

    Assumes a camera at the origin, looking toward the -Z axis.
    The camera's top points toward +Y, and right points toward +X.

    Points inside the perspective frustum are mapped to the range [-1, +1] along all three axes.
    """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        TransformParameter("affine", dims=(4, 4), default=lambda: AffineTransform(dims=(4, 4))),
    ]

    def __init__(self, affine=None, **kwds):
        kwds.setdefault('dims', (3, 3))
        if kwds['dims'] != (3, 3):
            raise ValueError("PerspectiveTransform must have dims (3, 3)")
        super().__init__(**kwds, affine=affine)

    @property
    def affine(self):
        return self._state["affine"]

    def _map(self, arr):
        arr4 = np.empty((arr.shape[0], 4), dtype=arr.dtype)
        arr4[:, :3] = arr
        arr4[:, 3] = 1
        out = self.affine.map(arr4)
        return out[:, :3] / out[:, 3:4]

    def as_affine(self):
        # TODO is this correct?
        return self.affine.as_affine()

    @property
    def full_matrix(self):
        # TODO is this correct?
        return self.affine.full_matrix

    def set_ortho(self, left, right, bottom, top, znear, zfar):
        """Set orthographic transform."""
        assert right != left
        assert bottom != top
        assert znear != zfar

        M = np.zeros((4, 4), dtype=np.float32)
        M[0, 0] = +2.0 / (right - left)
        M[3, 0] = -(right + left) / float(right - left)
        M[1, 1] = +2.0 / (top - bottom)
        M[3, 1] = -(top + bottom) / float(top - bottom)
        M[2, 2] = -2.0 / (zfar - znear)
        M[3, 2] = -(zfar + znear) / float(zfar - znear)
        M[3, 3] = 1.0
        self.affine.matrix = M.T

    def set_perspective(self, fovy, aspect, znear, zfar):
        """Set the perspective

        Parameters
        ----------
        fov : float
            Field of view.
        aspect : float
            Aspect ratio.
        near : float
            Near location.
        far : float
            Far location.
        """
        assert znear != zfar
        h = np.tan(fovy * np.pi / 360.0) * znear
        w = h * aspect
        self.set_frustum(-w, w, -h, h, znear, zfar)

    def set_frustum(self, left, right, bottom, top, near, far):  # noqa
        """Set the frustum"""
        M = matrices.frustum(left, right, bottom, top, near, far)
        self.affine.matrix = M.T

    def __eq__(self, tr):
        if not isinstance(tr, PerspectiveTransform):
            return False
        return self.affine == tr.affine


class BilinearTransform(Transform):
    """2D bilinear transform."""

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        ArrayParameter("matrix", dtype=float, shape=(2, 4), default=lambda shape: np.eye(4)[:2]),
        ArrayParameter("inv_matrix", dtype=float, shape=(2, 4), default=lambda shape: np.eye(4)[:2]),
    ]

    def __init__(self, **kwds):
        kwds.setdefault('dims', (2, 2))
        if kwds['dims'] != (2, 2):
            raise ValueError("BilinearTransform must have dims (2, 2)")
        super().__init__(**kwds)

    def set_mapping(self, points1, points2):
        """Set to a transformation matrix that maps points1 onto points2.

        Arguments must be array-like with shape (4, 2).
        """
        # convert inputs to (4, 2) arrays
        points1 = self._prepare_arg_for_mapping(points1)[0]
        points2 = self._prepare_arg_for_mapping(points2)[0]
        assert (
            points1.shape == points2.shape == (4, 2)
        ), f"Input arrays must have shape (4, 2); got ({points1.shape}, {points2.shape})"

        self.set_params(matrix=self._solve_matrix(points1, points2), inv_matrix=self._solve_matrix(points2, points1))

    @staticmethod
    def _solve_matrix(a, b):
        # solve 2 sets of linear equations to determine transformation matrix elements
        c = BilinearTransform._prepare_for_mapping(a)
        matrix = np.zeros((2, 4))
        for i in range(2):
            # solve Ax = B; x is one row of the desired transformation matrix
            matrix[i] = numpy.linalg.solve(c, b[:, i])
        return matrix

    @staticmethod
    def _prepare_for_mapping(points):
        # convert [[x, y], ...] to [[x*y, x, y, 1], ...]
        assert points.ndim >= 2
        assert points.shape[-1] == 2
        out = np.empty(points.shape[:-1] + (4,), dtype=points.dtype)
        out[..., 0] = points[..., 0] * points[..., 1]
        out[..., 1:3] = points
        out[..., 3] = 1
        return out

    @property
    def matrix(self):
        return self._state["matrix"].copy()

    @property
    def inv_matrix(self):
        return self._state["inv_matrix"].copy()

    def _map(self, arr):
        arr4 = BilinearTransform._prepare_for_mapping(arr)
        out = np.dot(arr4, self._state["matrix"].T)
        return out[:, :2]

    def _imap(self, arr):
        arr4 = BilinearTransform._prepare_for_mapping(arr)
        out = np.dot(arr4, self._state["inv_matrix"].T)
        return out[:, :2]


class Homography2DTransform(Transform):
    """2D homography transform."""

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        ArrayParameter("matrix", dtype=float, shape=(3, 3), default=lambda shape: np.eye(3)),
        ArrayParameter("inv_matrix", dtype=float, shape=(3, 3), default=lambda shape: np.eye(3)),
    ]

    def __init__(self, **kwds):
        kwds.setdefault('dims', (2, 2))
        if kwds['dims'] != (2, 2):
            raise ValueError("Homography2DTransform must have dims (2, 2)")
        super().__init__(**kwds)

    def matrix(self):
        return self._state["matrix"].copy()

    @property
    def inv_matrix(self):
        return self._state["inv_matrix"].copy()

    def _map(self, arr):
        arr4 = self._prepare_for_mapping(arr)
        out = np.dot(arr4, self._state["matrix"].T)
        return out[:, :2] / out[:, 2:3]

    def _imap(self, arr):
        arr4 = self._prepare_for_mapping(arr)
        out = np.dot(arr4, self._state["inv_matrix"].T)
        return out[:, :2] / out[:, 2:3]
