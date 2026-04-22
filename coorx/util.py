import numpy as np
import scipy.ndimage

from .base_transform import Transform
from .params import ArrayParameter, TransformParameter


def affine_resample(data, shape, origin, vectors, axes, order=1, **kwds):
    """Resample *data* on an affinely-transformed grid.

    For each output index ``(i, j, ...)``, the corresponding location in the
    original data is ``origin + i*vectors[0] + j*vectors[1] + ...``.  The
    output is computed with ``scipy.ndimage.map_coordinates``.

    Adapted from ``pyqtgraph.functions.affineSlice``.

    Parameters
    ----------
    data : ndarray
        Source data array.
    shape : tuple
        Output shape along the *axes* dimensions.
    origin : array-like, length ``len(axes)``
        Starting position in *data* coordinates.
    vectors : array-like, shape ``(len(shape), len(axes))``
        Basis vectors of the new grid expressed in *data* coordinates.
        Each row is the step in *data*-space for one unit along the
        corresponding output axis.
    axes : sequence of int
        Which axes of *data* the resampling applies to.
    order : int
        Spline interpolation order (default 1 = linear).
    **kwds
        Extra keyword arguments forwarded to
        ``scipy.ndimage.map_coordinates``.

    Returns
    -------
    ndarray
        Resampled array.  The resampled axes appear first (in the order given
        by *shape*), followed by any axes of *data* not listed in *axes*.
    """
    origin = np.asarray(origin, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    shape = tuple(int(np.ceil(s)) for s in shape)
    n_out = len(shape)
    n_ax = len(axes)

    if len(vectors) != n_out:
        raise ValueError("len(shape) must equal len(vectors)")
    if len(origin) != n_ax:
        raise ValueError("len(origin) must equal len(axes)")
    for v in vectors:
        if len(v) != n_ax:
            raise ValueError("each vector must have length len(axes)")

    # Build coordinate grid: coords[k, i, j, ...] is the position along
    # axes[k] in *data* that corresponds to output index (i, j, ...).
    origin_bc = origin.reshape((n_ax,) + (1,) * n_out)
    grid = np.mgrid[tuple(slice(0, s) for s in shape)]  # (n_out, s0, s1, ...)
    coords = (
        grid[np.newaxis, ...] * vectors.T[(Ellipsis,) + (np.newaxis,) * n_out]
    ).sum(axis=1) + origin_bc  # (n_ax, s0, s1, ...)

    # Transpose data so sampled axes come first, iterate over the rest.
    other_axes = [i for i in range(data.ndim) if i not in axes]
    data_t = data.transpose(tuple(axes) + tuple(other_axes))
    extra_shape = data_t.shape[n_ax:]
    output = np.empty(shape + extra_shape, dtype=data.dtype)
    for inds in np.ndindex(*extra_shape):
        output[(Ellipsis,) + inds] = scipy.ndimage.map_coordinates(
            data_t[(Ellipsis,) + inds], coords, order=order, **kwds
        )
    return output


class AxisSelectionEmbeddedTransform(Transform):
    """Wraps any transform in a larger-dimensional transform that passes specific axes to the wrapped
    transform, while keeping other axes unchanged.

    For example, you could use a 2D transform to affect just the x,y axes in a 3D space.
    """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = False

    parameter_spec = [
        ArrayParameter("axes", dtype=int, default=lambda shape: np.arange(shape[0])),
        TransformParameter("transform"),
    ]

    def __init__(self, axes, transform, **kwds):
        super().__init__(**kwds, axes=axes, transform=transform)

    def _map(self, arr):
        out = arr.copy()
        out[:, self._state["axes"]] = self._state["transform"].map(arr[:, self._state["axes"]])
        return out

    def _imap(self, arr):
        out = arr.copy()
        out[:, self._state["axes"]] = self._state["transform"].imap(arr[:, self._state["axes"]])
        return out


class HomogeneousEmbeddedTransform(Transform):
    """Wraps any transform that uses homogeneous coordinates,
    allowing to operate with nonhomogeneous inputs/outputs instead.
    """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = False

    parameter_spec = [
        TransformParameter("transform"),
    ]

    def __init__(self, transform, **kwds):
        expected_dims = (transform.dims[0] - 1, transform.dims[1] - 1)
        kwds.setdefault("dims", expected_dims)
        if kwds["dims"] != expected_dims:
            raise ValueError(
                f"Transform has dims {transform.dims}, expected "
                f"{(transform.dims[0]-1, transform.dims[1]-1)} for HomogeneousEmbeddedTransform"
            )
        super().__init__(**kwds, transform=transform)

    def _map(self, arr):
        out = self._state["transform"].map(self._to_homogeneous(arr))
        return self._from_homogeneous(out)

    def _imap(self, arr):
        out = self._state["transform"].imap(self._to_homogeneous(arr))
        return self._from_homogeneous(out)

    @staticmethod
    def _to_homogeneous(arr):
        hom = np.empty((arr.shape[0], arr.shape[1] + 1), dtype=arr.dtype)
        hom[:, :-1] = arr
        hom[:, -1] = 1
        return hom

    @staticmethod
    def _from_homogeneous(arr):
        return arr[:, :-1] / arr[:, -1:]
