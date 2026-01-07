import numpy as np

from .base_transform import Transform
from .params import ArrayParameter, TransformParameter


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
