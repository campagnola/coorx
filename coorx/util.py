import numpy as np
from .base_transform import BaseTransform


class AxisSelectionEmbeddedTransform(BaseTransform):
    """Wraps any transform in a larger-dimensional transform that passes specific axes to the wrapped
    transform, while keeping other axes unchanged.

    For example, you could use a 2D transform to affect just the x,y axes in a 3D space.
    """
    def __init__(self, axes, transform, dims):
        super().__init__(dims=dims)
        self.axes = axes
        self.subtr = transform

    def _map(self, arr):
        out = arr.copy()
        out[:, self.axes] = self.subtr.map(arr[:, self.axes])
        return out

    def _imap(self, arr):
        out = arr.copy()
        out[:, axes] = self.subtr.imap(arr[:, axes])
        return out


class HomogeneousEmbeddedTransform(BaseTransform):
    """Wraps any transform that uses homogeneous coordinates, 
    allowing to operate with nonhomogeneous inputs/outputs instead. 
    """
    def __init__(self, transform):
        super().__init__(dims=(transform.dims[0]-1, transform.dims[1]-1))
        self.subtr = transform

    def _map(self, arr):
        out = self.subtr.map(self._to_homogeneous(arr))
        return self._from_homogeneous(out)

    def _imap(self, arr):
        out = self.subtr.imap(self._to_homogeneous(arr))
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
