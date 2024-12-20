import numpy as np
from .base_transform import Transform


class AxisSelectionEmbeddedTransform(Transform):
    """Wraps any transform in a larger-dimensional transform that passes specific axes to the wrapped
    transform, while keeping other axes unchanged.

    For example, you could use a 2D transform to affect just the x,y axes in a 3D space.
    """
    def __init__(self, axes, transform, **kwds):
        super().__init__(**kwds)
        self.axes = axes
        self.subtr = transform

    def _map(self, arr):
        out = arr.copy()
        out[:, self.axes] = self.subtr.map(arr[:, self.axes])
        return out

    def _imap(self, arr):
        out = arr.copy()
        out[:, self.axes] = self.subtr.imap(arr[:, self.axes])
        return out
    
    @property
    def params(self):
        return {'axes': self.axes, 'transform': self.subtr}

    def set_params(self, axes, transform):
        self.axes = axes
        self.subtr = transform
        self._update()



class HomogeneousEmbeddedTransform(Transform):
    """Wraps any transform that uses homogeneous coordinates, 
    allowing to operate with nonhomogeneous inputs/outputs instead. 
    """
    def __init__(self, transform, **kwds):
        expected_dims = (transform.dims[0]-1, transform.dims[1]-1)
        kwds.setdefault('dims', expected_dims)
        assert kwds['dims'] == expected_dims, "HomogeneousEmbeddedTransform dims must be %s" % (expected_dims,)
        super().__init__(**kwds)
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

    @property
    def params(self):
        return {'transform': self.subtr}
    
    def set_params(self, transform):
        self.subtr = transform
        self._update()
