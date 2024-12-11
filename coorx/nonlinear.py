import warnings
import numpy as np
from .base_transform import BaseTransform


class LogTransform(BaseTransform):
    """ND transform perfoming logarithmic transformation.

    Maps (x, y, z) => (log(base_x, x), log(base_y, y), log(base_z, z))

    No transformation is applied for axes with base == 0.

    If base < 0, then the inverse function is applied: x => base.x ** x

    Parameters
    ----------
    base : array-like
        Base values for each axis; length must be the same as the dimensionality of the transform. 
        A base value of 0 disables the transform for that axis.
    """
    Linear = False
    Orthogonal = True
    NonScaling = False
    Isometric = False

    def __init__(self, base=None, dims=None, **kwargs):
        if base is not None:
            base = np.asarray(base)
            if base.ndim != 1:
                raise TypeError("Base must be 1-D array-like")
        dims = self._dims_from_params(dims=dims, params={'base': base})
        
        super().__init__(dims, **kwargs)
        
        self._base = np.zeros(self.dims[0], dtype=np.float32)
        if base is not None:
            self.base = base

    @property
    def base(self):
        """
        *base* is a tuple containing the log base values that should be
        applied to each axis of the input vector. If any axis has a base == 0,
        then that axis is not affected.
        """
        return self._base.copy()

    @base.setter
    def base(self, s):
        self._base[:] = s

    def _map(self, coords, base=None):
        ret = np.empty(coords.shape, coords.dtype)
        if base is None:
            base = self.base
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # divide-by-zeros and invalid values
            for i in range(min(ret.shape[-1], 3)):
                if base[i] > 1.0:
                    ret[..., i] = np.log(coords[..., i]) / np.log(base[i])
                elif base[i] < -1.0:
                    ret[..., i] = -base[i] ** coords[..., i]
                else:
                    ret[..., i] = coords[..., i]
        ret[~np.isfinite(ret)] = np.nan  # set all non-finite values to NaN
        return ret

    def _imap(self, coords):
        return self.map(coords, -self.base)

    @property
    def params(self):
        return {'base': self.base}
    
    def set_params(self, base):
        self.base = base

    def __repr__(self):
        return "<LogTransform base=%s>" % (self.base)


class PolarTransform(BaseTransform):
    """Polar transform

    Maps (theta, r, z) to (x, y, z), where `x = r*cos(theta)`
    and `y = r*sin(theta)`.
    """
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=None, **kwargs):
        if dims is None:
            dims = (3, 3)
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        ret = np.empty(coords.shape, coords.dtype)
        ret[..., 0] = coords[..., 1] * np.cos(coords[..., 0])
        ret[..., 1] = coords[..., 1] * np.sin(coords[..., 0])
        for i in range(2, coords.shape[-1]):  # copy any further axes
            ret[..., i] = coords[..., i]
        return ret

    def _imap(self, coords):
        ret = np.empty(coords.shape, coords.dtype)
        ret[..., 0] = np.arctan2(coords[..., 0], coords[..., 1])
        ret[..., 1] = (coords[..., 0]**2 + coords[..., 1]**2) ** 0.5
        for i in range(2, coords.shape[-1]):  # copy any further axes
            ret[..., i] = coords[..., i]
        return ret

    @property
    def params(self):
        return {}
    
    def set_params(self):
        return


#class BilinearTransform(BaseTransform):
#    # TODO
#    pass


#class WarpTransform(BaseTransform):
#    """ Multiple bilinear transforms in a grid arrangement.
#    """
#    # TODO




class LensDistortionTransform(BaseTransform):
    """https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    Coefficients are (k1, k2, p1, p2, k3)
    Where k1, k2, and k3 are radial distortion (coordinates are multiplied by 1 + k1*r^2 + k2*r^4 + k3*r^6),
    and p1, p2 are tangential distortion coefficients.
    """
    def __init__(self, coeff=(0, 0, 0, 0, 0)):
        super().__init__(dims=(2, 2))
        self.coeff = coeff

    def set_coeff(self, coeff):
        self.coeff = coeff
        self._update()

    def _map(self, arr):
        k1, k2, p1, p2, k3 = self.coeff

        # radial distortion
        r = np.linalg.norm(arr, axis=1)
        dist = (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        out = arr * dist[:, None]

        # tangential distortion
        x = out[:, 0]
        y = out[:, 1]
        xy = x * y
        r2 = r**2
        out[:, 0] += 2 * p1 * xy + p2 * (r2 + 2 * x**2)
        out[:, 1] += 2 * p2 * xy + p1 * (r2 + 2 * y**2)

        return out
