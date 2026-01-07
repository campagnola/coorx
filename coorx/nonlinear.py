import warnings

import numpy as np

from .base_transform import Transform
from .params import ArrayParameter, TupleParameter


class LogTransform(Transform):
    """ND transform performing logarithmic transformation.

    Maps (x, y, z) => (log(base_x, x), log(base_y, y), log(base_z, z))

    Special base values:
    - None: Identity transformation (no change to that axis)
    - Negative values: Inverse (exponential) transformation (x => -base^x)
    - 1 or 0: Mathematically nonsensical but allowed (may produce inf/nan)
    - Fractional values: Standard logarithm (log(x)/log(base))

    Parameters
    ----------
    base : array-like
        Base values for each axis; length must be the same as the dimensionality of the transform.
        A base value of None provides identity transformation for that axis.
        Negative bases apply inverse (exponential) transformation.
    """

    Linear = False
    Orthogonal = True
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        TupleParameter("base", length="dims0", default=lambda shape: np.array([None] * shape[0])),
    ]

    def __init__(self, base=None, dims=None, **kwargs):
        super().__init__(dims, base=base, **kwargs)

    @property
    def base(self) -> tuple[float | None]:
        """
        *base* is a tuple containing the log base values that should be
        applied to each axis of the input vector. If any axis has a base == None,
        then that axis is not affected (identity transformation).
        """
        return self._state['base']

    @base.setter
    def base(self, s):
        self.set_params(base=s)

    def _map(self, coords):
        return self._map_with_base(coords, self._state['base'])

    def _imap(self, coords):
        return self._map_with_base(coords, [b if b is None else -b for b in self.base])

    @staticmethod
    def _map_with_base(coords, base):
        # Ensure output dtype can handle floating point values including NaN
        ret = _empty_array_like(coords)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # divide-by-zeros and invalid values
            for i in range(min(ret.shape[-1], 3)):
                if base[i] is None:
                    ret[..., i] = coords[..., i]
                elif base[i] > 0.0:
                    ret[..., i] = np.log(coords[..., i]) / np.log(base[i])
                else:  # base < 0 treated as inverse
                    ret[..., i] = (-base[i]) ** coords[..., i]

        ret[~np.isfinite(ret)] = np.nan  # set all non-finite values to NaN
        return ret

    def __repr__(self):
        return f"<LogTransform base={self._state['base']}>"


def _empty_array_like(data):
    output_dtype = data.dtype if data.dtype.kind == 'f' else np.float64
    return np.empty(data.shape, output_dtype)


class PolarTransform(Transform):
    """Polar transform

    Maps (theta, r, ...) to (x, y, ...), where `x = r*cos(theta)`
    and `y = r*sin(theta)`.
    """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = True

    def __init__(self, dims=None, **kwargs):
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        ret = _empty_array_like(coords)
        ret[..., 0] = coords[..., 1] * np.cos(coords[..., 0])
        ret[..., 1] = coords[..., 1] * np.sin(coords[..., 0])
        for i in range(2, coords.shape[-1]):  # copy any further axes
            ret[..., i] = coords[..., i]
        return ret

    def _imap(self, coords):
        ret = _empty_array_like(coords)
        ret[..., 0] = np.arctan2(
            coords[..., 1], coords[..., 0]
        )  # arctan2(y, x) for correct quadrant
        ret[..., 1] = (coords[..., 0] ** 2 + coords[..., 1] ** 2) ** 0.5
        for i in range(2, coords.shape[-1]):  # copy any further axes
            ret[..., i] = coords[..., i]
        return ret


# class SphericalTransform(Transform):
#    # TODO
#    pass


# class WarpTransform(Transform):
#    """ Multiple bilinear transforms in a grid arrangement.
#    """
#    # TODO


class LensDistortionTransform(Transform):
    """https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    Coefficients are (k1, k2, p1, p2, k3)
    Where k1, k2, and k3 are radial distortion (coordinates are multiplied by 1 + k1*r^2 + k2*r^4 + k3*r^6),
    and p1, p2 are tangential distortion coefficients.
    """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = True

    parameter_spec = [
        ArrayParameter(
            "coeff", dtype=float, shape=(5,), default=lambda shape: np.array([0, 0, 0, 0, 0])
        ),
    ]

    def __init__(self, **kwds):
        kwds.setdefault('dims', (2, 2))
        if kwds['dims'] != (2, 2):
            raise ValueError("LensDistortionTransform only supports 2D transforms")
        super().__init__(**kwds)

    @property
    def coeff(self):
        """
        *coeff* is an array of lens distortion coefficients (k1, k2, p1, p2, k3).
        """
        return self._state['coeff']

    def set_coeff(self, coeff):
        self.set_params(coeff=coeff)

    def _map(self, arr):
        k1, k2, p1, p2, k3 = self.coeff

        # radial distortion
        r = np.linalg.norm(arr, axis=1)
        dist = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
        out = arr * dist[:, None]

        # tangential distortion
        x = out[:, 0]
        y = out[:, 1]
        xy = x * y
        r2 = r**2
        out[:, 0] += 2 * p1 * xy + p2 * (r2 + 2 * x**2)
        out[:, 1] += 2 * p2 * xy + p1 * (r2 + 2 * y**2)

        return out

    def _imap(self, arr):
        """Inverse lens distortion mapping using iterative numerical method.

        Since lens distortion is nonlinear, we use Newton-Raphson iteration
        to find the undistorted coordinates that would map to the given distorted ones.
        """
        k1, k2, p1, p2, k3 = self.coeff

        # If all coefficients are zero, it's identity transform
        if all(c == 0 for c in self.coeff):
            return arr.copy()

        # Initial guess: use input as starting point
        undistorted = arr.copy()

        # Newton-Raphson iteration for inverse mapping
        for iteration in range(10):  # Maximum 10 iterations
            # Compute forward mapping of current guess
            forward = self._map(undistorted)

            # Compute residual (error)
            residual = forward - arr

            # Check convergence
            if np.allclose(residual, 0, atol=1e-8):
                break

            # Compute Jacobian matrix numerically
            eps = 1e-6
            jac = np.zeros((arr.shape[0], 2, 2))

            for i in range(2):
                perturbed = undistorted.copy()
                perturbed[:, i] += eps
                forward_perturbed = self._map(perturbed)
                jac[:, :, i] = (forward_perturbed - forward) / eps

            # Solve Jacobian * delta = -residual for delta
            delta = np.zeros_like(undistorted)
            try:
                # Use np.linalg.solve for each point
                for i in range(arr.shape[0]):
                    delta[i] = np.linalg.solve(jac[i], -residual[i])

                # Update guess
                undistorted += delta

            except np.linalg.LinAlgError:
                # If Jacobian is singular, use pseudo-inverse
                for i in range(arr.shape[0]):
                    delta[i] = np.linalg.pinv(jac[i]) @ (-residual[i])
                undistorted += delta

        return undistorted
