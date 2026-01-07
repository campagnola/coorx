"""
API Issues to work out:
  - Need a transform.map_rect function that returns the bounding rectangle of
    a rect after transformation. Non-linear transforms might need to work
    harder at this, but we can provide a default implementation that
    works by mapping a selection of points across a grid within the original
    rect.
"""

import contextlib
import inspect
import threading
import weakref
from copy import deepcopy
from typing import Callable

import numpy as np

from ._types import Dims, StrOrNone, Mappable
from .systems import CoordinateSystemGraph, CoordinateSystem


class ChangeEvent:
    def __init__(self, transform, source_event=None):
        self.transform = transform
        self.source_event = source_event

    @property
    def sources(self):
        """A list of all transforms that changed leading to this event"""
        s = [self]
        if self.source_event is not None:
            s += self.source_event.sources
        return s


class CallbackRegistry:
    def __init__(self):
        # List of (is_weakref, callback or weakref) tuples
        self._callbacks: list[tuple[bool, Callable]] = []
        self.lock = threading.Lock()

    def add(self, cb, keep_reference):
        if keep_reference:
            cb_ref = (False, cb)
        else:
            weak_self = weakref.ref(self)

            def cleanup(dead_ref):
                registry = weak_self()
                if registry is not None:
                    registry.remove(dead_ref)

            if inspect.ismethod(cb):
                cb_ref = (True, weakref.WeakMethod(cb, cleanup))
            else:
                cb_ref = (True, weakref.ref(cb, cleanup))

        with self.lock:
            self._callbacks.append(cb_ref)

    def remove(self, cb):
        with self.lock:
            new_callbacks = []
            for is_ref, maybe_cb in self._callbacks:
                if is_ref:
                    cb_from_ref = maybe_cb()
                    if cb_from_ref is None:
                        # Clean up dead weak refs, too
                        continue
                    if cb_from_ref == cb:
                        continue
                else:
                    if maybe_cb == cb:
                        continue
                new_callbacks.append((is_ref, maybe_cb))
            self._callbacks = new_callbacks

    def __iter__(self):
        with self.lock:
            # Make a snapshot of callbacks to invoke
            callbacks = [cb_ref() if is_ref else cb_ref for is_ref, cb_ref in self._callbacks]
        return iter([cb for cb in callbacks if cb is not None])


class Transform(object):
    """
    Transform is a base class that defines a pair of complementary
    coordinate mapping functions in both python and GLSL.

    All Transform subclasses define map() and imap() methods that map
    an object through the forward or inverse transformation, respectively.

    Optionally, an inverse() method returns a new transform performing the
    inverse mapping.

    Note that although all classes should define both map() and imap(), it
    is not necessarily the case that imap(map(x)) == x; there may be instances
    where the inverse mapping is ambiguous or otherwise meaningless.

    """

    # Flags used to describe the transformation. Subclasses should define each
    # as True or False.
    # (usually used for making optimization decisions)

    # If True, then for any set of colinear points, the
    # transformed points will also be colinear.
    Linear = None

    # The transformation's effect on one axis is independent
    # of the input position along any other axis.
    Orthogonal = None

    # If True, then the distance between two input points is the
    # same as the distance between the transformed points.
    NonScaling = None

    # Scale factors are applied equally to all axes.
    Isometric = None

    # Multiplying the input by a scale factor causes the output to be multiplied by
    # the same factor:  Tr(s * x) = s * Tr(x)
    Homogeneous = None

    # The transform of two added vectors is the same as the sum of the individually
    # transformed vectors:  T(a + b) = T(a) + T(b)
    Additive = None

    # If True, then input and output dimensionality must be the same.
    Equidimensional = None

    # Specification of parameters for this transform. Each subclass should
    # define its own parameter_spec list.
    parameter_spec = []

    def __init__(
        self,
        dims: Dims = None,
        dynamic: bool = False,
        from_cs: StrOrNone = None,
        to_cs: StrOrNone = None,
        cs_graph: StrOrNone = None,
        **kwargs,
    ):
        self._state = {"dynamic": dynamic, "dims": self._validate_dims(dims, **kwargs)}
        self._init_with_no_state()

        # optional coordinate system tracking
        if from_cs is not None:
            self.set_systems(from_cs, to_cs, cs_graph)
        self.set_params(**kwargs)

    def _init_with_no_state(self):
        self._change_callbacks = CallbackRegistry()
        self._inverse = None
        self._systems = (None, None)

    def _validate_dims(self, dims: None | int | tuple[int, int], **kwargs):
        """Determine dimensionality from parameters or *dims* argument.

        If *dims* has a value, then it must be a tuple (in, out) or int and it must agree with the
        length of all provided parameters. If *dims* is not provided, then it is determined from the
        length of the parameters that use dims, which must all be in agreement.
        """
        params = {
            spec.name: kwargs[spec.name]
            for spec in self.parameter_spec
            if spec.uses_dims and spec.name in kwargs
        }
        inferred_dims = {k: (len(v), len(v)) for k, v in params.items() if v is not None}
        if dims is not None:
            if np.isscalar(dims):
                dims = (dims, dims)
            if not isinstance(dims, tuple) or len(dims) != 2:
                raise ValueError(f"dims must be an int or a tuple of two ints, not {dims}")
            for k, v in inferred_dims.items():
                assert v == dims, f"Length of {k} ({len(dims[k])}) does not match dims {dims}"
            if self.Equidimensional and dims[0] != dims[1]:
                raise ValueError("Equidimensional transforms must have equal input and output dims")
            return dims
        elif len(inferred_dims) == 0:
            raise ValueError("dims must be specified if no parameters use dims")

        if len(inferred_dims) == 0:
            msg = f"Could not determine dimensionality of transform. "
            param_names = ' '.join(list(params.keys()))
            if len(params) == 1:
                msg += f"Specify dims or {param_names}."
            else:
                msg += f"Specify dims or at least one of {param_names}."
            raise ValueError(msg)

        keys = list(inferred_dims.keys())
        dims = inferred_dims[keys[0]]
        for k in keys[1:]:
            if inferred_dims[k] != dims:
                raise ValueError(
                    f"Could not determine dimensionality of transform: length of {k}"
                    f"({inferred_dims[k]}) does not match length of {keys[0]} ({dims})"
                )
        return dims

    @property
    def dims(self):
        """Tuple holding the (input, output) dimensions for this transform."""
        return self._state['dims']

    @property
    def dynamic(self):
        """Boolean flag that indicates whether this transform is expected to
        change frequently.

        Transforms that are flagged as dynamic will not be collapsed in
        ``CompositeTransform.simplified``. This allows changes to the transform
        to propagate through the chain without requiring the chain to be
        re-simplified.
        """
        return self._state['dynamic']

    @dynamic.setter
    def dynamic(self, d):
        self._state['dynamic'] = d  # no spec, so not through set_params

    @property
    def systems(self) -> tuple[CoordinateSystem | None, CoordinateSystem | None]:
        """The CoordinateSystem instances mapped from and to by this transform."""
        return self._systems

    def set_systems(self, from_cs, to_cs, cs_graph=None):
        assert (from_cs is None) == (
            to_cs is None
        ), "from_cs and to_cs must both be None or both be coordinate systems"
        if from_cs is not None:
            if cs_graph is None and isinstance(from_cs, CoordinateSystem):
                cs_graph = from_cs.graph
            else:
                cs_graph = CoordinateSystemGraph.get_graph(cs_graph)
            cs_graph.add_transform(self, from_cs=from_cs, to_cs=to_cs)

    def map(self, obj: Mappable):
        """
        Return *obj* mapped through the forward transformation.

        Parameters
        ----------
            obj : tuple (x,y) or (x,y,z)
                  array with shape (..., 2) or (..., 3)
        """
        return self._prepare_and_map(obj)

    def _map(self, arr):
        """Map a 2D array (n_pts, n_dims) through this transform.

        This method must be redefined in sublcasses.
        """
        raise NotImplementedError

    def imap(self, obj: Mappable):
        """
        Return *obj* mapped through the inverse transformation.

        Parameters
        ----------
            obj : tuple (x,y) or (x,y,z)
                  array with shape (..., 2) or (..., 3)
        """
        return self.inverse.map(obj)

    def _imap(self, arr):
        """Map a 2D array (n_pts, n_dims) through the inverse of this transform.

        This method may be redefined in sublcasses.
        """
        raise NotImplementedError

    def _prepare_and_map(self, obj: Mappable):
        """
        Convert a mappable object to a 2D numpy array, pass it through this Transform's _map method,
        then convert and return the result.

        The Transform's _map method will be called with a 2D array
        of shape (N, M), where N is the number of points and M is the number of dimensions.
        Accepts lists, tuples, and arrays of any dimensionality and flattens extra dimensions into N.
        After mapping, any flattened axes are re-expanded to match the original input shape.

        For list, tuple, and array inputs, the return value is a numpy array of the same shape as
        the input, with the exception that the last dimension is determined only by the return value.

        Alternatively, any class may determine how to map itself by defining a _coorx_transform()
        method that accepts this transform as an argument.
        """
        if hasattr(obj, '_coorx_transform'):
            # let the object decide how to apply this transform
            return obj._coorx_transform(tr=self)
        elif isinstance(obj, (tuple, list, np.ndarray)):
            arr_2d, original_shape = self._prepare_arg_for_mapping(obj)
            if self.dims[0] not in (None, arr_2d.shape[1]):
                raise TypeError(
                    f"Transform maps from {self.dims[0]}D, but data to be mapped is {arr_2d.shape[1]}D"
                )
            ret = self._map(arr_2d)
            assert ret.ndim == 2
            assert self.dims[1] in (
                None,
                ret.shape[1],
            ), f"Transform maps to {self.dims[1]}D, but mapping generated {ret.shape[1]}D"
            return self._restore_shape(ret, original_shape)
        elif hasattr(obj, '__len__') and len(obj) == self.dims[0]:
            try:
                # fudge for single points passed as list/tuple-like objects
                arr = np.asarray(obj)
                ret = self._map(arr)
                return type(obj)(*ret)
            except Exception as e:
                raise TypeError(f"Cannot directly map object through Transforms: {obj}") from e
        else:
            raise TypeError(f"Cannot use argument for mapping: {obj}")

    @staticmethod
    def _prepare_arg_for_mapping(arg):
        """Convert arg to a 2D numpy array.

        If the argument ndim is > 2, then all dimensions except the last are flattened.

        Return the reshaped array and a tuple containing the original shape.
        """
        arg = np.asarray(arg)
        original_shape = arg.shape
        arg = arg.reshape(int(np.prod(arg.shape[:-1])), arg.shape[-1])
        return arg, original_shape

    @staticmethod
    def _restore_shape(arg, shape):
        """Return an array with shape determined by shape[:-1] + (arg.shape[-1],)"""
        if arg is None:
            return arg
        return arg.reshape(shape[:-1] + (arg.shape[-1],))

    @property
    def inverse(self):
        """The inverse of this transform."""
        if self._inverse is None:
            self._inverse = InverseTransform(self)
        return self._inverse

    @property
    def params(self):
        """Return a dict of parameters specifying this transform."""
        return deepcopy(self._state)

    def set_params(self, **kwds):
        """Set parameters specifying this transform.

        Parameter names must be the same as the keys in self._state.
        """
        any_changed = False
        for name in kwds:
            validator = self.get_validator(name)
            value, this_changed = validator.validate(kwds, self._state)

            if this_changed:
                any_changed = True
                self._state[name] = value
        if any_changed:
            self._update()

    @classmethod
    def get_validator(cls, param_name: str):
        if param_name not in cls.param_spec_dict():
            raise NameError(f"Transform {cls.__name__} has no parameter '{param_name}'")
        return cls.param_spec_dict()[param_name]

    @classmethod
    def param_spec_dict(cls):
        """Return a dict mapping parameter names to Parameter instances for this class."""
        if not hasattr(cls, '_param_spec_dict'):
            cls._param_spec_dict = {p.name: p for p in cls.parameter_spec}
        return cls._param_spec_dict

    def as_affine(self):
        """Return an equivalent affine transform if possible."""
        raise NotImplementedError()

    @property
    def full_matrix(self) -> np.ndarray:
        """
        Return the full transformation matrix for this transform, if possible.

        Modifying the returned array has no effect on the transform instance that generated it.
        """
        return self.as_affine().full_matrix

    def as_vispy(self):
        """Return a VisPy transform that is equivalent to this transform, if possible."""
        if self.dims != (3, 3):
            raise NotImplementedError("as_vispy is only implemented for 3D transforms")
        from vispy.visuals.transforms import MatrixTransform

        # a functional default if nothing else is implemented
        return MatrixTransform(self.full_matrix.T)

    def as_pyqtgraph(self):
        """Return a PyQtGraph transform that is equivalent to this transform, if possible."""
        from pyqtgraph import SRTTransform3D
        from pyqtgraph.Qt import QtGui

        # a functional default if nothing else is implemented
        return SRTTransform3D(QtGui.QMatrix4x4(self.full_matrix.reshape(-1)))

    def to_qmatrix4x4(self, QtGui=None):
        """Return a QMatrix4x4 that is equivalent to this transform."""
        from .qt import import_qt_gui

        return import_qt_gui().QMatrix4x4(self.full_matrix.reshape(-1))

    def add_change_callback(self, cb: Callable[[ChangeEvent], None], keep_reference: bool = False):
        """Add a callback that will be called whenever parameters of this transform change. If
        keep_reference is False, the callback will be held with a weak reference. This allows the
        object and its context to be garbage collected. Typically, keep_reference should be False
        whenever using a bound method, and True when using a standalone function whose side effects
        are desired even if no other references to that function exist.
        """
        self._change_callbacks.add(cb, keep_reference)

    def remove_change_callback(self, cb):
        """Remove a change callback."""
        self._change_callbacks.remove(cb)

    def _update(self, source_event=None):
        """
        Called to inform any listeners that this transform has changed.
        """
        event = ChangeEvent(transform=self, source_event=source_event)

        # Get a snapshot of callbacks to invoke (under lock to prevent races)
        for cb in self._change_callbacks:
            try:
                cb(event)
            except Exception:
                print(f"Error invoking callback {cb}")
                raise

    def __mul__(self, tr):
        """
        Transform multiplication returns a new transform that is equivalent to
        the two operands performed in series.

        By default, multiplying two Transforms `A * B` will return
        ChainTransform([A, B]). Subclasses may redefine this operation to
        return more optimized results.

        To ensure that both operands have a chance to simplify the operation,
        all subclasses should follow the same procedure. For `A * B`:

        0. The inner systems of A and B must be the same (all
           implementations must call validate_transform_for_mul to ensure this).
        1. A.__mul__(B) attempts to generate an optimized transform product.
        2. If that fails, it must:

               * return super(A).__mul__(B) OR
               * return NotImplemented if the superclass would return an
                 invalid result.

        3. When Transform.__mul__(A, B) is called, it returns
           NotImplemented, which causes B.__rmul__(A) to be invoked.
        4. B.__rmul__(A) attempts to generate an optimized transform product.
        5. If that fails, it must:

               * return super(B).__rmul__(A) OR
               * return ChainTransform([B, A]) if the superclass would return
                 an invalid result.

        6. When Transform.__rmul__(B, A) is called, ChainTransform([A, B])
           is returned.
        """
        # switch to __rmul__ attempts.
        # Don't use the "return NotImplemented" trick, because that won't work if
        # self and tr are of the same type.
        return tr.__rmul__(self)

    def __rmul__(self, tr):
        """tr * self"""
        tr.validate_transform_for_mul(self)
        return CompositeTransform([self, tr])

    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):x}>"

    def __getstate__(self):
        state = self._state.copy()
        if self.systems[0] is None:
            state['systems'] = (None, None, None)
        else:
            state['systems'] = (
                self.systems[0].name,
                self.systems[1].name,
                self.systems[0].graph.name,
            )
        return state

    def __setstate__(self, state):
        self._init_with_no_state()
        from_cs, to_cs, *graph = state.pop('systems', (None, None, None))
        if len(graph) == 0:
            graph = None
        else:
            graph = graph[0]

        self._state = {k: v for k, v in state.items() if k not in self.param_spec_dict()}
        self.set_params(**{k: v for k, v in state.items() if k in self.param_spec_dict()})
        from .util import DependentTransformError

        with contextlib.suppress(DependentTransformError):
            self._systems = (None, None)
            self.set_systems(from_cs, to_cs, graph)

    def copy(self, from_cs=None, to_cs=None):
        """Return a copy of this transform."""
        state = self.save_state()
        if from_cs is not None or to_cs is not None:
            from_cs = from_cs or self.systems[0]
            to_cs = to_cs or self.systems[1]
            graph = None
            if from_cs is not None and not isinstance(from_cs, str):
                graph = from_cs.graph.name
            if graph is None and to_cs is not None and not isinstance(to_cs, str):
                graph = to_cs.graph.name
            state['systems'] = (from_cs, to_cs)
            state['graph'] = graph
        return self.from_state(state)

    def save_state(self):
        """Return serializable parameters that specify this transform. Distinct from __getstate__
        for no good long-term reason, but we need to support yaml serialization somehow for now."""

        def to_simple(o):
            if isinstance(o, Transform):
                return o.save_state()
            elif np.isscalar(o) or o is None:
                return o
            else:
                return [to_simple(e) for e in np.asarray(o).tolist()]

        return {
            'type': type(self).__name__,
            **{k: to_simple(v) for k, v in self.__getstate__().items()},
        }

    @classmethod
    def from_state(cls, state):
        """Return a Transform instance created from saved state."""
        tr = cls.__new__(cls)
        tr.__setstate__(state)
        return tr

    def __eq__(self, tr):
        if type(self) is not type(tr):
            return False
        if self._state.keys() != tr._state.keys():
            return False
        for k in self._state:
            v1 = self._state[k]
            v2 = tr._state[k]
            if np.isscalar(v1):
                if not (np.isscalar(v2) and v1 == v2):
                    return False
            else:
                if not np.all(np.asarray(v1) == np.asarray(v2)):
                    return False
        return True

    def validate_transform_for_mul(self, tr):
        if tr.systems[1] != self.systems[0]:
            raise TypeError(
                f"Cannot multiply transforms with different inner coordinate systems: {self.systems[0]} != {tr.systems[1]}"
            )


class InverseTransform(Transform):
    def __init__(self, transform):
        super().__init__(inverse=transform)

    def _validate_dims(self, dims, **kwargs):
        # dims are determined by the inverse transform
        return None

    def set_systems(self, from_cs, to_cs, cs_graph=None):
        from .util import DependentTransformError

        raise DependentTransformError("Cannot set systems on a dependent inverse transform")

    def as_affine(self):
        affine = self._state["inverse"].as_affine()
        return type(affine)(
            matrix=affine.inv_matrix,
            offset=affine.inv_matrix @ affine.inv_offset,
            from_cs=self.systems[0],
            to_cs=self.systems[1],
        )

    def copy(self, from_cs=None, to_cs=None):
        return self._state['inverse'].copy(from_cs=to_cs, to_cs=from_cs).inverse

    def set_params(self, inverse):
        if isinstance(inverse, Transform):
            self._state['inverse'] = inverse
        else:
            from . import create_transform

            self._state['inverse'] = create_transform(**inverse)
        self._map = self._state['inverse']._imap
        self._imap = self._state['inverse']._map
        self._update()

    @property
    def dims(self):
        return self._state['inverse'].dims[::-1]

    @property
    def systems(self):
        return self._state['inverse'].systems[::-1]

    @property
    def Linear(self):
        return self._state['inverse'].Linear

    @property
    def Orthogonal(self):
        return self._state['inverse'].Orthogonal

    @property
    def NonScaling(self):
        return self._state['inverse'].NonScaling

    @property
    def Isometric(self):
        return self._state['inverse'].Isometric

    def __repr__(self):
        return f"<Inverse of {self._state['inverse']!r}>"


# import here to avoid import cycle; needed for Transform.__mul__.
from .composite import CompositeTransform
