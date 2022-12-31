# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.

"""
API Issues to work out:

  - MatrixTransform and STTransform both have 'scale' and 'translate'
    attributes, but they are used in very different ways. It would be nice
    to keep this consistent, but how?

  - Need a transform.map_rect function that returns the bounding rectangle of
    a rect after transformation. Non-linear transforms might need to work
    harder at this, but we can provide a default implementation that
    works by mapping a selection of points across a grid within the original
    rect.
"""

import numpy as np
from .systems import CoordinateSystemGraph
from .types import Dims, StrOrNone, Mappable


class BaseTransform(object):
    """
    BaseTransform is a base class that defines a pair of complementary
    coordinate mapping functions in both python and GLSL.

    All BaseTransform subclasses define map() and imap() methods that map
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

    def __init__(self, dims:Dims=None, from_cs:StrOrNone=None, to_cs:StrOrNone=None, cs_graph:StrOrNone=None):
        if dims is None or np.isscalar(dims):
            dims = (dims, dims)
        if not isinstance(dims, tuple) or len(dims) != 2:
            raise TypeError("dims must be length-2 tuple")
        self._dims = dims
        self._inverse = None
        self._dynamic = False
        self._change_callbacks = []
        self._systems = (None, None)

        # optional coordinate system tracking
        self._cs_graph = None
        assert (from_cs is None) == (to_cs is None), "from_cs and to_cs must both be None or both be str"
        if from_cs is not None:
            self._cs_graph = CoordinateSystemGraph.get_graph(cs_graph)
            self._cs_graph.add_transform(self, from_cs=from_cs, to_cs=to_cs)

    @property
    def dims(self):
        """Tuple holding the (input, output) dimensions for this transform.
        """
        return self._dims

    @property
    def systems(self):
        """The CoordinateSystem instances mapped from and to by this transform.
        """
        return self._systems

    def map(self, obj:Mappable):
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

    def imap(self, obj:Mappable):
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

    def _prepare_and_map(self, obj:Mappable):
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
                raise TypeError(f"Transform maps from {self.dims[0]}D, but data to be mapped is {arr_2d.shape[1]}D")
            ret = self._map(arr_2d)
            assert ret.ndim == 2
            assert self.dims[1] in (None, ret.shape[1]), f"Transform maps to {self.dims[1]}D, but mapping generated {ret.shape[1]}D"
            return self._restore_shape(ret, original_shape)
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
        arg = arg.reshape(int(np.product(arg.shape[:-1])), arg.shape[-1])
        return arg, original_shape

    @staticmethod
    def _restore_shape(arg, shape):
        """Return an array with shape determined by shape[:-1] + (arg.shape[-1],)
        """
        if arg is None:
            return arg
        return arg.reshape(shape[:-1] + (arg.shape[-1],))

    @property
    def inverse(self):
        """ The inverse of this transform. 
        """
        if self._inverse is None:
            self._inverse = InverseTransform(self)
        return self._inverse

    @property
    def dynamic(self):
        """Boolean flag that indicates whether this transform is expected to 
        change frequently.
        
        Transforms that are flagged as dynamic will not be collapsed in 
        ``ChainTransform.simplified``. This allows changes to the transform
        to propagate through the chain without requiring the chain to be
        re-simplified.
        """
        return self._dynamic

    @dynamic.setter
    def dynamic(self, d):
        self._dynamic = d

    @property
    def params(self):
        """Return a dict of parameters specifying this transform.
        """
        raise NotImplementedError()

    def set_params(self, **kwds):
        """Set parameters specifying this transform.
        
        Parameter names must be the same as the keys in self.params.
        """
        raise NotImplementedError()

    def save_state(self):
        return self.__getstate__()

    def add_change_callback(self, cb):
        self._change_callbacks.append(cb)
        
    def remove_change_callback(self, cb):
        self._change_callbacks.remove(cb)

    def update(self, *args):
        """
        Called to inform any listeners that this transform has changed.
        """
        for cb in self._change_callbacks:
            cb(*args)

    def __mul__(self, tr):
        """
        Transform multiplication returns a new transform that is equivalent to
        the two operands performed in series.

        By default, multiplying two Transforms `A * B` will return
        ChainTransform([A, B]). Subclasses may redefine this operation to
        return more optimized results.

        To ensure that both operands have a chance to simplify the operation,
        all subclasses should follow the same procedure. For `A * B`:

        1. A.__mul__(B) attempts to generate an optimized transform product.
        2. If that fails, it must:

               * return super(A).__mul__(B) OR
               * return NotImplemented if the superclass would return an
                 invalid result.

        3. When BaseTransform.__mul__(A, B) is called, it returns 
           NotImplemented, which causes B.__rmul__(A) to be invoked.
        4. B.__rmul__(A) attempts to generate an optimized transform product.
        5. If that fails, it must:

               * return super(B).__rmul__(A) OR
               * return ChainTransform([B, A]) if the superclass would return
                 an invalid result.

        6. When BaseTransform.__rmul__(B, A) is called, ChainTransform([A, B])
           is returned.
        """
        # switch to __rmul__ attempts.
        # Don't use the "return NotImplemted" trick, because that won't work if
        # self and tr are of the same type.
        return tr.__rmul__(self)

    def __rmul__(self, tr):
        return CompositeTransform([tr, self])

    def __repr__(self):
        return "<%s at 0x%x>" % (self.__class__.__name__, id(self))

    def __getstate__(self):
        """Return serializable parameters that specify this transform.
        """
        return {'type': self.__class__.__name__, 'params': self.params}

    def __setstate__(self, state):
        """Set the state of this transform from parameters generated by 
        __getstate__().
        """
        if state['type'] != self.__class__.__name__:
            raise TypeError("%s cannot use state saved from %s" % 
                            (self.__class__.__name__, state['type']))
        self.set_params(**state['params'])


class InverseTransform(BaseTransform):
    def __init__(self, transform):
        BaseTransform.__init__(self)
        self._inverse = transform
        self._map = transform._imap
        self._imap = transform._map
    
    @property
    def dims(self):
        return self._inverse.dims[::-1]

    @property
    def systems(self):
        return self._inverse.systems[::-1]

    @property
    def Linear(self):
        return self._inverse.Linear

    @property
    def Orthogonal(self):
        return self._inverse.Orthogonal

    @property
    def NonScaling(self):
        return self._inverse.NonScaling

    @property
    def Isometric(self):
        return self._inverse.Isometric
    
    def __repr__(self):
        return ("<Inverse of %r>" % repr(self._inverse))
        

# import here to avoid import cycle; needed for BaseTransform.__mul__.
from .composite import CompositeTransform
