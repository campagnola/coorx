import numpy as np


class Parameter:
    def __init__(self, name):
        self.name = name

    def validate(self, value, old_value, dims):
        """Return the validated value for this parameter and whether it has changed.
        Raise an exception if the value is invalid.
        """
        return value, value != old_value

    def check_dims(self, unvalidated_value, dims) -> bool | None:
        """Return True if using *unvalidated_value* for this parameter is compatible with *dims*.
        Return None if the parameter does not depend on dims.
        """
        return None

    def infer_dims(self, unvalidated_value) -> tuple[float | None, float | None]:
        """Return the dims that would be required to use *value* for this parameter.
        Return None for either dim if the parameter does not constrain it.
        """
        return None, None


class FloatParameter(Parameter):
    def __init__(self, name, default=None):
        super().__init__(name)
        self.default = default

    def validate(self, value, old_value, dims):
        if value is None and self.default is not None:
            value = self.default
        try:
            value = float(value)
        except Exception as e:
            raise ValueError(f"Parameter '{self.name}' must be a float") from e
        return value, value != old_value


class TupleParameter(Parameter):
    def __init__(self, name, length=None, dtype=None, default=None):
        super().__init__(name)
        if length is not None and length not in ('dims0', 'dims1') and not isinstance(length, int):
            raise ValueError("length must be None, integer, or 'dims0'/'dims1'")
        self.length = length
        self.dtype = dtype
        self.default = default

    def validate(self, value, old_value, dims):
        # make sure value is tuple-like
        if value is None and self.default is not None:
            value = self.default(dims)
        try:
            value = tuple(value)
        except Exception as e:
            raise TypeError(f"Parameter '{self.name}' must be tuple-like") from e

        # check length
        length = self.length
        if length in ('dims0', 'dims1'):
            length = {'dims0': dims[0], 'dims1': dims[1]}[length]
        if length is not None and len(value) != length:
            raise ValueError(f"Parameter '{self.name}' must have length {length}")

        # convert elements to dtype
        if self.dtype is not None:
            try:
                value = tuple(self.dtype(v) for v in value)
            except Exception as e:
                raise TypeError(
                    f"Elements of parameter '{self.name}' must be of type {self.dtype.__name__}"
                ) from e

        return value, value != old_value

    def check_dims(self, unvalidated_value, dims):
        if unvalidated_value is None:
            return None
        if isinstance(self.length, str):
            expected_length = dims[0] if self.length == "dims0" else dims[1]
            return len(unvalidated_value) == expected_length
        return None

    def infer_dims(self, unvalidated_value):
        if unvalidated_value is None:
            return None, None
        if isinstance(self.length, str):
            return (
                (len(unvalidated_value), None)
                if self.length == "dims0"
                else (None, len(unvalidated_value))
            )
        return None, None


class ArrayParameter(Parameter):
    def __init__(self, name, dtype=float, shape=None, default=None):
        super().__init__(name)
        if not isinstance(shape, tuple) and shape is not None:
            raise ValueError("shape must be a tuple or None")
        if shape is not None:
            for x in shape:
                if not (x is None or isinstance(x, int) or x in ("dims0", "dims1")):
                    raise ValueError("shape elements must be None, integer, or 'dims0'/'dims1'")
        self.dtype = dtype
        self.shape = shape
        self.default = default

    def validate(self, value, old_value, dims):
        # determine shape constraints
        shape = self.shape
        if shape is not None:
            str_dims = {"dims0": dims[0], "dims1": dims[1]}
            shape = tuple([str_dims.get(dim, dim) for dim in shape])

        # handle default
        if value is None:
            if self.default is None:
                raise ValueError(f"Parameter '{self.name}' cannot be None")
            if shape is None or None in shape:
                raise ValueError(
                    f"Parameter '{self.name}' has undefined shape, cannot use callable default"
                )
            if callable(self.default):
                value = self.default(shape)
            elif self.default is not None:
                value = np.ones(shape, dtype=self.dtype) * self.default

        # check array / dtype
        try:
            value = np.asarray(value, dtype=self.dtype)
        except Exception as e:
            dtype_err = ' of type ' + self.dtype.__name__ if self.dtype is not None else ''
            raise TypeError(f"Parameter '{self.name}' must be an array{dtype_err}") from e

        # check shape
        if shape is not None:
            assert (
                len(shape) == value.ndim
            ), f"Parameter '{self.name}' must have ndim {len(shape)}, got {value.ndim}"
            for i, dim in enumerate(shape):
                if dim is not None and value.shape[i] != dim:
                    raise ValueError(
                        f"Parameter '{self.name}' must have shape[{i}]=={shape[i]}, got {value.shape[i]}"
                    )

        return value, not np.array_equal(value, old_value)

    def check_dims(self, unvalidated_value, dims):
        if unvalidated_value is None:
            return None
        unvalidated_value = np.asarray(unvalidated_value)
        if self.shape is not None:
            str_dims = {"dims0": dims[0], "dims1": dims[1]}
            for i, dim in enumerate(self.shape):
                expected_dim = str_dims.get(dim, dim)
                if expected_dim is not None and unvalidated_value.shape[i] != expected_dim:
                    return False
        return None

    def infer_dims(self, unvalidated_value):
        dims0 = None
        dims1 = None
        if unvalidated_value is None:
            return dims0, dims1
        unvalidated_value = np.asarray(unvalidated_value)
        if self.shape is not None:
            for dim in self.shape:
                if dim == "dims0":
                    dims0 = unvalidated_value.shape[self.shape.index(dim)]
                elif dim == "dims1":
                    dims1 = unvalidated_value.shape[self.shape.index(dim)]
        return dims0, dims1


class TransformParameter(Parameter):
    def __init__(self, name, dims=None, default=None):
        super().__init__(name)
        self.dims = dims
        self.default = default

    def validate(self, value, old_value, dims):
        from . import Transform, create_transform

        if isinstance(value, dict):
            try:
                value = create_transform(**value)
            except Exception as e:
                raise ValueError(
                    f"Transform for parameter {self.name} could not be created from {value}"
                ) from e
        if value is None and callable(self.default):
            value = self.default()
        if not isinstance(value, Transform):
            raise TypeError(
                f"Parameter '{self.name}' must be a Transform instance, not {type(value).__name__}"
            )
        if self.dims is not None and value.dims != self.dims:
            raise ValueError(
                f"Parameter '{self.name}' must have dims {self.dims}, got {value.dims}"
            )
        changed = value != old_value
        return value, changed


class TransformListParameter(Parameter):
    def __init__(self, name):
        super().__init__(name)

    def validate(self, new_value, old_value, dims):
        raise NotImplementedError("this isn't being used yet")
        from . import create_transform, Transform
        if new_value is None:
            new_value = []
        new_value = [t if isinstance(t, Transform) else create_transform(**t) for t in new_value]
        changed = new_value != old_value
        return new_value, changed
