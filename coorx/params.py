import numpy as np


class Parameter:
    def __init__(self, name):
        self.name = name
        self.uses_dims = False

    def validate(self, new_params, current_state):
        value = new_params.get(self.name)
        old_value = current_state[self.name]
        return value, value == old_value


class TupleParameter(Parameter):
    def __init__(self, name, length):
        super().__init__(name)
        self.length = length
        self.uses_dims = isinstance(length, str)

    def validate(self, new_params, current_state):
        value = new_params.get(self.name)
        length = self.length
        if isinstance(length, str):
            length = current_state["dims"][0] if length == "dims0" else current_state["dims"][1]
        if not (isinstance(value, (tuple, list)) and len(value) == length):
            raise ValueError(f"Parameter '{self.name}' must be a tuple/list of length {length}")
        value = tuple(value)
        old_value = current_state[self.name]
        changed = value != old_value
        return value, changed


class ArrayParameter(Parameter):
    def __init__(self, name, dtype=float, shape=None, default=None):
        super().__init__(name)
        self.dtype = dtype
        self.shape = shape
        self.uses_dims = shape is not None and any(isinstance(dim, str) for dim in shape)
        self.default = default

    def validate(self, new_params, current_state):
        value = new_params.get(self.name)
        shape = self.shape
        if shape is not None:
            for i, dim in enumerate(shape):
                if dim == "dims0":
                    dim = current_state["dims"][0]
                elif dim == "dims1":
                    dim = current_state["dims"][1]
                shape = shape[:i] + (dim,) + shape[i + 1 :]
        if value is None:
            if self.default is not None:
                value = np.ones(shape, dtype=self.dtype) * self.default
            else:
                raise ValueError(f"Parameter '{self.name}' cannot be None")
        else:
            value = np.asarray(value, dtype=self.dtype)
            if shape is not None and value.shape != shape:
                raise ValueError(f"Parameter '{self.name}' must have shape {shape}, got {value.shape}")
        return value, not np.array_equal(value, current_state.get(self.name))


class TransformParameter(Parameter):
    def __init__(self, name, shape=None):
        super().__init__(name)
        self.shape = shape

    def validate(self, new_params, current_state):
        value = new_params.get(self.name)
        from .base_transform import Transform

        if not isinstance(value, Transform):
            raise TypeError(f"Parameter '{self.name}' must be a Transform instance")
        return super().validate(new_params, current_state)
