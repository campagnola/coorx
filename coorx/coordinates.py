from __future__ import annotations

import numpy as np

from .systems import CoordinateSystem, CoordinateSystemGraph, get_coordinate_system
from ._types import StrOrNone, CoordSysOrStr


class PointArray:
    """Represents an N-dimensional array of points.

    Although the array is instantiated with the coordinates of these points in a particular
    coordinate system, one may request the coordinates mapped to any other system.

    Parameters
    ----------
    coordinates : array-like
        Array or list of coordinates. These may be numerical or Point instances.
    system : str | System | None
        The coordinate system to which th points in *coordinates* belong.
    graph : str | CoordinateSystemGraph
        If *system* is a string, then the coordinate system is looked up from this graph.
    """

    def __init__(self, coordinates, system: CoordSysOrStr = None, graph: StrOrNone = None):
        coord_arr, source_system = self._interpret_input(coordinates)

        if not np.issubdtype(coord_arr.dtype, np.number):
            raise TypeError(f"Coordinates must be numerical or Point instances, not {coord_arr.dtype}")
        self._coordinates = coord_arr

        # get the requested coordinate system
        self.system = None
        if system is not None:
            self.set_system(system, graph)

        # if the input came with a coordinate system, use that and
        # verify that it does not conflict with the previously specified system
        if source_system is not None:
            if self.system not in (None, source_system):
                raise TypeError(f"System {system} does not match source coordinates {source_system}")
            self.system = source_system

        if self.system is not None and self.shape[-1] != self.system.ndim:
            raise TypeError(f"System ndim is {self.system.ndim}, but coordinate data is {self.shape[-1]}D")

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def shape(self):
        return self.coordinates.shape

    @property
    def ndim(self):
        return self.coordinates.ndim

    @property
    def dtype(self):
        return self.coordinates.dtype

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]

    def __iter__(self):
        yield from self.coordinates

    def __add__(self, b):
        if isinstance(b, (Vector, VectorArray)):
            self._check_vector_operand(b)
            new_coords = self.coordinates + b.displacement
            if isinstance(self, Point) and isinstance(b, Vector):
                return Point(new_coords, system=self.system)
            else:
                return PointArray(new_coords, system=self.system)
        else:
            self._check_point_operand(b)
            # This case should likely not return raw coordinates, but maybe another PointArray?
            # For now, keeping original behavior for PointArray + PointArray.
            # Consider if PointArray + PointArray should be allowed or raise TypeError.
            return self.coordinates + b.coordinates

    def __sub__(self, b: PointArray | VectorArray) -> VectorArray | PointArray:
        if isinstance(b, PointArray):
            self._check_point_operand(b)
            return VectorArray(b, self)
        elif isinstance(b, VectorArray):
            self._check_vector_operand(b)
            new_coords = self.coordinates - b.displacement
            return PointArray(new_coords, system=self.system)
        raise TypeError("Unsupported operand type")

    def mapped_through(self, cs_list) -> PointArray:
        chain = self.system.graph.transform_chain([self.system] + cs_list)
        return chain.map(self)

    def mapped_to(self, system) -> "PointArray":
        path = self.system.graph.transform_path(self.system, system)
        chain = self.system.graph.transform_chain(path)
        return chain.map(self)

    def set_system(self, system, graph=None):
        self.system = get_coordinate_system(system, graph=graph, ndim=self.shape[-1], create=True)

    def _coorx_transform(self, tr):
        if tr.systems[0] is not self.system:
            raise TypeError(
                f"The transform {tr} maps from system {tr.systems[0]}, but this PointArray is defined in {self.system}"
            )
        # Map the raw coordinates
        mapped_coords = tr.map(self.coordinates)
        output_system = tr.systems[1]
        if isinstance(self, Point):
            return Point(mapped_coords, system=output_system)
        else:
            return PointArray(mapped_coords, system=output_system)

    def __array__(self, dtype=None):
        # Ensure subclasses like Point also work correctly with np.asarray
        coords = self.coordinates
        if dtype is None:
            return coords
        else:
            return coords.astype(dtype, copy=False)

    def __repr__(self):
        return f"<{type(self).__name__} {self.shape} in {self.system.name}>"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["system"] = None if self.system is None else (self.system.name, self.system.graph.name)
        return state

    def __setstate__(self, state):
        sys, graph = state.pop("system")
        self.__dict__.update(state)
        self.set_system(sys, graph)

    def __eq__(self, b):
        if type(b) is not type(self):
            return False
        if self.coordinates.shape != b.coordinates.shape:
            return False
        if not np.all(self.coordinates == b.coordinates):
            return False
        if self.system is not b.system:
            return False
        return True

    def _interpret_input(self, coordinates):
        """Interpret initial coordinates argument, return a numerical array of coordinates
        and a coordinate system if it was present in the input.
        """
        # convert coordinates to array
        coord_arr = np.asarray(coordinates)
        if coord_arr.size == 0:
            raise ValueError("Coordinates cannot be empty.")

        # if input is an object array, it might need extra help
        if coord_arr.dtype is np.dtype(object):
            first = coord_arr.ravel()[0]
            if isinstance(first, Point):
                # this will cause Points to be unpacked into numerical array
                coord_arr = np.array(coord_arr.tolist())
            else:
                raise TypeError(f"Object array with item type {type(first)} not supported as input.")
        else:
            first = coordinates
            for _ in range(coord_arr.ndim - 1):
                first = first[0]

        if isinstance(first, Point):
            source_system = first.system
        # If input was already PointArray, use its system
        elif isinstance(coordinates, PointArray):
            source_system = coordinates.system
        else:
            source_system = None

        return coord_arr, source_system

    def _check_point_operand(self, a):
        if not isinstance(a, PointArray):
            raise TypeError(f"Operand must be a PointArray or Point (received {type(a)})")
        if a.system is not self.system:
            raise ValueError(f"Operand system '{a.system}' does not match this PointArray's system '{self.system}'")

    def _check_vector_operand(self, a):
        if not isinstance(a, (Vector, VectorArray)):
            raise TypeError(f"Operand must be a VectorArray or Vector (received {type(a)})")
        if a.system is not self.system:
            raise ValueError(f"Operand system '{a.system}' does not match this PointArray's system '{self.system}'")


class Point(PointArray):
    """Represents a single point in space; one may request the coordinates of this point
    in any coordinate system.
    """

    def __init__(self, coordinates, system: CoordSysOrStr = None, graph: StrOrNone = None):
        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 1:
            raise TypeError("Point coordinates must be 1D")
        super().__init__(coordinates[np.newaxis, :], system, graph)

    @property
    def coordinates(self):
        return self._coordinates[0]

    def __sub__(self, b: PointArray | VectorArray) -> VectorArray | PointArray:
        if isinstance(b, Point):
            self._check_point_operand(b)
            return Vector(b, self)
        elif isinstance(b, Vector):
            self._check_vector_operand(b)
            new_coords = self.coordinates - b.displacement
            return Point(new_coords, system=self.system)
        return super().__sub__(b)

    def __repr__(self):
        # Ensure coordinates are displayed correctly even if _coordinates is 2D
        coords_tuple = tuple(self.coordinates)
        return f"<{type(self).__name__} {coords_tuple} in {self.system.name}>"


class VectorArray:
    """
    Represents an N-dimensional array of vectors (displacements).

    A vector is defined by two endpoints, p1 (start) and p2 (end),
    within the same coordinate system. The displacement is p2 - p1.

    Initialization only accepts two PointArray/Point instances as endpoints.
    """

    def __init__(self, p1: PointArray | Point, p2: PointArray | Point):
        if not isinstance(p1, (PointArray, Point)) or not isinstance(p2, (PointArray, Point)):
            raise TypeError("Vector endpoints (p1, p2) must be PointArray or Point instances.")
        if p1.system is not p2.system:
            raise ValueError(f"Vector endpoints must share the same coordinate system ({p1.system} != {p2.system}).")

        # Shape check needs to compare the structural shape, ignoring the last coord dim.
        # Allow one operand to be a Point (ndim==1) to broadcast against a PointArray.
        if p1.ndim == p2.ndim and p1.shape[0] != p2.shape[0]:
            raise ValueError(
                f"Vector endpoints must have the same structural shape ({p1.shape[:-1]} != {p2.shape[:-1]})."
            )

        self._p1 = p1
        self._p2 = p2
        self._displacement = p2.coordinates - p1.coordinates
        # Cache displacement? For now, compute on demand.
        # self._displacement = self._p2.coordinates - self._p1.coordinates

    @property
    def p1(self) -> PointArray:
        """The starting points of the vectors."""
        return self._p1

    @property
    def p2(self) -> PointArray:
        """The ending points of the vectors."""
        return self._p2

    @property
    def system(self) -> CoordinateSystem:
        """The coordinate system of the vector endpoints."""
        return self._p1.system

    @property
    def displacement(self) -> np.ndarray:
        """The displacement array (p2.coordinates - p1.coordinates)."""
        # Ensure coordinates are used for subtraction
        return self._displacement

    @property
    def shape(self) -> tuple:
        """The shape of the array structure (like PointArray)."""
        return self.displacement.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array structure (like PointArray)."""
        return self.displacement.ndim

    @property
    def dtype(self) -> np.dtype:
        """The data type of the vector components."""
        # Displacement dtype might differ from p1/p2 if they are integers
        return self.displacement.dtype

    def __len__(self) -> int:
        """The number of vectors in the array (first dimension, like PointArray)."""
        return len(self.displacement)

    def __getitem__(self, index) -> np.ndarray:
        """Get the displacement vector(s) at the given index."""
        # Return the displacement component, not a new VectorArray slice
        return self.displacement[index]

    def __iter__(self):
        """Iterate over the displacement vectors."""
        yield from self.displacement

    def _check_vector_operand(self, other: VectorArray):
        """Validate operand for vector-vector operations."""
        if not isinstance(other, (Vector, VectorArray)):
            raise TypeError(f"Operand must be a VectorArray or Vector (received {type(other)})")
        if self.system is not other.system:
            raise ValueError(
                f"Operand system '{other.system}' does not match this VectorArray's system '{self.system}'"
            )
        # Shape check for addition requires broadcast compatibility or equality
        # Simple equality check for now, consider broadcasting later if needed.
        if self.shape[-1] != other.shape[-1]:
            raise ValueError(f"Operand shapes {self.shape} and {other.shape} are not compatible.")

    def __add__(self, other: VectorArray | PointArray) -> VectorArray | PointArray:
        """
        Add this vector array to another vector array or a point array.

        Vector + Vector = Vector (Result starts at self.p1)
        Vector + Point = Point
        """
        if isinstance(other, (Vector, VectorArray)):
            self._check_vector_operand(other)

            if isinstance(self, Vector) and isinstance(other, Vector):
                return Vector(self.p1, self.p2 + other)
            return VectorArray(self.p1, self.p2 + other)
        elif isinstance(other, PointArray):
            # Adding vector to point: Point + Vector = Point
            # Use PointArray._check_vector_operand for system check
            other._check_vector_operand(self)  # Check system match from Point's perspective
            new_coords = other.coordinates + self.displacement
            if isinstance(other, Point) and isinstance(self, Vector):
                return Point(new_coords, system=self.system)
            return PointArray(new_coords, system=self.system)
        return NotImplemented  # Let Python try other.__radd__(self)

    def mapped_to(self, system) -> "VectorArray":
        """
        Map this vector/vector array to a new coordinate system.

        This is done by mapping the start and end points of the vector(s)
        to the target system. The new vector(s) are defined by these
        newly mapped points.

        Parameters
        ----------
        system : str | CoordinateSystem
            The target coordinate system.

        Returns
        -------
        Vector | VectorArray
            A new vector or vector array in the target coordinate system.
            The type of the returned object will match the type of `self`.
        """
        p1_mapped = self.p1.mapped_to(system)
        p2_mapped = self.p2.mapped_to(system)
        if isinstance(self, Vector):
            return Vector(p1_mapped, p2_mapped)
        else:
            return VectorArray(p1_mapped, p2_mapped)

    def __repr__(self):
        # Use PointArray repr for consistency if not Vector subclass
        if type(self) is VectorArray:
            # Show structural shape and system
            return f"<{type(self).__name__} shape={self.shape} system={self.system.name}>"
        # Fallback for subclasses (like Vector) - this part will be overridden by Vector.__repr__
        return f"<{type(self).__name__} from={self.p1} to={self.p2}>"

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __eq__(self, b):
        if not isinstance(b, type(self)):
            return False
        if self.system is not b.system:
            return False
        # Check shapes of underlying points match structurally
        if self.shape != b.shape:
            return False
        # Let's compare displacements for robustness with floats
        return np.allclose(self.displacement, b.displacement)


class Vector(VectorArray):
    """
    Represents a single vector (displacement) in space.

    Defined by two Point endpoints, p1 (start) and p2 (end).
    Inherits transformation and addition logic from VectorArray.
    """

    def __init__(self, p1: Point, p2: Point):
        if not isinstance(p1, Point) or not isinstance(p2, Point):
            raise TypeError("Vector endpoints (p1, p2) must be Point instances.")
        # Base class init handles system and shape checks
        super().__init__(p1, p2)

    # Properties (p1, p2, system, displacement) are inherited

    # Ensure p1/p2 return Point type
    @property
    def p1(self) -> Point:
        """The starting point of the vector."""
        return self._p1  # Should already be a Point due to __init__ check

    @property
    def p2(self) -> Point:
        """The ending point of the vector."""
        return self._p2  # Should already be a Point due to __init__ check

    # Override shape/ndim/len to reflect single vector nature (like Point)
    @property
    def shape(self) -> tuple:
        """The shape of the coordinate array for the vector (e.g., (3,))."""
        # Return shape of the 1D coordinate array
        return self.p1.shape  # Same as p1.coordinates.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the coordinate array (always 1 for Vector)."""
        return 1  # A single vector is 1D in terms of coordinates

    def __len__(self) -> int:
        """The number of coordinate dimensions."""
        return self.shape[0]  # Length of the 1D coordinate array

    def __repr__(self):
        disp_tuple = tuple(self.displacement)
        # Use short repr for endpoints
        p1_repr = repr(self.p1)
        p2_repr = repr(self.p2)
        return f"<{type(self).__name__} {disp_tuple} system={self.system.name} from={p1_repr} to={p2_repr}>"

    # _coorx_transform is handled by base class checking instance type
    # __add__ is handled by base class checking instance type
    # __eq__ uses base class implementation (comparing displacements)
