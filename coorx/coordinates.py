from __future__ import annotations
import numpy as np
from .types import StrOrNone, CoordSysOrStr, Mappable
from .systems import CoordinateSystem, CoordinateSystemGraph, get_coordinate_system


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
    def __init__(self, coordinates, system: CoordSysOrStr=None, graph: StrOrNone=None):
        coord_arr, source_system = self._interpret_input(coordinates)

        assert coord_arr.dtype is not np.dtype(object)
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

    def _check_operand(self, a):
        if not isinstance(a, PointArray):
            raise TypeError(f"Operand must be a PointArray (received {type(a)})")
        if a.system is not self.system:
            raise ValueError(f"Operand system {a.system} does not match this PointArray's system {self.system}")

    def __add__(self, b):
        if isinstance(b, (Vector, VectorArray)):
            self._check_vector_operand(b)
            new_coords = self.coordinates + b.displacement
            if isinstance(self, Point):
                return Point(new_coords, system=self.system)
            else:
                return PointArray(new_coords, system=self.system)
        else:
            self._check_point_operand(b)
            # This case should likely not return raw coordinates, but maybe another PointArray?
            # For now, keeping original behavior for PointArray + PointArray.
            # Consider if PointArray + PointArray should be allowed or raise TypeError.
            return self.coordinates + b.coordinates

    def __sub__(self, b: PointArray) -> VectorArray:
        self._check_point_operand(b)
        if isinstance(self, Point) and isinstance(b, Point):
            return Vector(b, self)
        else:
            # Ensure operands are PointArray if one is Point
            p1 = b if isinstance(b, PointArray) else PointArray(b.coordinates, system=b.system)
            p2 = self if isinstance(self, PointArray) else PointArray(self.coordinates, system=self.system)
            return VectorArray(p1, p2)

    def mapped_through(self, cs_list) -> PointArray:
        chain = self.system.graph.transform_chain([self.system] + cs_list)
        return chain.map(self)

    def mapped_to(self, system) -> "PointArray":
        path = self.system.graph.transform_path(self.system, system)
        chain = self.system.graph.transform_chain(path)
        return chain.map(self)

    def set_system(self, system, graph=None):
        from .systems import get_coordinate_system
        self.system = get_coordinate_system(system, graph=graph, ndim=self.shape[-1], create=True)

    def _coorx_transform(self, tr):
        if tr.systems[0] is not self.system:
            raise TypeError(f"The transform {tr} maps from system {tr.systems[0]}, but this PointArray is defined in {self.system}")
        mapped = tr.map(self.coordinates)
        if tr.systems[0] is not self.system:
            raise TypeError(f"The transform {tr} maps from system {tr.systems[0]}, but this PointArray is defined in {self.system}")
        
        # Map the raw coordinates
        mapped_coords = tr.map(self.coordinates)
        
        # Determine the output type based on the input type
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
        state['system'] = None if self.system is None else (self.system.name, self.system.graph.name)
        return state

    def __setstate__(self, state):
        sys, graph = state.pop('system')
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

        # if input contains Point instances, carry their system forward
        if coord_arr.size > 0:
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
    def __init__(self, coordinates, system:CoordSysOrStr=None, graph:StrOrNone=None):
        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 1:
            raise TypeError("Point coordinates must be 1D")
        super().__init__(coordinates[np.newaxis, :], system, graph)

    @property
    def coordinates(self):
        return self._coordinates[0]

    def _coorx_transform(self, tr):
        if tr.systems[0] is not self.system:
            raise TypeError(f"The transform {tr} maps from system '{tr.systems[0]}', but this Point is defined in '{self.system}'")
        mapped = tr.map(self.coordinates)
        # This method is now handled by the overridden _coorx_transform in PointArray
        # which checks the instance type before returning.
        # We rely on the base class implementation.
        return super()._coorx_transform(tr)

    def __repr__(self):
        # Ensure coordinates are displayed correctly even if _coordinates is 2D
        coords_tuple = tuple(self.coordinates)
        return f"<{type(self).__name__} {coords_tuple} in {self.system.name}>"
