import numpy as np
from .types import StrOrNone, CoordSysOrStr

    
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
    def __init__(self, coordinates, system:CoordSysOrStr=None, graph:StrOrNone=None):
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
        for x in self.coordinates:
            yield x

    def _check_operand(self, a):
        assert isinstance(a, PointArray)
        assert a.system is self.system

    def __add__(self, b):
        self._check_operand(b)
        return self.coordinates + b.coordinates

    def __sub__(self, b):
        self._check_operand(b)
        return self.coordinates - b.coordinates

    def mapped_through(self, cs_list):
        chain = self.system.graph.transform_chain([self.system] + cs_list)
        return chain.map(self.coordinates)

    def mapped_to(self, system):
        # todo: automatically determine longer chains
        chain = self.system.graph.transform_chain([self.system, system])
        return chain.map(self.coordinates)

    def set_system(self, system, graph=None):
        from .systems import get_coordinate_system
        self.system = get_coordinate_system(system, graph=graph, ndim=self.shape[-1], create=True)

    def _coorx_transform(self, tr):
        if tr.systems[0] is not self.system:
            raise TypeError(f"The transform {tr} maps from system {tr.systems[0]}, but this PointArray is defined in {self.system}")
        mapped = tr.map(self.coordinates)
        return PointArray(mapped, system=tr.systems[1])

    def __array__(self, dtype=None):
        return self.coordinates.astype(dtype, copy=False)

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
        if type(b) != type(self):
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
                for i in range(coord_arr.ndim - 1):
                    first = first[0]

            if isinstance(first, Point):
                source_system = first.system
            else:
                source_system = None

        return coord_arr, source_system
        
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
            raise TypeError(f"The transform {tr} maps from system {tr.systems[0]}, but this Point is defined in {self.system}")
        mapped = tr.map(self.coordinates)
        return Point(mapped, system=tr.systems[1])

    def __repr__(self):
        return f"<{type(self).__name__} {tuple(self.coordinates)} in {self.system.name}>"

