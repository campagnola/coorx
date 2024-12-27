import numpy as np

from .types import StrOrNone, CoordSysOrStr


class PointArray(np.ndarray):
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
    def __new__(cls, coordinates, system: CoordSysOrStr = None, graph: StrOrNone = None):
        points_arr, points_system, points_graph = cls._interpret_input(coordinates)
        assert points_arr.dtype is not np.dtype(object)

        obj = np.array(points_arr).view(cls)

        # TODO using str here sucks
        if str(points_system) != str(system) and None not in (points_system, system):
            raise TypeError(f"System {system} does not match source coordinates {points_system}")
        system = points_system or system
        graph = points_graph or graph
        if system is not None:
            obj.set_system(system, graph)
        if obj.system is not None and obj.shape[-1] != obj.system.ndim:
            raise ValueError(f"System ndim is {obj.system.ndim}, but coordinate data is {obj.shape[-1]}D")

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.system = getattr(obj, "system", None)
        self.graph = getattr(obj, "graph", None)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        return Point(result, self.system, self.graph)

    @property
    def coordinates(self):
        return self.view(np.ndarray)

    def _check_operand(self, a):
        if isinstance(a, PointArray):
            assert a.system is self.system

    def __add__(self, b):
        self._check_operand(b)
        return super().__add__(b)

    def __sub__(self, b):
        self._check_operand(b)
        return super().__sub__(b)

    def mapped_through(self, cs_list):
        chain = self.system.graph.transform_chain([self.system] + cs_list)
        return chain.map(self)

    def mapped_to(self, system):
        # todo: automatically determine longer chains
        chain = self.system.graph.transform_chain([self.system, system])
        return chain.map(self)

    def set_system(self, system, graph=None):
        from .systems import get_coordinate_system

        self.system = get_coordinate_system(system, graph=graph, ndim=self.shape[-1], create=True)

    def _coorx_transform(self, tr):
        if tr.systems[0] is not self.system:
            raise TypeError(
                f"The transform {tr} maps from system {tr.systems[0]}, but this PointArray is defined in {self.system}"
            )
        mapped = tr.map(self.coordinates)
        return PointArray(mapped, system=tr.systems[1])

    def __repr__(self):
        return f"<{type(self).__name__} {self.shape} in {self.system}>"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["system"] = None if self.system is None else (self.system, self.system.graph.name)
        return state

    def __setstate__(self, state):
        sys, graph = state.pop("system")
        self.__dict__.update(state)
        self.set_system(sys, graph)

    def __eq__(self, b):
        self._check_operand(b)
        return super().__eq__(b)

    @staticmethod
    def _interpret_input(coordinates):
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
                source_graph = first.graph
            else:
                source_system = None
                source_graph = None

        return coord_arr, source_system, source_graph


class Point(PointArray):
    """Represents a single point in space; one may request the coordinates of this point
    in any coordinate system.
    """
    def __new__(cls, coordinates, system: CoordSysOrStr = None, graph: StrOrNone = None):
        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 1:
            raise TypeError("Point coordinates must be 1D")
        return super().__new__(cls, coordinates, system, graph).view(cls)

    def __getitem__(self, idx):
        return self.coordinates[idx]

    def _coorx_transform(self, tr):
        if tr.systems[0] is not self.system:
            raise TypeError(
                f"The transform {tr} maps from system '{tr.systems[0]}', but this Point is defined in '{self.system}'"
            )
        mapped = tr.map(self.coordinates)
        return Point(mapped, system=tr.systems[1])

    def __repr__(self):
        return f"<{type(self).__name__} {tuple(self)} in {self.system}>"
