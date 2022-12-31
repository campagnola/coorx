import numpy as np
from .types import StrOrNone, CoordSysOrStr

    
class PointArray:
    """Represents an N-dimensional array of points. 

    Although the array is instantiated with the coordinates of these points in a particular
    coordinate system, one may request the coordinates mapped to any other system.
    """

    def __init__(self, coordinates, system:CoordSysOrStr, graph:StrOrNone=None):
        from .systems import get_coordinate_system
        coordinates = np.asarray(coordinates)
        self.ndim = coordinates.shape[-1]
        self.coordinates = coordinates

        if system is not None:
            self.system = get_coordinate_system(system, graph=graph, ndim=self.ndim, create=True)

    def mapped_through(self, cs_list):
        chain = self.system.graph.transform_chain([self.system] + cs_list)
        return chain.map(self.coordinates)

    def mapped_to(self, system):
        # todo: automatically determine longer chains
        chain = self.system.graph.transform_chain([self.system, system])
        return chain.map(self.coordinates)

    def _coorx_transform(self, tr):
        if tr.systems[0] is not self.system:
            raise TypeError(f"The transform {tr} maps from system {tr.systems[0]}, but this PointArray is defined in {self.system}")
        mapped = tr.map(self.coordinates)
        return PointArray(mapped, system=tr.systems[1])


class Point(PointArray):
    """Represents a single point in space; one may request the coordinates of this point 
    in any coordinate system.
    """

    def __init__(self, coordinate, system, graph:StrOrNone=None):
        PointArray.__init__(self, [coordinate], system=system, graph=graph)

