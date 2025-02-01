from __future__ import annotations

from .types import StrOrNone, CoordSysOrStr


def get_coordinate_system(system: CoordSysOrStr, graph: StrOrNone = None, ndim=None, create=False):
    """Return a CoordinateSystem instance."""
    graph = CoordinateSystemGraph.get_graph(graph)
    return graph.check_system(system, ndim=ndim, create=create)


class CoordinateSystemGraph:
    """Multiple coordinate systems connected by transforms.

    A CoordinateSystemGraph keeps track of the relationships between many coordinate systems.
    This makes it possible to automatically determine a chain of transforms that map
    between any two coordinate systems (as long as they are indirectly connected within
    the graph).

    CoordinateSystemGraph also provides basic sanity checking:
    - Only one coordinate system of each name, and that it has the expected properties
      (such as dimensionality)
    - Only one transform connecting any pair of coordinate systems
      (requires unique_transforms=True; the default graph ignores this constraint)

    Caveats
    =======
    * setting a CoordinateSystemGraph to have unique_transforms is incompatible with
      using `as_affine` or anything that depends on that (`full_matrix`, `as_vispy`,
      etc.)
    """

    all_graphs = {}

    @classmethod
    def get_graph(cls, graph_name=None):
        return cls.all_graphs[graph_name]

    def __init__(self, name: str, unique_transforms=False):
        assert name not in self.all_graphs, f"A coordinate system graph named {name} already exists."
        self.all_graphs[name] = self
        self.name = name
        self.unique_transforms = unique_transforms
        self.systems = {}  # maps {system_name: CoordinateSystem}
        self.transforms = {}  # maps {cs1: {cs2: transform, ...}, ...}

    def add_system(self, name: str, ndim: int):
        assert name not in self.systems, f"A system named '{name}' is already added to this graph"
        assert isinstance(ndim, int)
        sys = CoordinateSystem(name=name, ndim=ndim, graph=self)
        self.systems[name] = sys
        return sys

    def get_system(self, name):
        return self.systems[name]

    def add_transform(self, transform: "Transform", from_cs: CoordSysOrStr, to_cs: CoordSysOrStr):
        # look up coordinate systems
        cs = (
            self.check_system(from_cs, ndim=transform.dims[0], create=True),
            self.check_system(to_cs, ndim=transform.dims[1], create=True),
        )

        # make sure no transform exists linking these systems
        if self.unique_transforms:
            have_transform_already = cs[0] in self.transforms and cs[1] in self.transforms[cs[0]]
            have_inverse_already = cs[1] in self.transforms and cs[0] in self.transforms[cs[1]]
            assert not have_transform_already, f"A transform is already added connecting '{from_cs}' to '{to_cs}'"
            assert not have_inverse_already, f"A transform is already added connecting '{to_cs}' to '{from_cs}'"

        # make sure this transform has not been assigned elsewhere
        assert transform._systems == (
            None,
            None,
        ), "This transform already connects a different set of coordinate systems"
        transform._systems = cs

        # record the new transform connection
        self.transforms.setdefault(cs[0], {})[cs[1]] = transform

    def check_system(self, system: CoordSysOrStr, ndim: StrOrNone = None, create: bool = False):
        """Check that a system exists with the given name and ndim.

        If `create` is True, then the system may be created if it does not already exist.

        Return the named CoordinateSystem instance.
        """
        # get the CoordinateSystem instance, creating if needed
        if isinstance(system, str) and system not in self.systems and create:
            system = self.add_system(system, ndim)
        else:
            system = self.system(system)

        # check ndim is correct
        if ndim is not None and system.ndim != ndim:
            raise TypeError(f"System '{system}' is {system.ndim}D (expected {ndim}D)")

        return system

    def system(self, system: CoordSysOrStr) -> "CoordinateSystem":
        """Return a CoordinateSystem belonging to this graph."""
        if isinstance(system, str):
            if system not in self.systems:
                raise NameError(f"No coordinate system named '{system}' in this graph")
            return self.systems[system]
        elif isinstance(system, CoordinateSystem):
            assert system.graph is self, "CoordinateSystem {name} belongs to a different graph"
            return system
        else:
            raise TypeError("system must be str or CoordinateSystem instance")

    def transform(self, cs1: CoordSysOrStr, cs2: CoordSysOrStr):
        """Return the transform linking cs1 to cs2, or raise KeyError if none is defined."""
        # check that coordinate systems exist
        cs1, cs2 = self.system(cs1), self.system(cs2)

        fwd = self.transforms.get(cs1, {}).get(cs2, None)
        if fwd is not None:
            return fwd
        inv = self.transforms.get(cs2, {}).get(cs1, None)
        if inv is not None:
            return inv.inverse

        raise TypeError(f"No transform defined linking '{cs1}' to '{cs2}'")

    def transform_path(self, start, end) -> list:
        """Return a list of transforms needed to map from start to end."""
        start, end = self.system(start), self.system(end)
        if start == end:
            return []
        if found := self._find_path(start, end, set()):
            return [start] + found
        raise TypeError(f"No transform path from {start} to {end}")

    def _find_path(self, start, end, visited) -> list | None:
        if start in visited:
            return None
        visited.add(start)
        from_start = self.transforms.get(start, {})
        if end in from_start:
            return [end]
        for next_cs in from_start:
            path = self._find_path(next_cs, end, visited)
            if path is not None:
                return [next_cs] + path
        to_start = {cs for cs in self.transforms if start in self.transforms[cs]}
        if end in to_start:
            return [end]
        for next_cs in to_start:
            path = self._find_path(next_cs, end, visited)
            if path is not None:
                return [next_cs] + path
        return None

    def transform_chain(self, systems):
        from .composite import CompositeTransform

        transforms = []
        for i in range(1, len(systems)):
            cs1, cs2 = systems[i - 1 : i + 1]
            transforms.append(self.transform(cs1, cs2))
        return CompositeTransform(transforms)


class CoordinateSystem:
    def __init__(self, name: str, ndim: int, graph: CoordinateSystemGraph):
        self.graph = graph
        self.name = name
        self.ndim = ndim

    def __repr__(self):
        if self.graph.name is not None:
            name = f"{self.graph.name}:{self.name}"
        else:
            name = self.name
        return f"<CoordinateSystem {name}[{self.ndim}]>"

    def __str__(self):
        return self.name

    def save_state(self):
        return {"type": type(self).__name__, "name": self.name, "ndim": self.ndim, "graph": self.graph.name}


default_cs_graph = CoordinateSystemGraph(name=None, unique_transforms=False)
