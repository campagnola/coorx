from ._types import Mappable

from .base_transform import Transform, DependentTransformError, InverseTransform
from .params import TransformListParameter


class CompositeTransform(Transform):
    """
    Transform subclass that performs a sequence of transformations in
    order.

    Arguments:

    transforms : list of Transform instances
        See ``transforms`` property.
    """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    Equidimensional = False

    parameter_spec = [
        TransformListParameter("transforms"),
    ]

    def __init__(self, transforms=None, **kwargs):
        super().__init__(transforms=transforms, **kwargs)

    def _init_with_no_state(self):
        super()._init_with_no_state()
        self._simplified = None

    @property
    def transforms(self):
        """The list of transform that make up the transform chain.

        The order of transforms is given such that the first transform in the
        list is the first to be invoked when mapping coordinates through
        the chain.

        For example, the following two mappings are equivalent::

            # Map coordinates through individual transforms:
            trans1 = STTransform(scale=(2, 3), translate=(0, 1))
            trans2 = PolarTransform()
            mapped = trans2.map(trans1.map(coords))

            # Equivalent mapping through chain:
            chain = CompositeTransform([trans1, trans2])
            mapped = chain.map(coords)

        """
        return self._state['transforms']

    @transforms.setter
    def transforms(self, value):
        self.set_params(transforms=value)

    def _validate_dims(self, dims, **kwargs):
        return None

    @property
    def systems(self):
        return self.transforms[0].systems[0], self.transforms[-1].systems[1]

    def set_systems(self, from_cs, to_cs, cs_graph=None):
        raise DependentTransformError("Cannot set systems on a CompositeTransform")

    def copy(self, from_cs=None, to_cs=None, system_names=None):
        """Return a copy of this transform chain.

        By default, the copied sub-transforms have no coordinate systems and are not
        registered in any coordinate system graph. To register the copy, systems must
        be named explicitly for every point along the chain; if any of the arguments
        below is given, then names must resolve for *all* of the chain's systems,
        otherwise ValueError is raised.

        Parameters
        ----------
        from_cs : str | CoordinateSystem | None
            Coordinate system to assign to the copy's input.
        to_cs : str | CoordinateSystem | None
            Coordinate system to assign to the copy's output.
        system_names : dict | str | None
            Names for systems not covered by *from_cs* / *to_cs*, derived from the
            original systems' names: a dict maps original name to new name (a missing
            key raises ValueError); a str is appended to the original name as a suffix.

        Each sub-transform is copied with its resolved endpoint pair, so the entire
        chain is registered as new graph edges. The registered chain may never replace
        or shadow existing graph edges: names derived via *system_names* must not
        collide with existing systems, and at most one of *from_cs* / *to_cs* may be
        an existing system (ValueError otherwise).

        A registered copy is flattened: nested chains are expanded in place so that
        their interior systems can be named too. A chain that visits the same pair of
        systems twice (a transform and its inverse, say) registers that edge once and
        shares it; if two *different* transforms would claim one edge, ValueError is
        raised and nothing is registered.

        Examples
        --------
        chain.copy()                      # system-less copies of all sub-transforms
        chain.copy(system_names='_f12')   # 'a'->'b'->'c' becomes 'a_f12'->'b_f12'->'c_f12'
        chain.copy(from_cs='global', system_names='_frame12')
        """
        if len(self.transforms) == 0:
            if from_cs is not None or to_cs is not None or system_names is not None:
                raise ValueError("Cannot set systems on an empty CompositeTransform")
            return CompositeTransform([], dynamic=self.dynamic)
        flat = self._flat_transforms()
        systems = [flat[0].systems[0]] + [t.systems[1] for t in flat]
        resolved = Transform._resolve_copy_systems(systems, from_cs, to_cs, system_names)
        if resolved is None:
            copies = [t.copy() for t in self.transforms]
        else:
            graph = self._graph_for_new_systems(from_cs, to_cs)
            self._check_derived_systems(resolved, from_cs, to_cs, graph)
            copies = self._register_chain_copies(flat, resolved, graph)
        return CompositeTransform(copies, dynamic=self.dynamic)

    def _flat_transforms(self):
        """Return this chain's sub-transforms with nested chains expanded in place.

        Registering a copy of a chain names every coordinate system along it, which
        requires the systems interior to a nested chain to be visible here.
        """
        flat = []
        for tr in self.transforms:
            inner = tr.inverse if isinstance(tr, InverseTransform) else None
            if isinstance(tr, CompositeTransform):
                flat.extend(tr._flat_transforms())
            elif isinstance(inner, CompositeTransform):
                flat.extend(sub.inverse for sub in reversed(inner._flat_transforms()))
            else:
                flat.append(tr)
        return flat

    @staticmethod
    def _register_chain_copies(transforms, resolved, graph):
        """Return copies of *transforms* registered as canonical edges between
        consecutive *resolved* systems.

        A chain may visit the same pair of systems more than once, most often as a
        transform together with its inverse; all such visits share the single edge
        registered for that pair, rather than each overwriting the last. Two
        *different* transforms claiming one edge cannot be registered at all, so that
        case raises before anything is added to the graph.
        """
        plan = []
        sources = {}
        for i, tr in enumerate(transforms):
            src, edge_from, edge_to, inverted = tr._canonical_edge(resolved[i], resolved[i + 1])
            key = (str(edge_from), str(edge_to))
            if sources.setdefault(key, src) is not src:
                raise ValueError(
                    "Cannot register this chain: two different transforms claim the edge "
                    f"'{key[0]}' -> '{key[1]}'."
                )
            plan.append((key, src, edge_from, edge_to, inverted))

        copies = {}
        for key, src, edge_from, edge_to, _ in plan:
            if key not in copies:
                copies[key] = src._copy_registered(edge_from, edge_to, graph)
        return [copies[key].inverse if inverted else copies[key] for key, _, _, _, inverted in plan]

    @property
    def dims(self):
        if len(self.transforms) == 0:
            return (None, None)
        return self.transforms[-1].dims[0], self.transforms[0].dims[1]

    def set_params(self, transforms=None):
        self._state.setdefault("transforms", [])
        from . import create_transform

        transforms = [t if isinstance(t, Transform) else create_transform(**t) for t in transforms or []]
        for i in range(len(transforms) - 1):
            # None is a deliberate wildcard here, as in Transform.validate_transform_for_mul:
            # a system-less sub-transform joins any neighbour, and any mismatch it hides
            # surfaces only when a Point is mapped through the chain.
            if transforms[i].systems[1] is None or transforms[i + 1].systems[0] is None:
                continue
            if transforms[i].systems[1] != transforms[i + 1].systems[0]:
                raise TypeError(
                    f"Coordinate systems of transform {transforms[i]} "
                    f"({transforms[i].systems[1]}) does not map to {transforms[i+1]} "
                    f"({transforms[i+1].systems[0]})"
                )

        # Avoid extra effort if we already have the correct chain
        curr_xforms = self._state["transforms"]
        if len(transforms) == len(curr_xforms):
            changed = any(transforms[i] is not curr_xforms[i] for i in range(len(transforms)))
            if not changed:
                return

        for t in curr_xforms:
            t.remove_change_callback(self._subtr_changed)
        self._state["transforms"] = transforms
        self._simplified = None
        for t in transforms:
            t.add_change_callback(self._subtr_changed, keep_reference=False, duplicates='ignore')
        self._update()

    @property
    def simplified(self):
        """A simplified representation of the same transformation."""
        if self._simplified is None:
            self._simplified = SimplifiedCompositeTransform(self)
        return self._simplified

    @property
    def Linear(self):
        b = True
        for tr in self.transforms:
            b &= tr.Linear
        return b

    @property
    def Orthogonal(self):
        b = True
        for tr in self.transforms:
            b &= tr.Orthogonal
        return b

    @property
    def NonScaling(self):
        b = True
        for tr in self.transforms:
            b &= tr.NonScaling
        return b

    @property
    def Isometric(self):
        b = True
        for tr in self.transforms:
            b &= tr.Isometric
        return b

    def map(self, obj: Mappable):
        return self._map(obj)

    def _map(self, coords):
        """Map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to map.

        Returns
        -------
        coords : ndarray
            Coordinates.
        """
        for tr in self.transforms:
            coords = tr.map(coords)
        return coords

    def _imap(self, coords):
        """Inverse map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map.

        Returns
        -------
        coords : ndarray
            Coordinates.
        """
        for tr in reversed(self.transforms):
            coords = tr.imap(coords)
        return coords

    def _as_affine(self):
        ret = None
        for tr in self.transforms:
            if ret is None:
                ret = tr.as_affine()
            else:
                ret = tr.as_affine() * ret
        return ret

    @property
    def full_matrix(self):
        mat = None
        for tr in self.transforms:
            if mat is None:
                mat = tr.full_matrix
            else:
                mat = tr.full_matrix.dot(mat)
        return mat

    def as_vispy(self):
        from vispy.visuals.transforms import ChainTransform

        return ChainTransform([tr.as_vispy() for tr in reversed(self.transforms)])

    def append(self, tr):
        """
        Add a new transform to the end of this chain.

        Parameters
        ----------
        tr : instance of Transform
            The transform to use.
        """
        self.transforms.append(tr)
        tr.add_change_callback(self._subtr_changed, keep_reference=False, duplicates='ignore')
        self._update()

    def prepend(self, tr):
        """
        Add a new transform to the beginning of this chain.

        Parameters
        ----------
        tr : instance of Transform
            The transform to use.
        """
        self.transforms.insert(0, tr)
        tr.add_change_callback(self._subtr_changed, keep_reference=False, duplicates='ignore')
        self._update()

    def _subtr_changed(self, event):
        """One of the internal transforms changed; propagate the signal."""
        self._update(event)

    def __setitem__(self, index, tr):
        self.transforms[index].remove_change_callback(self._subtr_changed)
        self.transforms[index] = tr
        tr.add_change_callback(self._subtr_changed, keep_reference=False, duplicates='ignore')
        self._update()

    def __mul__(self, tr):
        if isinstance(tr, CompositeTransform):
            trs = tr.transforms
        else:
            trs = [tr]
        return CompositeTransform(trs + self.transforms)

    def __rmul__(self, tr):
        if isinstance(tr, CompositeTransform):
            trs = tr.transforms
        else:
            trs = [tr]
        return CompositeTransform(self.transforms + trs)

    def __eq__(self, b):
        if not isinstance(b, CompositeTransform):
            return False
        return all(t1 == t2 for t1, t2 in zip(self.transforms, b.transforms))

    def __str__(self):
        names = [tr.__class__.__name__ for tr in self.transforms]
        return f"<{self.__class__.__name__} [{', '.join(names)}] at 0x{id(self):x}>"

    def __repr__(self):
        tr = ",\n                 ".join(map(repr, self.transforms))
        return f"<{self.__class__.__name__} [{tr}] at 0x{id(self):x}>"


class SimplifiedCompositeTransform(CompositeTransform):
    def __init__(self, chain):
        CompositeTransform.__init__(self)
        self._chain = chain
        chain.add_change_callback(self.source_changed, keep_reference=False, duplicates='ignore')
        self.source_changed(None)

    def source_changed(self, event):
        """Generate a simplified chain by joining adjacent transforms."""
        # bail out early if the chain is empty
        transforms = self._chain.transforms[:]
        if len(transforms) == 0:
            self.transforms = []
            return

        # If the change signal comes from a transform that already appears in
        # our simplified transform list, then there is no need to re-simplify.
        if event is not None:
            for source in event.sources[::-1]:
                if source in self.transforms:
                    self._update(event)
                    return

        # First flatten the chain by expanding all nested chains
        new_chain = []
        while len(transforms) > 0:
            tr = transforms.pop(0)
            if isinstance(tr, CompositeTransform) and not tr.dynamic:
                transforms = tr.transforms[:] + transforms
            else:
                new_chain.append(tr)

        # Now combine together all compatible adjacent transforms
        cont = True
        tr = new_chain
        while cont:
            new_tr = [tr[0]]
            cont = False
            for t2 in tr[1:]:
                t1 = new_tr[-1]
                pr = t2 * t1
                if not t1.dynamic and not t2.dynamic and not isinstance(pr, CompositeTransform):
                    cont = True
                    new_tr.pop()
                    new_tr.append(pr)
                else:
                    new_tr.append(t2)
            tr = new_tr

        self.transforms = tr
