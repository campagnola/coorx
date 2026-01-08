from ._types import Mappable

from .base_transform import Transform, DependentTransformError
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

    def copy(self, from_cs=None, to_cs=None):
        if from_cs is not None or to_cs is not None:
            raise ValueError("Cannot set systems on a CompositeTransform")
        return super().copy()

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

    def as_affine(self):
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
        print("appending", tr)
        self.transforms.append(tr)
        print("appended?", self.transforms)
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
