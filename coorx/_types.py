from typing import Union, Protocol

import numpy as np

StrOrNone = Union[None, str]
Dim = Union[None, int]
Dims = Union[Dim, 'tuple[Dim, Dim]']
CoordSysOrStr = Union[str, 'CoordinateSystem']
GraphOrGraphName = Union[str, None, 'CoordinateSystemGraph']


class CustomMappable(Protocol):
    def _coorx_transform(self, tr):
        pass


Mappable = Union[np.ndarray, list, tuple, CustomMappable]
