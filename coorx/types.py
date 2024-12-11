import numpy as np
from typing import Union, Protocol


StrOrNone = Union[None, str]
Dim = Union[None, int]
Dims = Union[Dim, 'tuple[Dim]']
CoordSysOrStr = Union[str, 'CoordinateSystem']

class CustomMappable(Protocol):
    def _coorx_transform(self, tr):
        pass

Mappable = Union[np.array, list, tuple, CustomMappable]
