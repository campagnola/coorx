Coorx
==========

Object-oriented linear and nonlinear coordinate system transforms.

* A collection of different types of coordinate system transform classes with unit test coverage
* Easy methods for mapping coordinate data through these transforms
* Transform composition and simplification
* Transforms intelligently map data types including numpy arrays, lists, etc.

Wishlist:

* Automatic generation of composite transforms from a coordinate system graph
* Coordinate arrays that know which coordinate system they live in to handle automatic mapping
* Conversion of transforms between ITK, Qt, scikit-image, vispy, etc.
* Numba, cuda optimization


Installation
============

To install the package from PyPI, use the following command:

```
pip install coorx
```

Usage
=====

Scale and translate 2D coordinates:

```python
import numpy as np
from coorx import STTransform

coords = np.array([
    [ 0,  0],
    [ 1,  2],
    [20, 21],
])

tr = STTransform(scale=(10, 1), offset=(5, 5))

print(tr.map(coords))
```

Compose multiple transforms together:

```
import numpy as np
from coorx import STTransform, AffineTransform, CompositeTransform

coords = np.array([
    [0, 0, 0],
    [1, 2, 3],
    [-10, -200, -3000],
])

tr1 = STTransform(scale=(1, 10, 100))

tr2 = AffineTransform(dims=3)
tr2.rotate(90, axis=(0, 0, 1))

tr3 = CompositeTransform([tr2, tr1])

print(tr3.map(coords))
```

Todo
====

* import bilinear, SRT transforms from pyqtgraph
* import coordinate system graph handling from vispy
* make coordinate system dimensionality explicit
* unit tests against ITK output


Credit
======

Coorx is adapted from code originally written for VisPy (vispy.org),
inspired by the nice transform classes in ITK, and
maintained by the Allen Institute for Brain Science.
