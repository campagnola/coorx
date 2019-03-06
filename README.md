Transformy
==========

Object-oriented linear and nonlinear coordinate system transforms.

** This project is in early development; the list below is at least 50% wishful planning **

* A collection of different types of coordinate system transform classes with full unit test coverage
* Easy methods for mapping coordinate data through these transforms
* Transform composition and simplification
* Automatic generation of composite transforms from a coordinate system graph
* Coordinate arrays that know which coordinate system they live in to handle automatic mapping
* Conversion of transforms between ITK, Qt, scikit-image, vispy, etc.

Examples
========

Scale and translate 2D coordinates:

```
import numpy as np
from transformy import *

coords = np.array([
    [ 0,  0],
    [ 1,  2],
    [20, 21],
])

tr = STTransform(scale=(10, 1), offset=(5, 5))

tr.map(coords)
# returns:
# [ [  5.,  5.],
#   [ 15.,  7.],
#   [205., 26.]  ]
```

Compose multiple transforms together:
    
```
tr1 = STTransform(scale=(1, 10, 100))

tr2 = AffineTransform(dims=3)
tr2.rotate(90, axis=(0, 0, 1))

tr3 = CompositeTransform([tr2, tr1])

tr3.map(coords)
```



Todo
====

* import bilinear, SRT transforms from pyqtgraph
* import coordinate system graph handling from vispy
* make coordinate system dimensionality explicit
* unit tests against ITK output


Credit
======

Transformy is adapted from code originally written for VisPy (vispy.org),
inspired by the nice transform classes in ITK, and
maintained by the Allen Institute for Brain Science.
