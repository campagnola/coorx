"""
Image transformations with coorx transforms carried along. 

Vision for the future: a collection of tools that can process images through a pipeline of transformations
and also output coorx transforms that map from one end of the pipeline to the other.

- rotate/scale around a specific center point
- get a cropped region that includes a specific point(s) with padding
- add padding to an image
- get an interpolated region
- given two partially overlapping images, give the region of overlap (and use this region to automatically slice from
  either image)
- resample / downscale with averaging


Basic use pattern:

    # make some transformations to an image
    img = Image(image_data)
    rotated = img.rotate(angle, center=(row, col))
    cropped = rotated[10:-10, 10:-10]

    # map coordinates from the original image to the cropped image
    pt2 = img.point([row, col]).mapped_to(cropped.system)

    # maybe add a physical coordinate system
    frame_tr = Transform(from_cs='physical', to_cs=img.system)
    pt3 = cropped.point([row, col]).mapped_to('physical')

    
Things to work out:
- how to handle 3d image stacks, rgb images, etc.
- how to handle slicing, indexing, fancy indexing    
- can we stack a bunch of transforms together and apply them all at once?
    tr = CompositeTransform([tr1, tr2, tr3])
    img2 = img.apply_transform(tr, new_shape)
- garbage collection of transforms and coordinate systems attached to images
    - ask graph to keep transforms/systems in weak references
"""
from __future__ import annotations

import numpy as np
import scipy

from .coordinates import Point, PointArray
from .linear import AffineTransform
from .systems import CoordinateSystemGraph, CoordinateSystem


class Image:
    """Wraps image data with a coordinate system and methods
    for transforming the image and mapping coordinates through the transforms.

    Parameters
    ----------
    image : ndarray
        The image data. Must be 2D or higher.
    axes : tuple, optional
        The axes of the image that correspond to spatial dimensions. Defaults to all axes.
    cs_name : str | CoordinateSystem | None
        Optional name of the coordinate system to attach to the image.
    graph : str | CoordinateSystemGraph | None
        Optional graph to use for the coordinate system.
    """

    _image_graph_n = 0

    def __init__(self, image, axes=None, cs_name=None, graph=None):
        if axes is None:
            axes = tuple(range(image.ndim))
        self.axes = axes

        self.image = image
        self._parent_tr = None

        if graph is None:
            graph = 'image_graph'
            Image._image_graph_n += 1
        self.graph = CoordinateSystemGraph.get_graph(graph, create=True)

        if isinstance(cs_name, CoordinateSystem):
            self.system = cs_name
        else:
            if cs_name is None:
                index = 0
                while True:
                    cs_name = f'image_{index}'
                    if cs_name not in self.graph.systems:
                        break
                    index += 1
            self.system = self.graph.add_system(cs_name, ndim=self.ndim)

    @property
    def ndim(self):
        """Return the number of spatial dimensions of the image."""
        return len(self.axes)

    @property
    def shape(self):
        return [self.image.shape[i] for i in self.axes]

    def point(self, coords):
        """Return a Point object with the given (row, col) coordinates."""
        coords = np.asarray(coords)
        if coords.ndim != 1:
            raise ValueError("Point coordinates must be 1D")
        return Point(coords, system=self.system)

    def point_array(self, coords):
        """Return a PointArray object with the given (row, col) coordinates."""
        coords = np.asarray(coords)
        if coords.ndim < 2:
            raise ValueError("coords array must be at least 2D")
        if coords.shape[-1] != self.ndim:
            raise ValueError(
                f"coords.shape[-1] must be {self.ndim}, got {coords.shape[-1]}"
            )
        return PointArray(coords, system=self.system)

    def rotate(self, angle, axes=(0, 1), **kwds):
        """Rotate the image by the given angle around the specified axes.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the image.
        axes : (int, int), optional
            The two axes involved in the rotation. Defaults to (0, 1). Beware: this is different from how
            AffineTransform's rotations work.
        kwds : keyword arguments
            Additional keyword arguments to pass to `scipy.ndimage.rotate`.
        """
        img = self.image
        rotated_img = scipy.ndimage.rotate(img, angle, axes=axes, **kwds)
        img2 = self.copy(image=rotated_img)
        img2._parent_tr = self.make_rotation_transform(
            angle, axes, self.shape, img2.shape, from_cs=self.system, to_cs=img2.system
        )
        return img2

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) != self.ndim:
            item = item + (slice(None),) * (self.ndim - len(item))
        cropped_img = self.image[item]
        img2 = self.copy(image=cropped_img)
        img2._parent_tr = self.make_crop_transform(
            item, self.image, from_cs=self.system, to_cs=img2.system
        )
        return img2

    def zoom(self, factors, **kwds):
        # fill in missing image axes with 1
        if np.isscalar(factors):
            factors = [factors] * self.ndim
        actual = np.ones(self.image.ndim, dtype=float)
        for i, ax in enumerate(self.axes):
            actual[ax] = factors[i]
        scaled_img = scipy.ndimage.zoom(self.image, actual, **kwds)

        img2 = self.copy(image=scaled_img)
        tr = AffineTransform(
            dims=(self.ndim, self.ndim),
            from_cs=self.system,
            to_cs=img2.system,
        )
        tr.scale(factors)
        img2._parent_tr = tr
        return img2

    def copy(self, **updates):
        kwds = {'axes': self.axes, 'graph': self.graph}
        kwds.update(updates)
        return Image(**kwds)

    def make_rotation_transform(self, angle, axes, from_shape, to_shape, **kwds):
        # Make transform mapping unrotated to rotated coordinates
        from_center = np.array(from_shape) / 2
        to_center = np.array(to_shape) / 2
        tr = AffineTransform(dims=(self.ndim, self.ndim), **kwds)
        tr.translate(-from_center)
        if self.ndim == 2:
            tr.rotate(-angle)
        elif self.ndim == 3:
            ax1 = np.zeros(self.ndim)
            ax1[axes[0]] = 1
            ax2 = np.zeros(self.ndim)
            ax2[axes[1]] = 1
            axis = np.cross(ax1, ax2)
            tr.rotate(-angle, axis=axis)
        else:
            raise ValueError("Image must be 2D or 3D for rotation")
        tr.translate(to_center)
        return tr

    def make_crop_transform(self, crop, img, **kwds):
        tr = AffineTransform(dims=(self.ndim, self.ndim), **kwds)
        tr.translate([-crop[i].indices(img.shape[i])[0] for i in range(self.ndim)])
        return tr
