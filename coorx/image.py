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
from .linear import AffineTransform, TTransform, STTransform
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
    system : str | CoordinateSystem | None
        Optional name of the coordinate system to attach to the image.
    graph : str | CoordinateSystemGraph | None
        Optional graph to use for the coordinate system.
    """

    def __init__(self, image, axes=None, system=None, graph=None):
        if axes is None:
            axes = tuple(range(image.ndim))
        self.spatial_to_image_axes = np.asarray(axes)

        self.image = image
        self._parent_tr = None

        if graph is None:
            graph = system.graph if system is not None else 'image_graph'
        self.graph = CoordinateSystemGraph.get_graph(graph, create=True)

        if isinstance(system, CoordinateSystem):
            self.system = system
        else:
            if system is None:
                index = 0
                while True:
                    system = f'image_{index}'
                    if system not in self.graph.systems:
                        break
                    index += 1
            self.system = self.graph.add_system(system, ndim=self.spatial_ndim)

    @property
    def spatial_ndim(self):
        """Return the number of spatial dimensions of the image."""
        return len(self.spatial_to_image_axes)

    @property
    def spatial_shape(self):
        return tuple(self.image.shape[i] for i in self.spatial_to_image_axes)

    def point(self, coords):
        """Return a Point object with the given (row, col) coordinates in the CS of this image."""
        coords = np.asarray(coords)
        return Point(coords, system=self.system)

    def point_array(self, coords):
        """Return a PointArray object with the given (row, col) coordinates in the CS of this image."""
        coords = np.asarray(coords)
        return PointArray(coords, system=self.system)

    def rotate(self, angle, axes=(0, 1), **kwds):
        """Rotate the image by the given angle around the specified axes.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the image.
        axes : (int, int), optional
            The two spatial axes involved in the rotation. Defaults to (0, 1). Beware: this is different from how
            AffineTransform's rotations work, and different from how Image.__init__ axes work.
        kwds : keyword arguments
            Additional keyword arguments to pass to `scipy.ndimage.rotate`.
        """
        img = self.image
        rotated_img = scipy.ndimage.rotate(img, angle, axes=self.spatial_to_image_axes[list(axes)], **kwds)
        img2 = self.copy(image=rotated_img)
        img2._parent_tr = self.make_rotation_transform(
            angle, axes, self.spatial_shape, img2.spatial_shape, from_cs=self.system, to_cs=img2.system
        )
        return img2

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if not all(isinstance(i, slice) for i in item):
            raise ValueError(
                f"Image.__getitem__ requires a tuple of slices, got {item}"
            )
        if len(item) != self.spatial_ndim:
            item = item + (slice(None),) * (self.spatial_ndim - len(item))
        cropped_img = self.image[item]
        img2 = self.copy(image=cropped_img)
        img2._parent_tr = self.make_crop_transform(
            item, self.image, from_cs=self.system, to_cs=img2.system
        )
        return img2

    def zoom(self, factors, **kwds):
        # fill in missing image axes with 1
        if np.isscalar(factors):
            factors = [factors] * self.spatial_ndim
        actual = np.ones(self.image.ndim, dtype=float)
        for i, ax in enumerate(self.spatial_to_image_axes):
            actual[ax] = factors[i]
        scaled_img = scipy.ndimage.zoom(self.image, actual, **kwds)

        img2 = self.copy(image=scaled_img)
        tr = STTransform(
            dims=(self.spatial_ndim, self.spatial_ndim),
            from_cs=self.system,
            to_cs=img2.system,
            cs_graph=self.graph,
        )
        tr.scale = factors
        img2._parent_tr = tr
        return img2

    def copy(self, **updates):
        kwds = {'axes': self.spatial_to_image_axes, 'graph': self.graph}
        if 'image' not in updates:
            kwds['image'] = self.image.copy()
        kwds.update(updates)
        return Image(**kwds)

    def make_rotation_transform(self, angle, axes, from_shape, to_shape, **kwds):
        # Make transform mapping unrotated to rotated coordinates
        from_center = np.array(from_shape) / 2
        to_center = np.array(to_shape) / 2
        tr = AffineTransform(dims=(self.spatial_ndim, self.spatial_ndim), cs_graph=self.graph, **kwds)
        tr.translate(-from_center)
        if self.spatial_ndim == 2:
            tr.rotate(-angle)
        elif self.spatial_ndim == 3:
            ax1 = np.zeros(self.spatial_ndim)
            ax1[axes[0]] = 1
            ax2 = np.zeros(self.spatial_ndim)
            ax2[axes[1]] = 1
            axis = np.cross(ax1, ax2)
            tr.rotate(-angle, axis=axis)
        else:
            raise ValueError("Image must be 2D or 3D for rotation")
        tr.translate(to_center)
        return tr

    def make_crop_transform(self, crop, img, **kwds):
        offset = [-crop[i].indices(img.shape[i])[0] for i in self.spatial_to_image_axes]
        return TTransform(offset=offset, dims=(self.spatial_ndim, self.spatial_ndim), cs_graph=self.graph, **kwds)
