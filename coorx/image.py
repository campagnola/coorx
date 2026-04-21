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

    Each Image is assigned a CoordinateSystem that corresponds to the pixel 
    coordinates of the image. The axes of this coordinate system correspond to
    the array axes of the imge, as specified by the `axes` argument to the constructor. 
 
    Parameters
    ----------
    image : ndarray
        The image data. Must be 2D or higher.
    axes : tuple, optional
        The axes of the image that correspond to spatial dimensions. Defaults to all axes.
        This allows the image to have extra non-spatial dimensions (e.g. color channels, 
        time points) that are ignored for coordinate mapping.
    system : str | CoordinateSystem | None
        Optional name of the coordinate system to attach to the image.
    graph : str | CoordinateSystemGraph | None
        Optional graph to use for the coordinate system.


    Example
    -------

    .. code-block:: python

        # Let's say we have a 100x100 RGB video (10 frames)
        #   axes=(1, 2) means that we will track the coordinates of axes 1 and 2 (rows, cols)
        #   of the image, and ignore axis 0 (time) and axis 3 (color channels) for coordinate mapping purposes.
        img = Image(np.zeros((10, 100, 100, 3)), axes=(1, 2))

        # Pick a point at row 20, column 50 in the original image
        pt = img.point([20, 50])  # corresponds to img.data[:, 20, 50, :]

        # rotate and crop the image
        rotated = img.rotate(45)
        cropped = rotated[10:-10, 10:-10]

        # map the point through the transformations to get its coordinates in the cropped image
        pt2 = pt.mapped_to(cropped.system)

    
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
        """Return a Point object with the given coordinates in the CS of this image."""
        coords = np.asarray(coords)
        return Point(coords, system=self.system)

    def point_array(self, coords):
        """Return a PointArray object with the given coordinates in the CS of this image."""
        coords = np.asarray(coords)
        return PointArray(coords, system=self.system)

    def rotate(self, angle, axes=(0, 1), **kwds):
        """Rotate the image by the given angle around the specified axes.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the image.
        axes : (int, int), optional
            The two spatial axes (indices into the `axes` argument provided when the Image
            was initialized) involved in the rotation. Defaults to (0, 1). 
            Beware: this is different from how
            AffineTransform's rotations work, and different from how Image.__init__ axes work.
        kwds : keyword arguments
            Additional keyword arguments to pass to `scipy.ndimage.rotate`.

        Returns
        -------
        Image
            A new Image object containing the rotated image and a transform mapping from the original image coordinates
            to the rotated image coordinates.
        """
        img = self.image
        rotated_img = scipy.ndimage.rotate(img, angle, axes=self.spatial_to_image_axes[list(axes)], **kwds)
        img2 = self.copy(image=rotated_img)
        img2._parent_tr = self.make_rotation_transform(
            angle, axes, self.spatial_shape, img2.spatial_shape, from_cs=self.system, to_cs=img2.system
        )
        return img2

    def __getitem__(self, item):
        """Return a cropped version of the image corresponding to the given slice(s).

        Slices should be provided as a tuple of slice objects, one for each *spatial* dimension.
        """
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
        """Zoom the image by the given factors along each spatial axis.

        Parameters
        ----------
        factors : float or array-like
            The zoom factor(s) for each spatial axis. If a single float is given, the same factor is applied to all spatial axes.
        kwds : keyword arguments
            Additional keyword arguments to pass to `scipy.ndimage.zoom`.

        Returns
        -------
        Image
            A new Image object containing the zoomed image and a transform mapping from the original image coordinates
            to the zoomed image coordinates.
        """
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

    def crop_around(self, center, size, **kwds):
        """Crop a region of the given size around the specified center point.

        Parameters
        ----------
        center : array-like or Point
            The center of the crop region, in this image's pixel coordinate system.
            If a Point is given, it is mapped into this image's system automatically.
        size : array-like or int
            The total size of the cropped region in pixels along each spatial axis
            (i.e. size/2 extends on each side of center).
            If a single int is given, the same size is applied to all dimensions.
            The actual size may be smaller if the crop would go outside the image boundaries.
        """
        if isinstance(center, (Point, PointArray)):
            center = center.mapped_to(self.system).coordinates
        center = np.asarray(center, dtype=float)

        if np.isscalar(size):
            size = [size] * self.spatial_ndim
        size = np.asarray(size, dtype=float)

        slices = []
        for i, ax in enumerate(self.spatial_to_image_axes):
            img_size = self.image.shape[ax]
            start = max(0, int(np.floor(center[i] - size[i] / 2)))
            stop = min(img_size, int(np.ceil(center[i] + size[i] / 2)))
            slices.append(slice(start, stop))

        return self[tuple(slices)]

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
        indices = [crop[i].indices(img.shape[i]) for i in self.spatial_to_image_axes]
        starts = [idx[0] for idx in indices]
        steps = [idx[2] for idx in indices]
        if all(s == 1 for s in steps):
            return TTransform(
                offset=[-start for start in starts],
                dims=(self.spatial_ndim, self.spatial_ndim),
                cs_graph=self.graph,
                **kwds,
            )
        tr = STTransform(dims=(self.spatial_ndim, self.spatial_ndim), cs_graph=self.graph, **kwds)
        tr.scale = [1.0 / step for step in steps]
        tr.offset = [-start / step for start, step in zip(starts, steps)]
        return tr
