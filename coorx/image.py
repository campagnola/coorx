"""
Image transformations with coorx transforms carried along. 

Vision for the future: a collection of tools that can process images through a pipeline of transformations
and also output coorx transforms that map from one end of the pipeline to the other.

- rotate/scale around a specific center point
- get a cropped region that includes a specific point(s) with padding
- add padding to an image
- get an interpolated region
- given two partially overlapping images, give the region of overlap (and use this region to automatically slice from either image)
- resample / downscale with averging


Basic use pattern:

    # make some transformations to an image
    img = Image(image_data)
    rotated = img.rotate(angle, center=(row, col))
    cropped = rotated[10:-10, 10:-10]

    # map coordinates from the original image to the cropped image
    pt2 = img.point([row, col]).mapped_to(cropped.cs)

    # maybe add a physical coordinate system
    frame_tr = Transform(from_cs='physical', to_cs=img.cs)
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
import numpy as np
import scipy
import coorx
from coorx.systems import CoordinateSystemGraph


class Image:
    """Wraps image data with a coordinate system and methods 
    for transforming the image and mapping coordinates through the transforms.

    Parameters
    ----------
    image : ndarray
        The image data. Must be 2D or higher.
    axes : tuple | None
        The axis indices of the row and column dimensions in the image data.
        If *image* is 2D, then *axes* defaults to (0, 1).
    cs_name : str | None
        Optional name of the coordinate system to attach to the image.
    graph : str | CoordinateSystemGraph | None
        Optional graph to use for the coordinate system.
    """
    _image_graph_n = 0
    def __init__(self, image, axes: tuple|None = None, cs_name=None, graph=None):
        self.image = image
        self._parent_tr = None
        if axes is None:
            if image.ndim == 2:
                axes = (0, 1)
            else:
                raise ValueError("Must specify (row, col) axes for image data with more than 2 dimensions")
        self.axes = axes

        if graph is None:
            graph = f'image_graph'
            Image._image_graph_n += 1
        self.graph = CoordinateSystemGraph.get_graph(graph, create=True)

        if cs_name is None:
            index = 0
            while True:
                cs_name = f'image_{index}'
                if cs_name not in self.graph.systems:
                    break
                index += 1
        self.cs = self.graph.add_system(cs_name, 2)

    @property
    def shape(self):
        return self.image.shape
    
    @property
    def n_rows(self):
        return self.shape[self.axes[0]]
    
    @property
    def n_cols(self):
        return self.shape[self.axes[1]]

    def point(self, coords):
        """Return a Point object with the given (row, col) coordinates.
        """
        coords = np.asarray(coords)
        if coords.ndim != 1:
            raise ValueError("Point coordinates must be 1D")
        return coorx.Point(coords, system=self.cs)
    
    def point_array(self, coords):
        """Return a PointArray object with the given (row, col) coordinates.
        """
        coords = np.asarray(coords)
        if coords.ndim < 2:
            raise ValueError("coords array must be at least 2D")
        if coords.shape[-1] != 2:
            raise ValueError("coords.shape[-1] must be 2")
        return coorx.PointArray(coords, system=self.cs)

    def rotate(self, angle, **kwds):
        img = self.image
        rotated_img = scipy.ndimage.rotate(img, angle, axes=self.axes, **kwds)
        img2 = self.copy(image=rotated_img)
        shape1 = (self.shape[self.axes[0]], self.shape[self.axes[1]])
        shape2 = (rotated_img.shape[self.axes[0]], rotated_img.shape[self.axes[1]])
        img2._parent_tr = make_rotation_transform(angle, shape1, shape2, from_cs=self.cs, to_cs=img2.cs)
        return img2

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) != 2:
            raise ValueError("Only 2D slices are supported")
        rows, cols = item
        slices = [slice(None)] * self.image.ndim
        slices[self.axes[0]] = rows
        slices[self.axes[1]] = cols
        cropped_img = self.image[tuple(slices)]
        img2 = self.copy(image=cropped_img)
        img2._parent_tr = make_crop_transform((rows, cols), self.image, from_cs=self.cs, to_cs=img2.cs)
        return img2

    def zoom(self, factor, **kwds):
        if np.isscalar(factor):
            factor = [factor, factor]
        img = self.image
        # only zoom row/column axes
        ax_scale_factors = [1] * img.ndim
        ax_scale_factors[self.axes[0]] = factor[0]
        ax_scale_factors[self.axes[1]] = factor[1]
        scaled_img = scipy.ndimage.zoom(img, ax_scale_factors, **kwds)

        img2 = self.copy(image=scaled_img)
        tr = coorx.AffineTransform(dims=(2, 2), from_cs=self.cs, to_cs=img2.cs)
        tr.scale(factor)
        img2._parent_tr = tr
        return img2

    def copy(self, **updates):
        kwds = {'axes': self.axes, 'graph': self.graph}
        kwds.update(updates)
        return Image(**kwds)


# Make transform mapping unrotated to rotated coordinates
def make_rotation_transform(angle, shape1, shape2, **kwds):
    center1 = (np.array(shape1) + 0) / 2
    center2 = (np.array(shape2) + 0) / 2
    tr = coorx.AffineTransform(dims=(2, 2), **kwds)
    tr.translate(-center1)
    tr.rotate(-angle)
    tr.translate(center2)
    return tr


def make_crop_transform(crop, img, **kwds):
    tr = coorx.AffineTransform(dims=(2, 2), **kwds)
    tr.translate([
        -crop[0].indices(img.shape[-2])[0], 
        -crop[1].indices(img.shape[-1])[0], 
    ])
    return tr
