import pytest
import numpy as np
from coorx.image import Image


class TestImageSlicing:
    """Test Image.__getitem__ slice validation."""

    def test_valid_slice_operations(self):
        """Test that valid slice operations work correctly."""
        img_data = np.ones((100, 100, 3))
        img = Image(img_data, axes=(0, 1))
        
        # Single slice - only spatial dimensions count for shape
        result = img[slice(10, 20)]
        assert result.spatial_shape == (10, 100)  # Only spatial axes in shape
        assert result.image.shape == (10, 100, 3)  # Full image shape
        
        # Multiple slices
        result = img[slice(10, 20), slice(30, 40)]
        assert result.spatial_shape == (10, 10)  # Only spatial axes in shape
        assert result.image.shape == (10, 10, 3)  # Full image shape
        np.testing.assert_array_equal(result.image, img_data[10:20, 30:40])

    def test_invalid_non_slice_indexing_raises_error(self):
        """Test that non-slice indexing raises ValueError."""
        img_data = np.ones((100, 100, 3))
        img = Image(img_data, axes=(0, 1))
        
        # Single integer index
        with pytest.raises(ValueError, match="Image.__getitem__ requires a tuple of slices"):
            img[10]
        
        # Mixed slice and integer
        with pytest.raises(ValueError, match="Image.__getitem__ requires a tuple of slices"):
            img[slice(10, 20), 30]
        
        # Multiple integers
        with pytest.raises(ValueError, match="Image.__getitem__ requires a tuple of slices"):
            img[10, 20]
        
        # List indexing
        with pytest.raises(ValueError, match="Image.__getitem__ requires a tuple of slices"):
            img[[10, 20, 30]]

    def test_error_message_includes_actual_item(self):
        """Test that error message includes the actual item that caused the error."""
        img_data = np.ones((100, 100, 3))
        img = Image(img_data, axes=(0, 1))
        
        with pytest.raises(ValueError, match=r"got \(10, 20\)"):
            img[10, 20]
        
        with pytest.raises(ValueError, match=r"got \(slice\(10, 20, None\), 30\)"):
            img[slice(10, 20), 30]

    def test_slice_padding_still_works(self):
        """Test that slice padding with slice(None) still works correctly."""
        img_data = np.ones((100, 100, 3))
        img = Image(img_data, axes=(0, 1))
        
        # Should pad with slice(None) for remaining dimensions
        result = img[slice(10, 20)]
        assert result.spatial_shape == (10, 100)  # Only spatial axes in shape
        assert result.image.shape == (10, 100, 3)  # Full image shape
        
        # Should work with explicit padding of the remaining spatial axis
        result = img[slice(10, 20), slice(None)]
        assert result.spatial_shape == (10, 100)  # Only spatial axes in shape
        assert result.image.shape == (10, 100, 3)  # Full image shape

    def test_too_many_slices_raises_error(self):
        """Test that more slices than spatial dimensions raises ValueError.

        Non-spatial axes (here, the color channel) are never sliced, so a slice
        for each *image* axis is not a valid index.
        """
        img_data = np.ones((100, 100, 3))
        img = Image(img_data, axes=(0, 1))

        with pytest.raises(ValueError, match="at most one slice per spatial dimension"):
            img[slice(10, 20), slice(30, 40), slice(None)]

    def test_3d_image_slicing(self):
        """Test slicing works correctly with 3D images."""
        img_data = np.ones((10, 100, 100))
        img = Image(img_data, axes=(0, 1, 2))
        
        # Valid 3D slicing
        result = img[slice(2, 4), slice(10, 20), slice(30, 40)]
        assert result.spatial_shape == (2, 10, 10)
        
        # Invalid 3D slicing with integers
        with pytest.raises(ValueError, match="Image.__getitem__ requires a tuple of slices"):
            img[2, slice(10, 20), slice(30, 40)]


class TestImageSlicingStride:
    """Tests for strided slices in Image.__getitem__."""

    def test_stride_shape(self):
        """Strided slice halves the spatial shape."""
        img = Image(np.arange(100 * 100).reshape(100, 100))
        strided = img[::2, ::2]
        assert strided.spatial_shape == (50, 50)

    def test_stride_point_mapping(self):
        """A point in the original maps to (coord / step) in a stride-from-zero slice."""
        img = Image(np.arange(100 * 100).reshape(100, 100))
        strided = img[::2, ::2]
        # pixel [40, 60] in original → [20, 30] in strided image
        pt = img.point([40, 60]).mapped_to(strided.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [20, 30])

    def test_stride_with_offset_point_mapping(self):
        """A point maps correctly when the slice has both a start and a step."""
        img = Image(np.arange(100 * 100).reshape(100, 100))
        strided = img[10:90:2, 10:90:2]
        # pixel [50, 50] in original → (50 - 10) / 2 = 20 in each axis
        pt = img.point([50, 50]).mapped_to(strided.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [20, 20])


class TestCropAround:
    """Tests for Image.crop_around."""

    def _make_img(self, shape=(100, 100)):
        return Image(np.arange(np.prod(shape)).reshape(shape))

    # --- shape / pixel content ---

    def test_crop_center_shape_and_content(self):
        """Interior crop (no boundary clamping) produces correct shape and data."""
        img = self._make_img()
        cropped = img.crop_around([50, 50], 40)
        # start = floor(50 - 20) = 30, stop = ceil(50 + 20) = 70
        assert cropped.spatial_shape == (40, 40)
        np.testing.assert_array_equal(cropped.image, img.image[30:70, 30:70])

    def test_crop_edge_shape_and_content(self):
        """Crop near the edge clamps to image boundary and gives smaller result."""
        img = self._make_img()
        cropped = img.crop_around([5, 5], 40)
        # start = max(0, floor(5 - 20)) = 0, stop = min(100, ceil(5 + 20)) = 25
        assert cropped.spatial_shape == (25, 25)
        np.testing.assert_array_equal(cropped.image, img.image[0:25, 0:25])

    def test_crop_per_axis_size(self):
        """Different sizes per axis are handled independently."""
        img = self._make_img()
        cropped = img.crop_around([50, 50], [20, 40])
        # rows: 40-60 (20 px), cols: 30-70 (40 px)
        assert cropped.spatial_shape == (20, 40)
        np.testing.assert_array_equal(cropped.image, img.image[40:60, 30:70])

    # --- coordinate mapping ---

    def test_point_mapping_center_crop(self):
        """Points in the original image map correctly into an interior crop."""
        img = self._make_img()
        cropped = img.crop_around([50, 50], 40)  # slices [30:70, 30:70]

        # pixel [50, 50] in img → [50-30, 50-30] = [20, 20] in cropped
        pt_in_cropped = img.point([50, 50]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt_in_cropped.coordinates, [20, 20])

        # corner of the crop window maps to [0, 0]
        pt_corner = img.point([30, 30]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt_corner.coordinates, [0, 0])

    def test_point_mapping_edge_crop(self):
        """Points map correctly when the crop is clamped to the image boundary."""
        img = self._make_img()
        cropped = img.crop_around([5, 5], 40)  # slices [0:25, 0:25]

        # start == 0 so the coordinate offset is zero
        pt = img.point([5, 5]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [5, 5])

        pt_origin = img.point([0, 0]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt_origin.coordinates, [0, 0])

    # --- Point object as center ---

    def test_point_object_as_center(self):
        """crop_around accepts a Point object; it is mapped into the image's system."""
        img = self._make_img()
        center_pt = img.point([50, 50])
        cropped = img.crop_around(center_pt, 40)
        assert cropped.spatial_shape == (40, 40)
        np.testing.assert_array_equal(cropped.image, img.image[30:70, 30:70])

    # --- odd-size / stop-overshoot regression ---

    def test_odd_size_interior_shape(self):
        """Odd size at an integer center must not produce one extra pixel.

        With the old code, ceil(center + size/2) rounded up when size is odd,
        giving stop - start = size + 1 instead of size.
        """
        img = self._make_img()
        cropped = img.crop_around([50, 50], 11)
        # start = floor(50 - 5.5) = 44, stop must be 55 (not 56)
        assert cropped.spatial_shape == (11, 11)
        np.testing.assert_array_equal(cropped.image, img.image[44:55, 44:55])

    def test_odd_size_per_axis_shape(self):
        """Per-axis odd sizes are each clamped to exactly the requested count."""
        img = self._make_img()
        cropped = img.crop_around([50, 50], [11, 13])
        assert cropped.spatial_shape == (11, 13)


class TestNonDefaultAxes:
    """Cropping an Image whose spatial axes are not the leading image axes.

    Regression tests: the crop slices are given in spatial order, but were
    applied (and measured) against image axes, so any `axes` other than
    ``range(ndim)`` cropped the wrong axes or raised IndexError.
    """

    def _make_stack(self):
        """A 2-frame stack: axis 0 is the frame index, axes 1 and 2 are spatial."""
        data = np.arange(2 * 64 * 64).reshape(2, 64, 64)
        return Image(data, axes=(1, 2))

    # --- __getitem__ ---

    def test_getitem_shape_and_content(self):
        """Spatial slices apply to the spatial axes, leaving the frame axis whole."""
        img = self._make_stack()
        cropped = img[10:50, 8:40]
        assert cropped.spatial_shape == (40, 32)
        assert cropped.image.shape == (2, 40, 32)
        np.testing.assert_array_equal(cropped.image, img.image[:, 10:50, 8:40])

    def test_getitem_point_mapping(self):
        """Points map into the crop by the spatial starts, not the image starts."""
        img = self._make_stack()
        cropped = img[10:50, 8:40]
        pt = img.point([30, 20]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [20, 12])

    def test_getitem_omitted_trailing_slice(self):
        """A single slice crops the first spatial axis and leaves the rest whole."""
        img = self._make_stack()
        cropped = img[10:50]
        assert cropped.image.shape == (2, 40, 64)
        np.testing.assert_array_equal(cropped.image, img.image[:, 10:50, :])
        pt = img.point([30, 20]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [20, 20])

    def test_getitem_strided(self):
        """Strided slices scale as well as offset, still on the spatial axes."""
        img = self._make_stack()
        strided = img[10:50:2, ::2]
        assert strided.spatial_shape == (20, 32)
        np.testing.assert_array_equal(strided.image, img.image[:, 10:50:2, ::2])
        # pixel [30, 20] → [(30 - 10) / 2, 20 / 2]
        pt = img.point([30, 20]).mapped_to(strided.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [10, 10])

    def test_getitem_non_contiguous_axes(self):
        """Spatial axes need not be adjacent."""
        data = np.arange(4 * 6 * 8).reshape(4, 6, 8)
        img = Image(data, axes=(0, 2))
        cropped = img[1:3, 2:6]
        assert cropped.spatial_shape == (2, 4)
        assert cropped.image.shape == (2, 6, 4)
        np.testing.assert_array_equal(cropped.image, data[1:3, :, 2:6])
        pt = img.point([2, 4]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [1, 2])

    # --- crop_around ---

    def test_crop_around_shape_and_content(self):
        """crop_around picks the spatial window around the given center."""
        img = self._make_stack()
        cropped = img.crop_around([32, 32], 32)
        # start = floor(32 - 16) = 16, stop = 48 on each spatial axis
        assert cropped.spatial_shape == (32, 32)
        assert cropped.image.shape == (2, 32, 32)
        np.testing.assert_array_equal(cropped.image, img.image[:, 16:48, 16:48])

    def test_crop_around_point_mapping(self):
        """The crop center maps to the center of the cropped image."""
        img = self._make_stack()
        cropped = img.crop_around([32, 32], 32)
        pt = img.point([32, 32]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [16, 16])

    def test_crop_around_non_contiguous_axes(self):
        """crop_around clamps per spatial axis using that axis' image size."""
        data = np.arange(4 * 6 * 8).reshape(4, 6, 8)
        img = Image(data, axes=(0, 2))
        cropped = img.crop_around([2, 4], [2, 4])
        assert cropped.spatial_shape == (2, 4)
        np.testing.assert_array_equal(cropped.image, data[1:3, :, 2:6])
        pt = img.point([2, 4]).mapped_to(cropped.system)
        np.testing.assert_array_almost_equal(pt.coordinates, [1, 2])
