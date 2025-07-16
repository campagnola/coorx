# ABOUTME: Tests for the Image class functionality including slicing validation
# ABOUTME: Covers the __getitem__ method and its slice validation requirements

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
        
        # Full slice tuple
        result = img[slice(10, 20), slice(30, 40), slice(None)]
        assert result.spatial_shape == (10, 10)  # Only spatial axes in shape
        assert result.image.shape == (10, 10, 3)  # Full image shape

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
        
        # Should work with explicit padding
        result = img[slice(10, 20), slice(None), slice(None)]
        assert result.spatial_shape == (10, 100)  # Only spatial axes in shape
        assert result.image.shape == (10, 100, 3)  # Full image shape

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