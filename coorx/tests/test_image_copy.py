"""
ABOUTME: Comprehensive tests for Image.copy() method
ABOUTME: Tests metadata preservation and coordinate system independence
"""
import numpy as np
import pytest

from coorx.image import Image
from coorx.systems import CoordinateSystemGraph


@pytest.fixture(autouse=True)
def fresh_graph():
    """Ensure each test gets a fresh coordinate system graph."""
    # Store original graphs
    original_graphs = CoordinateSystemGraph.all_graphs.copy()
    # Clear existing graphs to avoid naming conflicts
    CoordinateSystemGraph.all_graphs.clear()
    yield
    # Restore original graphs
    CoordinateSystemGraph.all_graphs.clear()
    CoordinateSystemGraph.all_graphs.update(original_graphs)


class TestImageCopyMetadataPreservation:
    """Test that Image.copy() preserves expected metadata correctly."""

    @pytest.fixture
    def sample_image_data(self):
        """Create sample 2D image data for testing."""
        return np.random.rand(10, 15)

    @pytest.fixture
    def sample_3d_image_data(self):
        """Create sample 3D image data for testing."""
        return np.random.rand(5, 10, 15)

    def test_copy_preserves_image_data(self, sample_image_data):
        """Test that copy preserves image data correctly."""
        img1 = Image(sample_image_data)
        img2 = img1.copy()
        
        # Image data should be identical
        np.testing.assert_array_equal(img1.image, img2.image)
        
        # Should be independent copies, not sharing the same object
        assert img1.image is not img2.image

    def test_copy_preserves_axes_configuration(self, sample_3d_image_data):
        """Test that copy preserves axes configuration."""
        axes = (1, 2)  # Non-default axes
        img1 = Image(sample_3d_image_data, axes=axes)
        img2 = img1.copy()

        assert tuple(img1.spatial_to_image_axes) == tuple(img2.spatial_to_image_axes) == tuple(axes)

    def test_copy_preserves_graph_reference(self, sample_image_data):
        """Test that copy preserves graph reference."""
        graph_name = 'test_graph'
        img1 = Image(sample_image_data, graph=graph_name)
        img2 = img1.copy()
        
        # Should share the same graph object
        assert img1.graph is img2.graph
        assert img1.graph.name == img2.graph.name == graph_name

    def test_copy_allows_parameter_overrides(self, sample_image_data):
        """Test that copy allows parameter overrides."""
        new_data = np.ones((5, 5))
        new_axes = (1, 0)
        
        img1 = Image(sample_image_data)
        img2 = img1.copy(image=new_data, axes=new_axes)
        
        np.testing.assert_array_equal(img2.image, new_data)
        assert tuple(img2.spatial_to_image_axes) == tuple(new_axes)
        assert tuple(img1.spatial_to_image_axes) != tuple(img2.spatial_to_image_axes) or not np.array_equal(img1.image, img2.image)


class TestImageCopyIndependence:
    """Test that Image.copy() creates truly independent instances."""

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        return np.random.rand(10, 15)

    def test_copy_creates_independent_coordinate_system(self, sample_image_data):
        """Test that copy creates independent coordinate system."""
        img1 = Image(sample_image_data, system='img1', graph='independence')
        img2 = img1.copy()
        
        # Should have different coordinate systems
        assert img1.system is not img2.system
        assert img1.system.name != img2.system.name
        
        # But should be in the same graph
        assert img1.graph is img2.graph

    def test_copy_independence_data_modification(self, sample_image_data):
        """Test that copied images have independent image data."""
        img1 = Image(sample_image_data.copy())  # Copy to avoid modifying fixture
        img2 = img1.copy()
        
        # Images should be independent copies
        assert img1.image is not img2.image
        
        # Modifying one should not affect the other (independent copies)
        original_value = img1.image[0, 0]
        img2.image[0, 0] = 999.0
        assert img1.image[0, 0] == original_value
        
        # But when new image is provided, they're independent
        new_data = np.zeros_like(sample_image_data)
        img3 = img1.copy(image=new_data)
        assert img1.image is not img3.image

    def test_copy_independence_coordinate_mapping_failure(self, sample_image_data):
        """Test that points cannot be mapped between original and copy."""
        img1 = Image(sample_image_data, system='img1', graph='independence')
        img2 = img1.copy()
        
        # Create a point in img1's coordinate system
        pt1 = img1.point([5, 7])
        
        # Trying to map to img2's system should fail (no transform path)
        with pytest.raises(Exception):  # Could be various exception types
            pt1.mapped_to(img2.system)


class TestImageCopyEdgeCases:
    """Test edge cases and integration scenarios for Image.copy()."""

    def test_copy_with_custom_system_override(self):
        """Test copy with custom coordinate system name override."""
        img_data = np.ones((5, 5))
        img1 = Image(img_data, system='original', graph='custom-system')
        img2 = img1.copy(system='custom_copy')
        
        assert img1.system.name == 'original'
        assert img2.system.name == 'custom_copy'

    def test_copy_with_different_graph(self):
        """Test copy with different graph."""
        img_data = np.ones((5, 5))
        
        # Create first image in one graph
        img1 = Image(img_data, graph='graph1')
        
        # Copy to different graph
        img2 = img1.copy(graph='graph2')
        
        assert img1.graph is not img2.graph
        assert img1.graph.name != img2.graph.name

    @pytest.mark.parametrize("axes", [(0, 1), (1, 2), (0, 2)])
    def test_copy_with_different_axes_configurations(self, axes):
        """Test copy behavior with different axes configurations."""
        img_data = np.random.rand(5, 10, 15)
        img1 = Image(img_data, axes=axes)
        img2 = img1.copy()

        assert tuple(img1.spatial_to_image_axes) == tuple(img2.spatial_to_image_axes) == tuple(axes)
        # Shape properties should match
        assert img1.spatial_shape == img2.spatial_shape

    def test_copy_doesnt_interfere_with_existing_transforms(self):
        """Test that copying doesn't interfere with existing transforms."""
        img_data = np.random.rand(10, 10)
        img1 = Image(img_data, system='original_for_transform_test', graph='integration')
        
        # Create a transform chain
        img2 = img1.rotate(45)
        
        # Make a copy of img1 after transform is created
        img1_copy = img1.copy()
        
        # Original transform should still work
        pt1 = img1.point([5, 5])
        pt2 = pt1.mapped_to(img2.system)
        assert pt2 is not None
        
        # But copy should be independent
        pt1_copy = img1_copy.point([5, 5])
        with pytest.raises(Exception):
            pt1_copy.mapped_to(img2.system)
