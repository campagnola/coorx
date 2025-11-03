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
        
        assert img1.axes == img2.axes == axes

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
        new_axes = (0, 1)  # Different from default if different input
        
        img1 = Image(sample_image_data)
        img2 = img1.copy(image=new_data, axes=new_axes)
        
        np.testing.assert_array_equal(img2.image, new_data)
        assert img2.axes == new_axes
        assert img1.axes != img2.axes or not np.array_equal(img1.image, img2.image)

    def test_copy_preserves_shape_properties(self, sample_image_data):
        """Test that copy preserves shape-related properties."""
        img1 = Image(sample_image_data)
        img2 = img1.copy()
        
        assert img1.shape == img2.shape
        assert img1.n_rows == img2.n_rows
        assert img1.n_cols == img2.n_cols


class TestImageCopyIndependence:
    """Test that Image.copy() creates truly independent instances."""

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        return np.random.rand(10, 15)

    def test_copy_creates_independent_coordinate_system(self, sample_image_data):
        """Test that copy creates independent coordinate system."""
        img1 = Image(sample_image_data, cs_name='img1')
        img2 = img1.copy()
        
        # Should have different coordinate systems
        assert img1.system is not img2.system
        assert img1.system.name != img2.system.name
        
        # But should be in the same graph
        assert img1.graph is img2.graph

    def test_copy_gets_unique_system_name(self, sample_image_data):
        """Test that copied images get unique coordinate system names."""
        img1 = Image(sample_image_data)
        img2 = img1.copy()
        img3 = img1.copy()
        
        # All should have different system names
        system_names = {img1.system.name, img2.system.name, img3.system.name}
        assert len(system_names) == 3
        
        # All should follow the pattern 'image_N'
        for name in system_names:
            assert name.startswith('image_')

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
        img1 = Image(sample_image_data, cs_name='img1')
        img2 = img1.copy()
        
        # Create a point in img1's coordinate system
        pt1 = img1.point([5, 7])
        
        # Trying to map to img2's system should fail (no transform path)
        with pytest.raises(Exception):  # Could be various exception types
            pt1.mapped_to(img2.system)

    def test_copy_preserves_parent_transform_isolation(self, sample_image_data):
        """Test that copied image's parent transform is independent."""
        img1 = Image(sample_image_data)
        img2 = img1.copy()
        
        # Initially, both should have None parent transform
        assert img1._parent_tr is None
        assert img2._parent_tr is None
        
        # Rotate img1 (creates parent transform)
        img1_rotated = img1.rotate(45)
        
        # img2 should still have None parent transform
        assert img2._parent_tr is None
        assert img1_rotated._parent_tr is not None

    def test_copy_chain_independence(self, sample_image_data):
        """Test independence in a chain of copies."""
        img1 = Image(sample_image_data, cs_name='original')
        img2 = img1.copy()
        img3 = img2.copy()
        
        # All should have different coordinate systems
        systems = [img1.system, img2.system, img3.system]
        system_ids = [id(sys) for sys in systems]
        assert len(set(system_ids)) == 3
        
        # Create points in each system
        pt1 = img1.point([1, 2])
        pt2 = img2.point([3, 4])
        pt3 = img3.point([5, 6])
        
        # None should be mappable to the others
        with pytest.raises(Exception):
            pt1.mapped_to(img2.system)
        with pytest.raises(Exception):
            pt1.mapped_to(img3.system)
        with pytest.raises(Exception):
            pt2.mapped_to(img3.system)


class TestImageCopyEdgeCases:
    """Test edge cases and integration scenarios for Image.copy()."""

    def test_copy_with_custom_cs_name_override(self):
        """Test copy with custom coordinate system name override."""
        img_data = np.ones((5, 5))
        img1 = Image(img_data, cs_name='original')
        img2 = img1.copy(cs_name='custom_copy')
        
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

    def test_copy_preserves_image_reference_semantics(self):
        """Test that copy handles image reference correctly."""
        original_data = np.ones((5, 5))
        img1 = Image(original_data)
        
        # Copy without providing new image data
        img2 = img1.copy()
        
        # Should be independent copies when no image provided
        assert img1.image is not img2.image
        # But data content should be identical
        np.testing.assert_array_equal(img1.image, original_data)
        np.testing.assert_array_equal(img2.image, original_data)

    def test_copy_with_new_image_creates_new_reference(self):
        """Test that copy with new image creates proper reference."""
        original_data = np.ones((5, 5))
        new_data = np.zeros((3, 3))
        
        img1 = Image(original_data)
        img2 = img1.copy(image=new_data)
        
        # Should have different image data
        assert img1.image is not img2.image
        assert img1.image is original_data
        assert img2.image is new_data

    @pytest.mark.parametrize("axes", [(0, 1), (1, 2), (0, 2)])
    def test_copy_with_different_axes_configurations(self, axes):
        """Test copy behavior with different axes configurations."""
        img_data = np.random.rand(5, 10, 15)
        img1 = Image(img_data, axes=axes)
        img2 = img1.copy()
        
        assert img1.axes == img2.axes == axes
        # Shape properties should match
        assert img1.n_rows == img2.n_rows
        assert img1.n_cols == img2.n_cols


class TestImageCopyIntegration:
    """Integration tests for Image.copy() with other Image methods."""

    def test_copy_integration_with_transforms(self):
        """Test that copy works correctly with image transforms."""
        img_data = np.random.rand(20, 20)
        img1 = Image(img_data)
        
        # Create a chain of transforms
        img2_rotated = img1.rotate(30)
        img3_cropped = img2_rotated[5:15, 5:15]
        img4_zoomed = img3_cropped.zoom(2.0)
        
        # Each should have independent coordinate systems
        systems = [img1.system, img2_rotated.system, img3_cropped.system, img4_zoomed.system]
        system_ids = [id(sys) for sys in systems]
        assert len(set(system_ids)) == 4
        
        # But points should be mappable through the transform chain
        pt1 = img1.point([10, 10])
        pt4 = pt1.mapped_to(img4_zoomed.system)
        assert pt4 is not None  # Should succeed

    def test_copy_doesnt_interfere_with_existing_transforms(self):
        """Test that copying doesn't interfere with existing transforms."""
        img_data = np.random.rand(10, 10)
        img1 = Image(img_data, cs_name='original_for_transform_test')
        
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
