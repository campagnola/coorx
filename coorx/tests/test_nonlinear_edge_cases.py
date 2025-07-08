"""
ABOUTME: Comprehensive edge case and numerical stability tests for nonlinear transforms
ABOUTME: Tests LogTransform, PolarTransform, and LensDistortionTransform with extreme values
"""
import math
import warnings
import time

import numpy as np
import pytest

import coorx
from coorx.nonlinear import LogTransform, PolarTransform, LensDistortionTransform


class TestLogTransformEdgeCases:
    """Test LogTransform with boundary values and extreme cases."""

    def test_log_transform_boundary_bases(self):
        """Test LogTransform with base values at critical boundaries."""
        # Test base = None (should be identity - no transformation)
        lt = LogTransform(base=[None, 2, 10], dims=(3, 3))
        coords = np.array([[1, 4, 100], [2, 8, 1000]])
        result = lt.map(coords)
        
        # First axis should be unchanged (base=None means identity)
        np.testing.assert_array_equal(result[:, 0], coords[:, 0])
        # Other axes should be transformed  
        np.testing.assert_allclose(result[:, 1], [2, 3], rtol=1e-6)  # log_2(4)=2, log_2(8)=3
        np.testing.assert_allclose(result[:, 2], [2, 3], rtol=1e-6)  # log_10(100)=2, log_10(1000)=3

    def test_log_transform_base_one_zero(self):
        """Test LogTransform with base = 1 and base = 0 (nonsense but acceptable)."""
        # Base = 1 and base = 0 should produce nonsense results but not error
        lt = LogTransform(base=[1, 0, 1], dims=(3, 3))
        coords = np.array([[5, 10, 100]])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = lt.map(coords)
        
        # Results will be nonsense but should not raise errors
        # Base = 1 gives log(x)/log(1) = log(x)/0 = inf or -inf
        # Base = 0 gives log(x)/log(0) = log(x)/-inf = 0 or -0
        assert result.shape == coords.shape
        # Don't assert specific values since they're nonsense by design

    def test_log_transform_negative_bases(self):
        """Test LogTransform with negative bases (inverse exponential function)."""
        # Negative bases are supported as inverse: x => -base^x
        lt = LogTransform(base=[-2, -3, -10], dims=(3, 3))
        coords = np.array([[1, 2, 3], [0, 1, 2]])
        result = lt.map(coords)
        
        # For negative bases, should apply inverse exponential: -base^x
        expected = np.array([[2**1, 3**2, 10**3], [2**0, 3**1, 10**2]])
        np.testing.assert_allclose(result, expected)

    def test_log_transform_negative_inputs(self):
        """Test LogTransform with negative input coordinates."""
        lt = LogTransform(base=[2, 10], dims=(2, 2))
        coords = np.array([[-1, -10], [0, 0]])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = lt.map(coords)
        
        # Negative inputs should produce NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        # Zero inputs should produce NaN (log(0) is undefined)
        assert np.isnan(result[1, 0])
        assert np.isnan(result[1, 1])

    def test_log_transform_infinity_nan_inputs(self):
        """Test LogTransform with infinity and NaN inputs."""
        lt = LogTransform(base=[2, 10], dims=(2, 2))
        coords = np.array([[np.inf, np.inf], [np.nan, np.nan], [1e100, 1e-100]])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = lt.map(coords)
        
        # inf inputs produce NaN after log processing (log(inf)/log(base) -> nan)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        # NaN inputs should produce NaN results
        assert np.isnan(result[1, 0])
        assert np.isnan(result[1, 1])
        # Very large/small values should work
        assert np.isfinite(result[2, 0])
        assert np.isfinite(result[2, 1])

    def test_log_transform_extreme_bases(self):
        """Test LogTransform with very large and very small bases."""
        # Very large base
        lt_large = LogTransform(base=[1e10, 1e100], dims=(2, 2))
        coords = np.array([[1e20, 1e200]])
        result_large = lt_large.map(coords)
        assert np.isfinite(result_large).all()
        
        # Very small base (but > 1)
        lt_small = LogTransform(base=[1.0001, 1.001], dims=(2, 2))
        coords = np.array([[2, 10]])
        result_small = lt_small.map(coords)
        assert np.isfinite(result_small).all()
        # Should be very large values due to small base
        assert result_small[0, 0] > 1000
        assert result_small[0, 1] > 1000

    def test_log_transform_round_trip_accuracy(self):
        """Test round-trip accuracy for LogTransform."""
        bases = [[2, 10, math.e], [1.5, 3.7, 2.2]]  # Only bases > 1 for meaningful log transforms
        
        for base in bases:
            lt = LogTransform(base=base, dims=(3, 3))
            
            # Test with various coordinate ranges
            coords_sets = [
                np.array([[1, 1, 1], [2, 10, math.e], [0.5, 0.1, 0.1]]),
                np.array([[100, 1000, 10], [0.01, 0.001, 0.001]])
            ]
            
            for coords in coords_sets:
                if all(b > 1 for b in base):  # Only test positive bases for positive inputs
                    mapped = lt.map(coords)
                    unmapped = lt.imap(mapped)
                    np.testing.assert_allclose(coords, unmapped, rtol=1e-10, atol=1e-15)

    def test_log_transform_none_identity(self):
        """Test LogTransform with None as identity."""
        # None should act as identity transform
        lt = LogTransform(base=[None, None, None], dims=(3, 3))
        coords = np.array([[1, 2, 3], [4, 5, 6], [-1, 0, 100]])
        result = lt.map(coords)
        
        # Should be completely unchanged
        np.testing.assert_array_equal(result, coords)
        
        # Round-trip should also be identity
        back = lt.imap(result)
        np.testing.assert_array_equal(back, coords)

    def test_log_transform_fractional_bases(self):
        """Test LogTransform with fractional bases."""
        # Fractional bases should work normally (no special handling)
        lt = LogTransform(base=[0.5, 1.5, 2.5], dims=(3, 3))
        coords = np.array([[1, 4, 8], [2, 16, 32]])
        result = lt.map(coords)
        
        # Should compute log normally: log(x)/log(base)
        expected = np.array([
            [np.log(1)/np.log(0.5), np.log(4)/np.log(1.5), np.log(8)/np.log(2.5)],
            [np.log(2)/np.log(0.5), np.log(16)/np.log(1.5), np.log(32)/np.log(2.5)]
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_transform_numerical_precision(self):
        """Test LogTransform with extreme values for numerical precision."""
        lt = LogTransform(base=[2], dims=(1, 1))
        
        # Test very close to 1 (where log is nearly 0)
        coords_near_one = np.array([[1 + 1e-15], [1 + 1e-10], [1 - 1e-10]])
        result = lt.map(coords_near_one)
        assert np.isfinite(result).all()
        
        # Test very large values
        coords_large = np.array([[1e308], [1e100]])  # Near float64 limits
        result = lt.map(coords_large)
        assert np.isfinite(result).all()


class TestPolarTransformEdgeCases:
    """Test PolarTransform with critical angles and radius values."""

    def test_polar_transform_pi_multiples(self):
        """Test PolarTransform with theta at multiples of π/2."""
        pt = PolarTransform(dims=(3, 3))
        
        # Test critical angles: 0, π/2, π, 3π/2, 2π
        angles = [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
        radius = 5.0
        
        coords = np.array([[angle, radius, 1] for angle in angles])
        result = pt.map(coords)
        
        # Check expected cartesian coordinates
        expected = np.array([
            [5, 0, 1],      # θ=0: (r, 0)
            [0, 5, 1],      # θ=π/2: (0, r)  
            [-5, 0, 1],     # θ=π: (-r, 0)
            [0, -5, 1],     # θ=3π/2: (0, -r)
            [5, 0, 1]       # θ=2π: (r, 0)
        ])
        
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_polar_transform_negative_radius(self):
        """Test PolarTransform with negative radius values."""
        pt = PolarTransform(dims=(2, 2))
        
        # Negative radius should work but point in opposite direction
        coords = np.array([[0, -5], [math.pi/2, -3], [math.pi, -2]])
        result = pt.map(coords)
        
        expected = np.array([
            [-5, 0],        # θ=0, r=-5: (-5, 0)
            [0, -3],        # θ=π/2, r=-3: (0, -3)
            [2, 0]          # θ=π, r=-2: (2, 0)
        ])
        
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_polar_transform_zero_radius(self):
        """Test PolarTransform with zero radius (singularity at origin)."""
        pt = PolarTransform(dims=(2, 2))
        
        # Zero radius should always map to origin regardless of angle
        coords = np.array([[0, 0], [math.pi/4, 0], [math.pi, 0], [2*math.pi, 0]])
        result = pt.map(coords)
        
        expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_polar_transform_round_trip_accuracy(self):
        """Test round-trip accuracy: polar → cartesian → polar."""
        pt = PolarTransform(dims=(3, 3))
        
        # Test various polar coordinates
        polar_coords = np.array([
            [0, 1, 5],           # θ=0, r=1
            [math.pi/4, 2, 5],   # θ=π/4, r=2
            [math.pi/2, 3, 5],   # θ=π/2, r=3
            [math.pi, 4, 5],     # θ=π, r=4
            [3*math.pi/2, 5, 5], # θ=3π/2, r=5
            [2*math.pi, 1, 5],   # θ=2π, r=1
        ])
        
        cartesian = pt.map(polar_coords)
        polar_back = pt.imap(cartesian)
        
        # Note: angles should be normalized to [-π, π) by arctan2
        # and 2π should map back to 0, 3π/2 should map back to -π/2
        expected = polar_coords.copy()
        expected[-1, 0] = 0  # 2π → 0
        expected[4, 0] = -math.pi/2  # 3π/2 → -π/2
        
        np.testing.assert_allclose(polar_back, expected, rtol=1e-14, atol=1e-14)

    def test_polar_transform_precise_angle_verification(self):
        """Test PolarTransform with small fractions of π to verify arctan2 argument order."""
        pt = PolarTransform(dims=(2, 2))
        
        # Test smaller fractions of π to verify arctan2(y, x) is correct
        test_angles = [
            math.pi/8,     # 22.5°
            math.pi/6,     # 30°
            math.pi/4,     # 45°
            math.pi/3,     # 60°
            3*math.pi/8,   # 67.5°
            5*math.pi/6,   # 150°
            7*math.pi/8,   # 157.5°
            5*math.pi/4,   # 225°
            4*math.pi/3,   # 240°
            7*math.pi/4,   # 315°
        ]
        
        radius = 2.0
        for angle in test_angles:
            # Forward transform: (θ, r) → (x, y)
            polar_coord = np.array([[angle, radius]])
            cartesian = pt.map(polar_coord)
            
            # Verify forward transform: x = r*cos(θ), y = r*sin(θ)
            expected_x = radius * np.cos(angle)
            expected_y = radius * np.sin(angle)
            np.testing.assert_allclose(cartesian[0, 0], expected_x, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(cartesian[0, 1], expected_y, rtol=1e-14, atol=1e-14)
            
            # Inverse transform: (x, y) → (θ, r)
            polar_back = pt.imap(cartesian)
            
            # Verify inverse transform recovers original angle and radius
            # Note: arctan2 returns values in [-π, π], need to normalize to [0, 2π)
            recovered_angle = polar_back[0, 0]
            if recovered_angle < 0:
                recovered_angle += 2 * math.pi
            
            recovered_radius = polar_back[0, 1]
            
            np.testing.assert_allclose(recovered_angle, angle, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(recovered_radius, radius, rtol=1e-14, atol=1e-14)

    def test_polar_transform_origin_stability(self):
        """Test numerical stability near the origin."""
        pt = PolarTransform(dims=(2, 2))
        
        # Points very close to origin
        cartesian_near_origin = np.array([
            [1e-15, 1e-15],
            [1e-10, 1e-10], 
            [-1e-15, 1e-15],
            [0, 0]
        ])
        
        polar = pt.imap(cartesian_near_origin)
        cartesian_back = pt.map(polar)
        
        # Should be stable and return to original coordinates
        np.testing.assert_allclose(cartesian_near_origin, cartesian_back, atol=1e-14)

    def test_polar_transform_quadrant_accuracy(self):
        """Test correct quadrant mapping for all four quadrants."""
        pt = PolarTransform(dims=(2, 2))
        
        # Points in each quadrant
        cartesian_coords = np.array([
            [1, 1],    # Quadrant I
            [-1, 1],   # Quadrant II  
            [-1, -1],  # Quadrant III
            [1, -1]    # Quadrant IV
        ])
        
        polar = pt.imap(cartesian_coords)
        
        # Check angles are in correct quadrants (arctan2 returns [-π, π])
        # Q1: 0 < θ < π/2
        assert 0 <= polar[0, 0] < math.pi/2
        # Q2: π/2 < θ < π
        assert math.pi/2 < polar[1, 0] < math.pi
        # Q3: -π < θ < -π/2 (arctan2 returns negative for Q3)
        assert -math.pi < polar[2, 0] < -math.pi/2
        # Q4: -π/2 < θ < 0 (arctan2 returns negative for Q4)
        assert -math.pi/2 < polar[3, 0] <= 0
        
        # All radii should be sqrt(2)
        np.testing.assert_allclose(polar[:, 1], math.sqrt(2), rtol=1e-14)

    def test_polar_transform_angle_wraparound(self):
        """Test angle wraparound behavior."""
        pt = PolarTransform(dims=(2, 2))
        
        # Test angles outside [0, 2π] range
        coords = np.array([
            [3*math.pi, 1],      # > 2π
            [-math.pi/2, 1],     # < 0
            [5*math.pi, 1],      # >> 2π
            [-3*math.pi, 1]      # << 0
        ])
        
        result = pt.map(coords)
        
        # Should still produce valid cartesian coordinates
        assert np.isfinite(result).all()
        
        # Round-trip should work (though angles may be normalized to [-π, π])
        polar_back = pt.imap(result)
        cartesian_back = pt.map(polar_back)
        np.testing.assert_allclose(result, cartesian_back, rtol=1e-14)


class TestLensDistortionTransformEdgeCases:
    """Test LensDistortionTransform with extreme coefficients and edge cases."""

    def test_lens_distortion_zero_coefficients(self):
        """Test LensDistortionTransform with all zero coefficients (identity)."""
        lt = LensDistortionTransform(coeff=(0, 0, 0, 0, 0))
        
        coords = np.array([[0, 0], [1, 1], [-5, 3], [100, -50]])
        result = lt.map(coords)
        
        # Should be identity transform
        np.testing.assert_array_equal(result, coords)
        
        # Inverse should also be identity
        result_inv = lt.imap(coords)
        np.testing.assert_array_equal(result_inv, coords)

    def test_lens_distortion_extreme_radial_coefficients(self):
        """Test LensDistortionTransform with extreme radial distortion coefficients."""
        # Very high radial distortion
        lt_high = LensDistortionTransform(coeff=(10, 5, 0, 0, 1))
        
        # Test points at various distances from origin
        coords = np.array([[0, 0], [0.1, 0.1], [0.5, 0.5], [1, 1]])
        result = lt_high.map(coords)
        
        # Origin should be unchanged
        np.testing.assert_array_equal(result[0], coords[0])
        
        # Other points should be significantly distorted
        assert not np.allclose(result[1:], coords[1:])
        
        # Results should be finite
        assert np.isfinite(result).all()

    def test_lens_distortion_extreme_tangential_coefficients(self):
        """Test LensDistortionTransform with extreme tangential distortion coefficients."""
        # Very high tangential distortion
        lt_tangential = LensDistortionTransform(coeff=(0, 0, 10, 5, 0))
        
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, -1]])
        result = lt_tangential.map(coords)
        
        # Origin should be unchanged
        np.testing.assert_array_equal(result[0], coords[0])
        
        # Results should be finite
        assert np.isfinite(result).all()

    def test_lens_distortion_inverse_mapping_accuracy(self):
        """Test accuracy of the implemented inverse mapping."""
        # Test with moderate distortion coefficients
        coeffs_list = [
            (0.1, 0.05, 0.01, 0.01, 0.001),  # Moderate distortion
            (-0.1, 0.02, 0.005, -0.005, 0),  # Mixed signs
            (0, 0, 0.1, 0.05, 0)             # Tangential only
        ]
        
        for coeff in coeffs_list:
            lt = LensDistortionTransform(coeff=coeff)
            
            # Test various coordinate sets
            coords_sets = [
                np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5]]),
                np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
                np.array([[0.1, 0.2], [0.3, -0.4], [-0.5, -0.1]])
            ]
            
            for coords in coords_sets:
                # Forward then inverse
                distorted = lt.map(coords)
                undistorted = lt.imap(distorted)
                
                # Should recover original coordinates with reasonable accuracy
                np.testing.assert_allclose(coords, undistorted, rtol=1e-6, atol=1e-8)

    def test_lens_distortion_corner_and_edge_points(self):
        """Test LensDistortionTransform with points at image corners and edges."""
        lt = LensDistortionTransform(coeff=(0.2, 0.1, 0.05, 0.05, 0.01))
        
        # Simulate image corners and edges (normalized coordinates)
        image_points = np.array([
            # Corners
            [-1, -1], [1, -1], [1, 1], [-1, 1],
            # Edge centers
            [0, -1], [1, 0], [0, 1], [-1, 0],
            # Near corners (high distortion region)
            [0.9, 0.9], [-0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]
        ])
        
        result = lt.map(image_points)
        
        # Results should be finite
        assert np.isfinite(result).all()
        
        # Test inverse mapping with relaxed tolerances for high distortion
        undistorted = lt.imap(result)
        np.testing.assert_allclose(image_points, undistorted, rtol=1e-3, atol=1e-5)

    def test_lens_distortion_numerical_stability_high_distortion(self):
        """Test numerical stability with very high distortion scenarios."""
        # Extreme distortion coefficients that might cause numerical issues
        lt_extreme = LensDistortionTransform(coeff=(50, 25, 10, 10, 5))
        
        coords = np.array([[0, 0], [0.01, 0.01], [0.1, 0], [0, 0.1]])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = lt_extreme.map(coords)
        
        # Origin should be unchanged
        np.testing.assert_array_equal(result[0], coords[0])
        
        # Other results may be extreme but should be finite
        # (in practice, such extreme distortion is unrealistic)
        assert not np.any(np.isnan(result))

    def test_lens_distortion_performance_benchmark(self):
        """Benchmark performance of lens distortion transforms."""
        lt = LensDistortionTransform(coeff=(0.1, 0.05, 0.01, 0.01, 0.001))
        
        # Generate large coordinate array
        n_points = 10000
        coords = np.random.randn(n_points, 2) * 0.5  # Random points in reasonable range
        
        # Time forward mapping
        start_time = time.time()
        result_forward = lt.map(coords)
        forward_time = time.time() - start_time
        
        # Time inverse mapping
        start_time = time.time()
        result_inverse = lt.imap(result_forward[:100])  # Test smaller set for inverse
        inverse_time = time.time() - start_time
        
        # Basic performance check (not strict timing requirements)
        assert forward_time < 1.0  # Should complete in reasonable time
        assert inverse_time < 5.0  # Inverse is more expensive due to iteration
        
        # Verify results are reasonable
        assert np.isfinite(result_forward).all()
        assert np.isfinite(result_inverse).all()


class TestNonlinearTransformComposition:
    """Test composition of multiple nonlinear transforms."""

    def test_composite_nonlinear_accuracy(self):
        """Test accuracy of composite transforms with multiple nonlinear transforms."""
        # Create a composite of different nonlinear transforms
        log_transform = LogTransform(base=[2, 10], dims=(2, 2))
        polar_transform = PolarTransform(dims=(2, 2))
        
        # Note: Direct composition of these may not be mathematically meaningful
        # but we test the mechanism works
        
        # Test individual transforms first
        coords = np.array([[2, 10], [4, 100]])
        
        # Apply transforms in sequence
        step1 = log_transform.map(coords)  # Log transform
        step2 = polar_transform.map(step1)  # Then polar
        
        # Reverse the sequence
        rev_step1 = polar_transform.imap(step2)
        rev_step2 = log_transform.imap(rev_step1)
        
        # Should recover original coordinates with relaxed tolerance
        np.testing.assert_allclose(coords, rev_step2, rtol=1e-6)

    def test_nonlinear_as_affine_error(self):
        """Test that nonlinear transforms correctly raise NotImplementedError for as_affine."""
        transforms = [
            LogTransform(base=[2, 10], dims=(2, 2)),
            PolarTransform(dims=(2, 2)),
            LensDistortionTransform(coeff=(0.1, 0.05, 0.01, 0.01, 0.001))
        ]
        
        for transform in transforms:
            with pytest.raises(NotImplementedError):
                transform.as_affine()

    def test_nonlinear_composite_with_linear(self):
        """Test composition of nonlinear transforms with linear transforms."""
        # Create a mixed composite
        linear_transform = coorx.STTransform(scale=[2, 3], offset=[1, -1], dims=(2, 2))
        polar_transform = PolarTransform(dims=(2, 2))
        
        coords = np.array([[1, 0], [0, 1], [1, 1]])
        
        # Linear then nonlinear
        step1 = linear_transform.map(coords)
        step2 = polar_transform.map(step1)
        
        # Reverse
        rev_step1 = polar_transform.imap(step2)
        rev_step2 = linear_transform.imap(rev_step1)
        
        np.testing.assert_allclose(coords, rev_step2, rtol=1e-12)

    @pytest.mark.parametrize("precision", [np.float32, np.float64])
    def test_nonlinear_precision_handling(self, precision):
        """Test nonlinear transforms with different floating-point precisions."""
        # Test with different precisions, use positive values suitable for log transform
        coords = np.array([[1, 2], [3, 4]], dtype=precision)
        
        transforms = [
            LogTransform(base=[2, 10], dims=(2, 2)),
            PolarTransform(dims=(2, 2))
        ]
        
        for transform in transforms:
            result = transform.map(coords)
            
            # Result should maintain input precision when possible
            assert result.dtype == precision or result.dtype == np.float64
            
            # Round-trip test with appropriate tolerance for precision
            rtol = 1e-5 if precision == np.float32 else 1e-12
            back = transform.imap(result)
            np.testing.assert_allclose(coords, back, rtol=rtol)

    def test_nonlinear_transform_state_saving(self):
        """Test that nonlinear transforms properly save and restore state."""
        # Test LogTransform state
        log_transform = LogTransform(base=[2, 3, 5], dims=(3, 3))
        state = log_transform.save_state()
        
        # Create new transform from state
        log_transform2 = coorx.create_transform(**state)
        assert isinstance(log_transform2, LogTransform)
        np.testing.assert_array_equal(log_transform.base, log_transform2.base)
        
        # Test LensDistortionTransform state
        lens_transform = LensDistortionTransform(coeff=(0.1, 0.2, 0.3, 0.4, 0.5))
        state = lens_transform.save_state()
        
        lens_transform2 = coorx.create_transform(**state)
        assert isinstance(lens_transform2, LensDistortionTransform)
        assert lens_transform.coeff == lens_transform2.coeff