"""
Comprehensive edge case and numerical stability tests for nonlinear transforms
Tests LogTransform, PolarTransform, and LensDistortionTransform with extreme values
"""
import math
import warnings
import time

import numpy as np
import pytest

import coorx
from coorx.nonlinear import LogTransform, PolarTransform, LensDistortionTransform, SphericalTransform, MercatorSphericalTransform, LambertAzimuthalEqualAreaTransform


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
        np.testing.assert_allclose(
            result[:, 1], [2, 3], rtol=1e-6
        )  # log_2(4)=2, log_2(8)=3
        np.testing.assert_allclose(
            result[:, 2], [2, 3], rtol=1e-6
        )  # log_10(100)=2, log_10(1000)=3

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
        bases = [
            [2, 10, math.e],
            [1.5, 3.7, 2.2],
        ]  # Only bases > 1 for meaningful log transforms

        for base in bases:
            lt = LogTransform(base=base, dims=(3, 3))

            # Test with various coordinate ranges
            coords_sets = [
                np.array([[1, 1, 1], [2, 10, math.e], [0.5, 0.1, 0.1]]),
                np.array([[100, 1000, 10], [0.01, 0.001, 0.001]]),
            ]

            for coords in coords_sets:
                if all(
                    b > 1 for b in base
                ):  # Only test positive bases for positive inputs
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
        expected = np.array(
            [
                [
                    np.log(1) / np.log(0.5),
                    np.log(4) / np.log(1.5),
                    np.log(8) / np.log(2.5),
                ],
                [
                    np.log(2) / np.log(0.5),
                    np.log(16) / np.log(1.5),
                    np.log(32) / np.log(2.5),
                ],
            ]
        )
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
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radius = 5.0

        coords = np.array([[angle, radius, 1] for angle in angles])
        result = pt.map(coords)

        # Check expected cartesian coordinates
        expected = np.array(
            [
                [5, 0, 1],  # θ=0: (r, 0)
                [0, 5, 1],  # θ=π/2: (0, r)
                [-5, 0, 1],  # θ=π: (-r, 0)
                [0, -5, 1],  # θ=3π/2: (0, -r)
                [5, 0, 1],  # θ=2π: (r, 0)
            ]
        )

        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_polar_transform_negative_radius(self):
        """Test PolarTransform with negative radius values."""
        pt = PolarTransform(dims=(2, 2))

        # Negative radius should work but point in opposite direction
        coords = np.array([[0, -5], [math.pi / 2, -3], [math.pi, -2]])
        result = pt.map(coords)

        expected = np.array(
            [
                [-5, 0],  # θ=0, r=-5: (-5, 0)
                [0, -3],  # θ=π/2, r=-3: (0, -3)
                [2, 0],  # θ=π, r=-2: (2, 0)
            ]
        )

        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_polar_transform_zero_radius(self):
        """Test PolarTransform with zero radius (singularity at origin)."""
        pt = PolarTransform(dims=(2, 2))

        # Zero radius should always map to origin regardless of angle
        coords = np.array([[0, 0], [math.pi / 4, 0], [math.pi, 0], [2 * math.pi, 0]])
        result = pt.map(coords)

        expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_polar_transform_round_trip_accuracy(self):
        """Test round-trip accuracy: polar → cartesian → polar."""
        pt = PolarTransform(dims=(3, 3))

        # Test various polar coordinates
        polar_coords = np.array(
            [
                [0, 1, 5],  # θ=0, r=1
                [math.pi / 4, 2, 5],  # θ=π/4, r=2
                [math.pi / 2, 3, 5],  # θ=π/2, r=3
                [math.pi, 4, 5],  # θ=π, r=4
                [3 * math.pi / 2, 5, 5],  # θ=3π/2, r=5
                [2 * math.pi, 1, 5],  # θ=2π, r=1
            ]
        )

        cartesian = pt.map(polar_coords)
        polar_back = pt.imap(cartesian)

        # Note: angles should be normalized to [-π, π) by arctan2
        # and 2π should map back to 0, 3π/2 should map back to -π/2
        expected = polar_coords.copy()
        expected[-1, 0] = 0  # 2π → 0
        expected[4, 0] = -math.pi / 2  # 3π/2 → -π/2

        np.testing.assert_allclose(polar_back, expected, rtol=1e-14, atol=1e-14)

    def test_polar_transform_precise_angle_verification(self):
        """Test PolarTransform with small fractions of π to verify arctan2 argument order."""
        pt = PolarTransform(dims=(2, 2))

        # Test smaller fractions of π to verify arctan2(y, x) is correct
        test_angles = [
            math.pi / 8,  # 22.5°
            math.pi / 6,  # 30°
            math.pi / 4,  # 45°
            math.pi / 3,  # 60°
            3 * math.pi / 8,  # 67.5°
            5 * math.pi / 6,  # 150°
            7 * math.pi / 8,  # 157.5°
            5 * math.pi / 4,  # 225°
            4 * math.pi / 3,  # 240°
            7 * math.pi / 4,  # 315°
        ]

        radius = 2.0
        for angle in test_angles:
            # Forward transform: (θ, r) → (x, y)
            polar_coord = np.array([[angle, radius]])
            cartesian = pt.map(polar_coord)

            # Verify forward transform: x = r*cos(θ), y = r*sin(θ)
            expected_x = radius * np.cos(angle)
            expected_y = radius * np.sin(angle)
            np.testing.assert_allclose(
                cartesian[0, 0], expected_x, rtol=1e-14, atol=1e-14
            )
            np.testing.assert_allclose(
                cartesian[0, 1], expected_y, rtol=1e-14, atol=1e-14
            )

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
        cartesian_near_origin = np.array(
            [[1e-15, 1e-15], [1e-10, 1e-10], [-1e-15, 1e-15], [0, 0]]
        )

        polar = pt.imap(cartesian_near_origin)
        cartesian_back = pt.map(polar)

        # Should be stable and return to original coordinates
        np.testing.assert_allclose(cartesian_near_origin, cartesian_back, atol=1e-14)

    def test_polar_transform_quadrant_accuracy(self):
        """Test correct quadrant mapping for all four quadrants."""
        pt = PolarTransform(dims=(2, 2))

        # Points in each quadrant
        cartesian_coords = np.array(
            [
                [1, 1],  # Quadrant I
                [-1, 1],  # Quadrant II
                [-1, -1],  # Quadrant III
                [1, -1],  # Quadrant IV
            ]
        )

        polar = pt.imap(cartesian_coords)

        # Check angles are in correct quadrants (arctan2 returns [-π, π])
        # Q1: 0 < θ < π/2
        assert 0 < polar[0, 0] < math.pi / 2
        # Q2: π/2 < θ < π
        assert math.pi / 2 < polar[1, 0] < math.pi
        # Q3: -π < θ < -π/2 (arctan2 returns negative for Q3)
        assert -math.pi < polar[2, 0] < -math.pi / 2
        # Q4: -π/2 < θ < 0 (arctan2 returns negative for Q4)
        assert -math.pi / 2 < polar[3, 0] < 0

        # All radii should be sqrt(2)
        np.testing.assert_allclose(polar[:, 1], math.sqrt(2), rtol=1e-14)

    def test_polar_transform_angle_wraparound(self):
        """Test angle wraparound behavior."""
        pt = PolarTransform(dims=(2, 2))

        # Test angles outside [0, 2π] range
        coords = np.array(
            [
                [3 * math.pi, 1],  # > 2π
                [-math.pi / 2, 1],  # < 0
                [5 * math.pi, 1],  # >> 2π
                [-3 * math.pi, 1],  # << 0
            ]
        )

        result = pt.map(coords)

        # Should still produce valid cartesian coordinates
        assert np.isfinite(result).all()

        # Round-trip should work (though angles may be normalized to [-π, π])
        polar_back = pt.imap(result)
        cartesian_back = pt.map(polar_back)
        np.testing.assert_allclose(result, cartesian_back, rtol=1e-12, atol=1e-12)


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
            (0, 0, 0.1, 0.05, 0),  # Tangential only
        ]

        for coeff in coeffs_list:
            lt = LensDistortionTransform(coeff=coeff)

            # Test various coordinate sets
            coords_sets = [
                np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5]]),
                np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
                np.array([[0.1, 0.2], [0.3, -0.4], [-0.5, -0.1]]),
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
        image_points = np.array(
            [
                # Corners
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
                # Edge centers
                [0, -1],
                [1, 0],
                [0, 1],
                [-1, 0],
                # Near corners (high distortion region)
                [0.9, 0.9],
                [-0.9, 0.9],
                [-0.9, -0.9],
                [0.9, -0.9],
            ]
        )

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


class TestNonlinearTransformComposition:
    """Test composition of multiple nonlinear transforms."""

    def test_composite_nonlinear_accuracy(self):
        """Test accuracy of composite transforms with multiple nonlinear transforms."""
        # Create a composite of different nonlinear transforms
        log_transform = LogTransform(base=[2, 10], dims=(2, 2))
        polar_transform = PolarTransform(dims=(2, 2))
        comp_transform = polar_transform * log_transform  # Composite transform

        # Note: Direct composition of these may not be mathematically meaningful
        # but we test the mechanism works

        # Test individual transforms first
        coords = np.array([[2, 10], [4, 100]])

        # Apply transforms in sequence
        xformed_coords = comp_transform.map(coords)

        # Reverse the sequence
        reversed_coords = comp_transform.imap(xformed_coords)

        # Should recover original coordinates with relaxed tolerance
        np.testing.assert_allclose(coords, reversed_coords, rtol=1e-6)

    def test_nonlinear_as_affine_error(self):
        """Test that nonlinear transforms correctly raise NotImplementedError for as_affine."""
        transforms = [
            LogTransform(base=[2, 10], dims=(2, 2)),
            PolarTransform(dims=(2, 2)),
            LensDistortionTransform(coeff=(0.1, 0.05, 0.01, 0.01, 0.001)),
            SphericalTransform(dims=(3, 3)),
            MercatorSphericalTransform(dims=(2, 2)),
            LambertAzimuthalEqualAreaTransform(dims=(2, 2)),
        ]

        for transform in transforms:
            with pytest.raises(NotImplementedError):
                transform.as_affine()

    def test_nonlinear_composite_with_linear(self):
        """Test composition of nonlinear transforms with linear transforms."""
        # Create a mixed composite
        linear_transform = coorx.STTransform(scale=[2, 3], offset=[1, -1], dims=(2, 2))
        polar_transform = PolarTransform(dims=(2, 2))

        # Use coordinates that result in positive radii after linear transform
        # [0.5, 0.5] -> [2, 0.5], [0, 1] -> [1, 2], [1, 1] -> [3, 2]
        coords = np.array([[0.5, 0.5], [0, 1], [1, 1]])

        # Linear then nonlinear
        step1 = linear_transform.map(coords)
        step2 = polar_transform.map(step1)

        # Reverse
        rev_step1 = polar_transform.imap(step2)
        rev_step2 = linear_transform.imap(rev_step1)

        np.testing.assert_allclose(coords, rev_step2, rtol=1e-12)

    def test_polar_negative_radius_and_angle_wrapping(self):
        """Test polar transform behavior with negative radii and angle wrapping."""
        pt = PolarTransform(dims=(2, 2))

        # Test 1: Negative radius behavior
        # [θ=0, r=-1] should give same cartesian point as [θ=π, r=1]
        coords_neg = np.array([[0, -1]])
        coords_pos = np.array([[math.pi, 1]])

        result_neg = pt.map(coords_neg)
        result_pos = pt.map(coords_pos)

        # Should produce same (or nearly same) cartesian coordinates
        np.testing.assert_allclose(result_neg, result_pos, rtol=1e-12, atol=1e-15)

        # Test 2: Angle wrapping - angles that differ by 2π should produce same result
        angles_equiv = np.array(
            [[0, 1], [2 * math.pi, 1], [4 * math.pi, 1], [-2 * math.pi, 1]]
        )
        results = pt.map(angles_equiv)

        # All should produce same cartesian result
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-12, atol=1e-15)

        # Test 3: Inverse mapping canonical behavior
        # When mapping back from cartesian, negative radii become positive with angle adjustment
        cartesian_point = np.array([[1, 0]])  # Simple point on x-axis

        # Convert to polar
        polar_result = pt.imap(cartesian_point)

        # Should give θ≈0, r≈1 (canonical form)
        expected_theta = 0
        expected_r = 1
        np.testing.assert_allclose(polar_result[0, 0], expected_theta, atol=1e-12)
        np.testing.assert_allclose(polar_result[0, 1], expected_r, rtol=1e-12)

        # Test 4: Non-reversible negative radius case
        # [θ=3, r=-1] converts to cartesian then back to different polar coords
        problematic_coords = np.array([[3, -1]])
        cartesian = pt.map(problematic_coords)
        polar_back = pt.imap(cartesian)

        # Should NOT equal original due to negative radius canonicalization
        assert not np.allclose(problematic_coords, polar_back, rtol=1e-6)

        # But should still round-trip correctly if we go forward again
        cartesian_again = pt.map(polar_back)
        np.testing.assert_allclose(cartesian, cartesian_again, rtol=1e-12)

    @pytest.mark.parametrize("precision", [np.float32, np.float64])
    def test_nonlinear_precision_handling(self, precision):
        """Test nonlinear transforms with different floating-point precisions."""

        # Define test cases with transform, input coordinates, and expected precision
        test_cases = [
            # 2D Cartesian coordinate transforms
            {
                'transform': LogTransform(base=[2, 10], dims=(2, 2)),
                'coords': [[1, 2], [3, 4]],
                'rtol_float32': 1e-6,
                'rtol_float64': 1e-12,
            },
            {
                'transform': PolarTransform(dims=(2, 2)),
                'coords': [[1, 2], [3, 4]],
                'rtol_float32': 1e-6,
                'rtol_float64': 1e-12,
            },
            # 3D Cartesian to spherical coordinate transform
            {
                'transform': SphericalTransform(dims=(3, 3)),
                'coords': [[1, 1, 1], [2, 2, 2]],
                'rtol_float32': 1e-6,
                'rtol_float64': 1e-12,
            },
            # 2D spherical coordinate map projections (input: lon, lat in radians)
            {
                'transform': MercatorSphericalTransform(dims=(2, 2)),
                'coords': [[0.5, 0.3], [1.0, -0.2]],  # lon, lat in radians
                'rtol_float32': 1e-6,
                'rtol_float64': 1e-12,
            },
            {
                'transform': LambertAzimuthalEqualAreaTransform(dims=(2, 2)),
                'coords': [[0.5, 0.3], [1.0, -0.2]],  # lon, lat in radians
                'rtol_float32': 1e-6,
                'rtol_float64': 1e-12,
            },
        ]

        for case in test_cases:
            transform = case['transform']
            coords = np.array(case['coords'], dtype=precision)

            # Get expected precision for this precision type
            rtol = case['rtol_float32'] if precision == np.float32 else case['rtol_float64']

            # Forward transform
            result = transform.map(coords)

            # Result should maintain input precision when possible
            assert result.dtype == precision or result.dtype == np.float64

            # Round-trip test with expected precision
            back = transform.imap(result)
            np.testing.assert_allclose(coords, back, rtol=rtol)


class TestSphericalTransformEdgeCases:
    """Test SphericalTransform with edge cases and critical geometries."""

    def test_spherical_transform_cardinal_directions(self):
        """Test SphericalTransform with points along cardinal axes."""
        st = SphericalTransform(dims=(3, 3))

        # Test cardinal directions: +X, -X, +Y, -Y, +Z, -Z
        cartesian_coords = np.array([
            [1.0, 0.0, 0.0],    # +X axis
            [-1.0, 0.0, 0.0],   # -X axis
            [0.0, 1.0, 0.0],    # +Y axis
            [0.0, -1.0, 0.0],   # -Y axis
            [0.0, 0.0, 1.0],    # +Z axis (north pole)
            [0.0, 0.0, -1.0],   # -Z axis (south pole)
        ], dtype=np.float64)

        # Convert Cartesian → Spherical using _map (not _imap)
        spherical = st.map(cartesian_coords)

        # Check expected spherical coordinates
        # [lon, lat, r] where lon=arctan2(y,x), lat=arcsin(z/r), r=sqrt(x²+y²+z²)
        expected = np.array([
            [0.0, 0.0, 1.0],              # +X: lon=0, lat=0, r=1
            [math.pi, 0.0, 1.0],          # -X: lon=π, lat=0, r=1
            [math.pi/2, 0.0, 1.0],        # +Y: lon=π/2, lat=0, r=1
            [-math.pi/2, 0.0, 1.0],       # -Y: lon=-π/2, lat=0, r=1
            [0.0, math.pi/2, 1.0],        # +Z: lon=0 (arbitrary), lat=π/2, r=1
            [0.0, -math.pi/2, 1.0],       # -Z: lon=0 (arbitrary), lat=-π/2, r=1
        ], dtype=np.float64)

        np.testing.assert_allclose(spherical, expected, atol=1e-14)

        # Test inverse transformation: Spherical → Cartesian
        cartesian_back = st.imap(spherical)
        np.testing.assert_allclose(cartesian_back, cartesian_coords, rtol=1e-12, atol=1e-15)

    def test_spherical_transform_origin(self):
        """Test SphericalTransform with origin point."""
        st = SphericalTransform(dims=(3, 3))

        # Origin should have r=0, with lon and lat undefined (NaN)
        origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        spherical = st.map(origin)

        # r should be 0
        assert spherical[0, 2] == 0
        # lon and lat should be NaN due to division by zero in arctan2 and arcsin
        assert np.isnan(spherical[0, 0]) or spherical[0, 0] == 0  # arctan2(0,0) can return 0
        assert np.isnan(spherical[0, 1])  # arcsin(0/0) = NaN

    def test_spherical_transform_round_trip_accuracy(self):
        """Test round-trip accuracy: cartesian → spherical → cartesian."""
        st = SphericalTransform(dims=(3, 3))

        # Test various cartesian coordinates
        cartesian_coords = np.array([
            [1, 1, 1],           # First octant
            [-1, 1, 1],          # Second octant
            [-1, -1, 1],         # Third octant
            [1, -1, 1],          # Fourth octant
            [1, 1, -1],          # Fifth octant
            [-1, 1, -1],         # Sixth octant
            [-1, -1, -1],        # Seventh octant
            [1, -1, -1],         # Eighth octant
            [2, 3, 4],           # Different radii
            [0.1, 0.2, 0.3],     # Small values
        ])

        spherical = st.map(cartesian_coords)
        cartesian_back = st.imap(spherical)

        np.testing.assert_allclose(cartesian_coords, cartesian_back, rtol=1e-14, atol=1e-15)

    def test_spherical_transform_pole_singularities(self):
        """Test SphericalTransform at north and south poles where longitude is undefined."""
        st = SphericalTransform(dims=(3, 3))

        # Points at poles with different "longitudes" should all map to same cartesian point
        north_pole_coords = np.array([
            [0, math.pi/2, 1],        # North pole, lon=0
            [math.pi/4, math.pi/2, 1], # North pole, lon=π/4
            [math.pi/2, math.pi/2, 1], # North pole, lon=π/2
            [math.pi, math.pi/2, 1],   # North pole, lon=π
        ])

        # Convert spherical → cartesian using _imap
        cartesian = st.imap(north_pole_coords)

        # All should map to [0, 0, 1] (north pole in cartesian)
        expected_north = np.array([0, 0, 1])
        for i in range(len(cartesian)):
            np.testing.assert_allclose(cartesian[i], expected_north, atol=1e-14)

        # Same test for south pole
        south_pole_coords = np.array([
            [0, -math.pi/2, 1],
            [math.pi/4, -math.pi/2, 1],
            [math.pi/2, -math.pi/2, 1],
            [math.pi, -math.pi/2, 1],
        ])

        cartesian_south = st.imap(south_pole_coords)
        expected_south = np.array([0, 0, -1])
        for i in range(len(cartesian_south)):
            np.testing.assert_allclose(cartesian_south[i], expected_south, atol=1e-14)

    def test_spherical_transform_different_radii(self):
        """Test SphericalTransform with various radius values."""
        st = SphericalTransform(dims=(3, 3))

        # Test same direction with different radii
        radii = [0.1, 0.5, 1, 2, 10, 100]
        lon, lat = math.pi/4, math.pi/6  # 45°, 30°

        spherical_coords = np.array([[lon, lat, r] for r in radii])
        cartesian = st.imap(spherical_coords)

        # Check that all points lie on same ray from origin
        for i in range(1, len(cartesian)):
            # Direction vectors should be parallel
            direction1 = cartesian[0] / np.linalg.norm(cartesian[0])
            direction2 = cartesian[i] / np.linalg.norm(cartesian[i])
            np.testing.assert_allclose(direction1, direction2, rtol=1e-14)

        # Check that distances are correct
        for i, r in enumerate(radii):
            distance = np.linalg.norm(cartesian[i])
            np.testing.assert_allclose(distance, r, rtol=1e-14)

    def test_spherical_transform_extreme_values(self):
        """Test SphericalTransform with extreme coordinate values."""
        st = SphericalTransform(dims=(3, 3))

        # Very large radius
        large_coords = np.array([[0, 0, 1e6]])
        cartesian_large = st.imap(large_coords)
        spherical_back = st.map(cartesian_large)
        np.testing.assert_allclose(large_coords, spherical_back, rtol=1e-12)

        # Very small radius
        small_coords = np.array([[math.pi/4, math.pi/6, 1e-6]])
        cartesian_small = st.imap(small_coords)
        spherical_small_back = st.map(cartesian_small)
        np.testing.assert_allclose(small_coords, spherical_small_back, rtol=1e-10)

    def test_spherical_transform_angle_wraparound(self):
        """Test SphericalTransform with angles outside standard ranges."""
        st = SphericalTransform(dims=(3, 3))

        # Test longitude wraparound (angles outside [-π, π])
        coords = np.array([
            [0, 0, 1],           # Standard
            [2*math.pi, 0, 1],   # Equivalent to 0
            [3*math.pi, 0, 1],   # Equivalent to π
            [-3*math.pi, 0, 1],  # Equivalent to π
        ])

        cartesian = st.map(coords)

        # Should still produce valid results
        assert np.isfinite(cartesian).all()

        # Test latitude beyond [-π/2, π/2] - these are invalid but should not crash
        invalid_lat_coords = np.array([
            [0, math.pi, 1],     # Invalid latitude
            [0, -math.pi, 1],    # Invalid latitude
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cartesian_invalid = st.map(invalid_lat_coords)

        # Should produce some result (though may be nonsensical)
        assert cartesian_invalid.shape == invalid_lat_coords.shape


class TestMercatorSphericalTransformEdgeCases:
    """Test MercatorSphericalTransform with edge cases and critical latitudes."""

    def test_mercator_transform_equator(self):
        """Test MercatorSphericalTransform with points on the equator."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Points on equator (lat=0)
        equator_coords = np.array([
            [0, 0],                    # lon=0, lat=0
            [math.pi/2, 0],           # lon=π/2, lat=0
            [math.pi, 0],             # lon=π, lat=0
            [-math.pi/2, 0],          # lon=-π/2, lat=0
        ])

        mercator = mt.map(equator_coords)

        # On equator, y should be 0 (since log(tan(π/4 + 0/2)) = log(1) = 0)
        expected = np.array([
            [0, 0],
            [math.pi/2, 0],
            [math.pi, 0],
            [-math.pi/2, 0],
        ])

        np.testing.assert_allclose(mercator, expected, atol=1e-14)

    def test_mercator_transform_pole_clipping(self):
        """Test MercatorSphericalTransform with latitudes near poles."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Test latitudes very close to but not exactly at poles
        near_pole_coords = np.array([
            [0, math.pi/2 - 1e-6],    # Very close to north pole
            [0, -math.pi/2 + 1e-6],   # Very close to south pole
        ])

        mercator = mt.map(near_pole_coords)

        # Should produce large but finite y values
        assert np.isfinite(mercator).all()
        assert abs(mercator[0, 1]) > 10  # Should be large positive
        assert abs(mercator[1, 1]) > 10  # Should be large negative (or absolute value)

    def test_mercator_transform_round_trip_accuracy(self):
        """Test round-trip accuracy: spherical → mercator → spherical."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Test various spherical coordinates (avoiding poles)
        spherical_coords = np.array([
            [0, 0],                      # Equator
            [math.pi/4, math.pi/4],      # 45° lat
            [math.pi/2, math.pi/6],      # 30° lat
            [-math.pi/4, -math.pi/4],    # -45° lat
            [0, math.pi/3],              # 60° lat
        ])

        mercator = mt.map(spherical_coords)
        spherical_back = mt.imap(mercator)

        np.testing.assert_allclose(spherical_coords, spherical_back, rtol=1e-14, atol=1e-15)

    def test_mercator_transform_longitude_preservation(self):
        """Test that MercatorSphericalTransform preserves longitude exactly."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Test various longitudes with same latitude
        lat = math.pi/6  # 30°
        longitudes = np.array([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, -math.pi/2])

        coords = np.array([[lon, lat] for lon in longitudes])
        mercator = mt.map(coords)

        # x coordinate should exactly equal longitude
        for i, lon in enumerate(longitudes):
            assert mercator[i, 0] == lon

    def test_mercator_transform_extreme_latitudes(self):
        """Test MercatorSphericalTransform clipping behavior at poles."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Test with latitudes OUTSIDE the safe range that should be clipped
        # Implementation clips to [-π/2 + 1e-6, π/2 - 1e-6]
        extreme_coords = np.array([
            [0, math.pi/2],         # Exact north pole (would cause infinity)
            [0, -math.pi/2],        # Exact south pole (would cause -infinity)
            [1.0, math.pi/2 + 0.1], # Beyond north pole
            [-1.0, -math.pi/2 - 0.1], # Beyond south pole
        ], dtype=np.float64)

        # Apply transform
        mercator = mt.map(extreme_coords)

        # Should produce finite results due to clipping in the implementation
        assert np.isfinite(mercator).all()

        # Longitude should be preserved (no clipping on x-axis)
        np.testing.assert_array_equal(mercator[:, 0], [0, 0, 1.0, -1.0])

        # y values should be large but finite due to latitude clipping
        assert abs(mercator[0, 1]) > 10  # North pole clipped
        assert abs(mercator[1, 1]) > 10  # South pole clipped
        assert abs(mercator[2, 1]) > 10  # Beyond north pole clipped
        assert abs(mercator[3, 1]) > 10  # Beyond south pole clipped

        # Round-trip should also work (inverse should handle the clipped values)
        coords_back = mt.imap(mercator)
        # Due to clipping, we won't get back the exact extreme values
        # but should get back the clipped values
        expected_clipped = np.array([
            [0, math.pi/2 - 1e-6],
            [0, -(math.pi/2 - 1e-6)],
            [1.0, math.pi/2 - 1e-6],
            [-1.0, -(math.pi/2 - 1e-6)],
        ])
        np.testing.assert_allclose(coords_back, expected_clipped, rtol=1e-12)

    def test_mercator_transform_mathematical_accuracy(self):
        """Test mathematical accuracy of Mercator projection formulas."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Test specific known values
        test_cases = [
            # [lon, lat] -> expected [x, y]
            ([0.0, 0.0], [0.0, 0.0]),  # Equator, prime meridian
            ([math.pi/2, 0.0], [math.pi/2, 0.0]),  # Equator, 90°E
            ([0.0, math.pi/4], [0.0, math.log(math.tan(math.pi/4 + math.pi/8))]),  # 45°N
        ]

        for (lon, lat), (expected_x, expected_y) in test_cases:
            coords = np.array([[lon, lat]], dtype=np.float64)
            mercator = mt.map(coords)

            np.testing.assert_allclose(mercator[0, 0], expected_x, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(mercator[0, 1], expected_y, rtol=1e-12, atol=1e-15)

    def test_mercator_transform_inverse_accuracy(self):
        """Test accuracy of inverse Mercator projection."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Test specific inverse cases
        mercator_coords = np.array([
            [0.0, 0.0],                              # Origin
            [math.pi/2, 1.0],                     # Moderate y
            [-math.pi/4, -0.5],                 # Negative values
        ], dtype=np.float64)

        spherical = mt.imap(mercator_coords)
        mercator_back = mt.map(spherical)

        np.testing.assert_allclose(mercator_coords, mercator_back, rtol=1e-12, atol=1e-15)

    def test_mercator_transform_continuity(self):
        """Test continuity of Mercator transform across longitude boundaries."""
        mt = MercatorSphericalTransform(dims=(2, 2))

        # Test points very close to longitude boundary at ±π
        boundary_coords = np.array([
            [math.pi - 1e-10, math.pi/6],      # Just before +π
            [-math.pi + 1e-10, math.pi/6],     # Just after -π
        ])

        mercator = mt.map(boundary_coords)

        # Should produce finite, reasonable results
        assert np.isfinite(mercator).all()

        # The y values should be identical (same latitude)
        np.testing.assert_allclose(mercator[0, 1], mercator[1, 1], rtol=1e-12)


class TestLambertAzimuthalEqualAreaTransformEdgeCases:
    """Test LambertAzimuthalEqualAreaTransform with edge cases and projection geometry."""

    def test_lambert_transform_north_pole_center(self):
        """Test LambertAzimuthalEqualAreaTransform centered at north pole."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test projection center (north pole should map to origin)
        north_pole = np.array([[0, math.pi/2]])  # lon=0, lat=π/2
        projected = lt.map(north_pole)

        # Should map to origin (0, 0)
        expected = np.array([[0, 0]])
        np.testing.assert_allclose(projected, expected, atol=1e-14)

    def test_lambert_transform_equator_circle(self):
        """Test LambertAzimuthalEqualAreaTransform with points on the equator."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Points on equator at various longitudes
        equator_coords = np.array([
            [0, 0],              # lon=0, lat=0 (prime meridian)
            [math.pi/2, 0],      # lon=π/2, lat=0 (90°E)
            [math.pi, 0],        # lon=π, lat=0 (180°)
            [-math.pi/2, 0],     # lon=-π/2, lat=0 (90°W)
        ])

        projected = lt.map(equator_coords)

        # All equator points should have same distance from origin (equal area property)
        distances = [np.sqrt(projected[i, 0]**2 + projected[i, 1]**2) for i in range(len(projected))]

        # All distances should be equal
        for dist in distances:
            np.testing.assert_allclose(dist, distances[0], rtol=1e-12)

        # Expected distance for equator when centered at north pole
        # k = sqrt(2 / (1 + sin(0))) = sqrt(2)
        expected_distance = math.sqrt(2)
        np.testing.assert_allclose(distances[0], expected_distance, rtol=1e-12)

    def test_lambert_transform_round_trip_accuracy(self):
        """Test round-trip accuracy: spherical → lambert → spherical."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test various spherical coordinates
        spherical_coords = np.array([
            [0, math.pi/2],          # North pole (center)
            [0, 0],                  # Equator, prime meridian
            [math.pi/2, 0],          # Equator, 90°E
            [math.pi, 0],            # Equator, 180°
            [0, math.pi/4],          # 45°N, prime meridian
            [math.pi/4, math.pi/4],  # 45°N, 45°E
            [math.pi/2, math.pi/6],  # 30°N, 90°E
        ])

        lambert = lt.map(spherical_coords)
        spherical_back = lt.imap(lambert)

        np.testing.assert_allclose(spherical_coords, spherical_back, rtol=1e-14, atol=1e-15)

    def test_lambert_transform_antipodal_point(self):
        """Test LambertAzimuthalEqualAreaTransform near south pole (antipodal to center)."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test point moderately close to south pole (avoiding the exact singularity)
        near_south_pole = np.array([[0, -math.pi/2 + 1e-3]], dtype=np.float64)  # 1e-3 radians from pole

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            projected = lt.map(near_south_pole)

        # Should produce finite results (may be large due to proximity to singularity)
        assert np.isfinite(projected).all()

        # Distance should be reasonably large but finite
        distance = math.sqrt(projected[0, 0]**2 + projected[0, 1]**2)
        assert distance > 1.0  # Should be separated from center
        assert np.isfinite(distance)

    def test_lambert_transform_mathematical_accuracy(self):
        """Test mathematical accuracy of Lambert projection formulas."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test specific cases where we know the mathematical result
        # For projection centered at north pole (φ₀ = π/2, λ₀ = 0)

        # Equator at prime meridian: φ=0, λ=0
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
        projected = lt.map(coords)

        # k = sqrt(2 / (1 + sin(0))) = sqrt(2)
        # x = k * cos(0) * sin(0) = sqrt(2) * 1 * 0 = 0
        # y = -k * cos(0) * cos(0) = -sqrt(2) * 1 * 1 = -sqrt(2)
        k = math.sqrt(2 / (1 + math.sin(0.0)))
        expected_x = k * math.cos(0.0) * math.sin(0.0)  # = 0
        expected_y = -k * math.cos(0.0) * math.cos(0.0)  # = -sqrt(2) ≈ -1.414
        expected = np.array([[expected_x, expected_y]], dtype=np.float64)

        np.testing.assert_allclose(projected, expected, rtol=1e-12, atol=1e-15)

    def test_lambert_transform_symmetry(self):
        """Test symmetry properties of Lambert projection."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test longitude symmetry around prime meridian
        coords_east = np.array([[math.pi/4, math.pi/4]])   # 45°E, 45°N
        coords_west = np.array([[-math.pi/4, math.pi/4]])  # 45°W, 45°N

        projected_east = lt.map(coords_east)
        projected_west = lt.map(coords_west)

        # Should be mirror images across y-axis
        np.testing.assert_allclose(projected_east[0, 0], -projected_west[0, 0], rtol=1e-12)
        np.testing.assert_allclose(projected_east[0, 1], projected_west[0, 1], rtol=1e-12)

    def test_lambert_transform_edge_latitude_values(self):
        """Test LambertAzimuthalEqualAreaTransform with latitude edge values."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test latitudes very close to south pole (opposite to projection center)
        edge_coords = np.array([
            [0, -math.pi/2 + 1e-6],    # Very close to south pole
            [math.pi/2, -math.pi/3],   # 60°S
        ])

        projected = lt.map(edge_coords)

        # Should produce finite results
        assert np.isfinite(projected).all()

        # Distance from origin should be large for points near south pole
        distance_near_pole = math.sqrt(projected[0, 0]**2 + projected[0, 1]**2)
        assert distance_near_pole > 1.5  # Should be well separated from center

    def test_lambert_transform_inverse_accuracy(self):
        """Test accuracy of inverse Lambert projection."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test specific projected coordinates
        lambert_coords = np.array([
            [0, 0],                    # Origin (north pole)
            [1, 0],                    # Point on x-axis
            [0, -1],                   # Point on negative y-axis
            [0.5, 0.5],               # Diagonal point
        ])

        spherical = lt.imap(lambert_coords)
        lambert_back = lt.map(spherical)

        np.testing.assert_allclose(lambert_coords, lambert_back, rtol=1e-12, atol=1e-14)

    def test_lambert_transform_area_preservation_property(self):
        """Test that Lambert projection preserves area (equal-area property)."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Create a small spherical "rectangle" and verify its area is preserved
        # Note: This is a qualitative test since exact area computation is complex

        # Small region around 45°N, 0°
        delta = math.pi/36  # 5 degrees
        center_lat, center_lon = math.pi/4, 0

        corners_spherical = np.array([
            [center_lon - delta, center_lat - delta],
            [center_lon + delta, center_lat - delta],
            [center_lon + delta, center_lat + delta],
            [center_lon - delta, center_lat + delta],
        ])

        corners_projected = lt.map(corners_spherical)

        # Check that the projection produces a reasonable quadrilateral
        # (exact area computation would require more complex geometry)
        assert np.isfinite(corners_projected).all()

        # Verify that points maintain reasonable relative positions
        # The projected region should maintain its general shape properties
        x_coords = corners_projected[:, 0]
        y_coords = corners_projected[:, 1]

        # Should span some reasonable range in both x and y
        assert (x_coords.max() - x_coords.min()) > 0.01
        assert (y_coords.max() - y_coords.min()) > 0.01

    def test_lambert_transform_boundary_conditions(self):
        """Test LambertAzimuthalEqualAreaTransform at projection boundaries."""
        lt = LambertAzimuthalEqualAreaTransform(dims=(2, 2))

        # Test coordinates that might cause numerical issues, avoiding exact south pole
        boundary_coords = np.array([
            [0, math.pi/2],                    # Projection center (north pole)
            [math.pi, -math.pi/2 + 1e-8],     # Very close to south pole (avoiding singularity)
            [0, 1e-15],                       # Very close to equator
            [math.pi - 1e-15, 0],             # Very close to 180° longitude
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            projected = lt.map(boundary_coords)

        # Results should be mostly finite (except possibly the near-south-pole case)
        assert np.isfinite(projected[0]).all()  # North pole should be finite
        assert np.isfinite(projected[2]).all()  # Near equator should be finite
        assert np.isfinite(projected[3]).all()  # Near 180° should be finite

        # Test round-trip accuracy for the stable cases (excluding near south pole)
        stable_coords = boundary_coords[[0, 2, 3]]  # Skip the near-south-pole case
        stable_projected = projected[[0, 2, 3]]

        spherical_back = lt.imap(stable_projected)
        projected_again = lt.map(spherical_back)

        # Should be stable for non-singular cases
        np.testing.assert_allclose(stable_projected, projected_again, rtol=1e-10, atol=1e-12)
