# Tests for PetzvalCorrectionTransform.
# Physics: the Petzval focal surface satisfies z_offset(r) = Σ kᵢ·r^(2i),
# where r² = (x-cx)² + (y-cy)² and all kᵢ are even-power coefficients.

import numpy as np
import pytest

from coorx.nonlinear import PetzvalTransform


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def petzval_z_offset(xy, coeff, center=(0.0, 0.0)):
    """Reference implementation: sum of kᵢ * r^(2i) for i = 1, 2, ..."""
    r2 = (xy[:, 0] - center[0]) ** 2 + (xy[:, 1] - center[1]) ** 2
    offset = np.zeros(len(xy))
    for i, k in enumerate(coeff):
        offset += k * r2 ** (i + 1)
    return offset


# ---------------------------------------------------------------------------
# construction & parameters
# ---------------------------------------------------------------------------

class TestPetzvalConstruction:
    def test_default_dims_are_3d(self):
        t = PetzvalTransform(coeff=[0.1])
        assert t.dims == (3, 3)

    def test_explicit_dims_accepted(self):
        t = PetzvalTransform(coeff=[0.1], dims=(3, 3))
        assert t.dims == (3, 3)

    def test_non_3d_dims_raise(self):
        with pytest.raises(ValueError):
            PetzvalTransform(coeff=[0.1], dims=(2, 2))

    def test_default_center_is_origin(self):
        t = PetzvalTransform(coeff=[0.1])
        np.testing.assert_array_equal(t.center, [0.0, 0.0])

    def test_arbitrary_coeff_length(self):
        for n in (1, 2, 3, 5):
            coeff = [0.1] * n
            t = PetzvalTransform(coeff=coeff)
            assert len(t.coeff) == n

    def test_zero_coeff_is_valid(self):
        t = PetzvalTransform(coeff=[0.0, 0.0])
        assert t is not None

    def test_coeff_stored_as_float_array(self):
        t = PetzvalTransform(coeff=[1, 2, 3])
        assert t.coeff.dtype == np.float64


# ---------------------------------------------------------------------------
# forward mapping _map
# ---------------------------------------------------------------------------

class TestPetzvalMap:
    def test_identity_with_zero_coeff(self):
        t = PetzvalTransform(coeff=[0.0, 0.0])
        pts = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-5.0, 3.0, -1.0]])
        np.testing.assert_array_equal(t.map(pts), pts)

    def test_xy_pass_through_unchanged(self):
        t = PetzvalTransform(coeff=[0.5, 0.1])
        pts = np.array([[1.0, 2.0, 0.0], [-3.0, 4.0, 1.0]])
        result = t.map(pts)
        np.testing.assert_array_equal(result[:, :2], pts[:, :2])

    def test_single_k1_coefficient(self):
        k1 = 0.3
        t = PetzvalTransform(coeff=[k1])
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 5.0], [3.0, 4.0, -1.0]])
        result = t.map(pts)
        expected_dz = petzval_z_offset(pts[:, :2], [k1])
        np.testing.assert_allclose(result[:, 2], pts[:, 2] + expected_dz, rtol=1e-12)

    def test_two_coefficients(self):
        coeff = [0.2, 0.05]
        t = PetzvalTransform(coeff=coeff)
        pts = np.array([[1.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.5, -0.5, 2.0]])
        result = t.map(pts)
        expected_dz = petzval_z_offset(pts[:, :2], coeff)
        np.testing.assert_allclose(result[:, 2], pts[:, 2] + expected_dz, rtol=1e-12)

    def test_higher_order_coefficients(self):
        coeff = [0.1, 0.02, 0.003, 0.0004]
        t = PetzvalTransform(coeff=coeff)
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        result = t.map(pts)
        expected_dz = petzval_z_offset(pts[:, :2], coeff)
        np.testing.assert_allclose(result[:, 2], pts[:, 2] + expected_dz, rtol=1e-12)

    def test_on_axis_point_no_z_shift(self):
        """A point exactly on the optical axis (r=0) must not receive any z offset."""
        t = PetzvalTransform(coeff=[1.0, 2.0, 3.0])
        on_axis = np.array([[0.0, 0.0, 5.0]])
        result = t.map(on_axis)
        np.testing.assert_array_equal(result, on_axis)

    def test_custom_center(self):
        cx, cy = 3.0, -2.0
        coeff = [0.4]
        t = PetzvalTransform(coeff=coeff, center=(cx, cy))
        pts = np.array([[3.0, -2.0, 1.0], [4.0, -2.0, 0.0], [3.0, 0.0, 0.0]])
        result = t.map(pts)
        expected_dz = petzval_z_offset(pts[:, :2], coeff, center=(cx, cy))
        np.testing.assert_allclose(result[:, 2], pts[:, 2] + expected_dz, rtol=1e-12)

    def test_on_axis_with_custom_center(self):
        cx, cy = 2.0, 3.0
        t = PetzvalTransform(coeff=[5.0], center=(cx, cy))
        on_axis = np.array([[cx, cy, 7.0]])
        result = t.map(on_axis)
        np.testing.assert_array_equal(result, on_axis)

    def test_negative_coefficients(self):
        """Negative coefficients should produce a concave Petzval surface."""
        coeff = [-0.2, -0.05]
        t = PetzvalTransform(coeff=coeff)
        pts = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = t.map(pts)
        # Off-axis z should decrease
        assert result[1, 2] < result[0, 2]

    def test_rotational_symmetry(self):
        """Points at the same radial distance must receive the same z offset."""
        t = PetzvalTransform(coeff=[0.3, 0.07])
        r = 2.0
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(8)])
        result = t.map(pts)
        # All z values should be equal
        np.testing.assert_allclose(result[:, 2], result[0, 2], rtol=1e-12)

    def test_batch_of_points(self):
        """Result shape must match input shape."""
        t = PetzvalTransform(coeff=[0.1])
        pts = np.random.default_rng(0).standard_normal((100, 3))
        result = t.map(pts)
        assert result.shape == (100, 3)


# ---------------------------------------------------------------------------
# inverse mapping _imap
# ---------------------------------------------------------------------------

class TestPetzvalImap:
    def test_imap_is_closed_form_exact(self):
        """imap must invert map exactly (closed form, no iteration needed)."""
        coeff = [0.3, 0.07]
        t = PetzvalTransform(coeff=coeff)
        pts = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-2.0, 3.0, -1.5]])
        np.testing.assert_allclose(t.imap(t.map(pts)), pts, rtol=1e-14, atol=1e-14)

    def test_map_imap_round_trip_many_points(self):
        rng = np.random.default_rng(42)
        t = PetzvalTransform(coeff=[0.15, 0.03, 0.005])
        pts = rng.uniform(-5, 5, (200, 3))
        np.testing.assert_allclose(t.imap(t.map(pts)), pts, rtol=1e-13, atol=1e-13)

    def test_imap_map_round_trip(self):
        """imap ∘ map = id and map ∘ imap = id."""
        t = PetzvalTransform(coeff=[0.2, 0.04])
        pts = np.array([[1.0, 1.0, 0.0], [3.0, -2.0, 5.0]])
        np.testing.assert_allclose(t.map(t.imap(pts)), pts, rtol=1e-14, atol=1e-14)

    def test_imap_xy_unchanged(self):
        t = PetzvalTransform(coeff=[0.5])
        pts = np.array([[2.0, -1.0, 10.0], [0.5, 0.5, -3.0]])
        result = t.imap(pts)
        np.testing.assert_array_equal(result[:, :2], pts[:, :2])

    def test_imap_identity_with_zero_coeff(self):
        t = PetzvalTransform(coeff=[0.0])
        pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        np.testing.assert_array_equal(t.imap(pts), pts)

    def test_imap_on_axis_no_change(self):
        t = PetzvalTransform(coeff=[1.0, 2.0])
        on_axis = np.array([[0.0, 0.0, 7.0]])
        np.testing.assert_array_equal(t.imap(on_axis), on_axis)


# ---------------------------------------------------------------------------
# transform flags
# ---------------------------------------------------------------------------

class TestPetzvalFlags:
    def test_is_nonlinear(self):
        t = PetzvalTransform(coeff=[0.1])
        assert t.Linear is False

    def test_is_not_orthogonal(self):
        # z depends on x and y, so axes are coupled
        t = PetzvalTransform(coeff=[0.1])
        assert t.Orthogonal is False

    def test_is_equidimensional(self):
        t = PetzvalTransform(coeff=[0.1])
        assert t.Equidimensional is True


# ---------------------------------------------------------------------------
# numerical precision & stability
# ---------------------------------------------------------------------------

class TestPetzvalNumerics:
    def test_large_radii(self):
        """Must remain finite for large off-axis distances with small coefficients."""
        t = PetzvalTransform(coeff=[1e-6, 1e-10])
        pts = np.array([[1000.0, 1000.0, 0.0], [1e4, 0.0, 0.0]])
        result = t.map(pts)
        assert np.isfinite(result).all()

    def test_very_small_radii(self):
        t = PetzvalTransform(coeff=[0.3, 0.1])
        pts = np.array([[1e-10, 1e-10, 0.0], [0.0, 1e-8, 5.0]])
        result = t.map(pts)
        assert np.isfinite(result).all()
        # z offset should be essentially zero for near-on-axis points
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-15)

    def test_round_trip_extreme_values(self):
        t = PetzvalTransform(coeff=[1e-8, 1e-14])
        pts = np.array([[1e3, 1e3, 0.0]])
        np.testing.assert_allclose(t.imap(t.map(pts)), pts, rtol=1e-10)

    def test_float32_input(self):
        t = PetzvalTransform(coeff=[0.1])
        pts = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
        result = t.map(pts)
        assert result.shape == (1, 3)
        assert np.isfinite(result).all()
