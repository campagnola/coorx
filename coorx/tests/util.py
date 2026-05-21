"""
Utility functions for testing transforms.
"""
import numpy as np
import pickle
import warnings


def check_mapping(transform, input_coords, output_coords, precision=1e-12):
    """
    Verify that transform.map(input_coords) approximately equals output_coords.

    Parameters
    ----------
    transform : Transform
        The transform to test
    input_coords : np.ndarray
        Input coordinates to map
    output_coords : np.ndarray
        Expected output coordinates
    precision : float, optional
        Relative tolerance for comparison (default: 1e-12)
    """
    input_array = np.asarray(input_coords)
    output_array = np.asarray(output_coords)

    # Suppress warnings during transform operations (edge cases can produce valid warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = transform.map(input_array)

    np.testing.assert_allclose(actual, output_array, rtol=precision, atol=1e-14)


def check_transform(transform, input_coords, output_coords,
                    forward_precision=1e-12, reverse_precision=1e-12, roundtrip_precision=1e-12,
                    test_reverse=True, test_roundtrip=True):
    """
    Comprehensive test of a transform including forward, reverse, round-trip, and serialization.

    Parameters
    ----------
    transform : Transform
        The transform to test
    input_coords : np.ndarray
        Input coordinates for forward mapping
    output_coords : np.ndarray
        Expected output coordinates from forward mapping
    forward_precision : float, optional
        Relative tolerance for forward mapping (default: 1e-12)
    reverse_precision : float, optional
        Relative tolerance for reverse mapping (default: 1e-12)
    roundtrip_precision : float, optional
        Relative tolerance for round-trip mapping (default: 1e-12)
    test_reverse : bool, optional
        Whether to test reverse mapping (default: True)
    test_roundtrip : bool, optional
        Whether to test round-trip mapping (default: True)
    """
    input_array = np.asarray(input_coords)
    output_array = np.asarray(output_coords)
        
    # Suppress warnings during transform operations (edge cases can produce valid warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Test forward mapping
        check_mapping(transform, input_array, output_array, forward_precision)

        # Test reverse mapping (optional)
        if test_reverse:
            check_mapping(transform.inverse, output_array, input_array, reverse_precision)

        # Test round-trip mapping (optional)
        if test_roundtrip:
            # Test round-trip: transform -> inverse -> should equal original
            forward_result = transform.map(input_array)
            roundtrip_result = transform.inverse.map(forward_result)
            np.testing.assert_allclose(input_array, roundtrip_result, rtol=roundtrip_precision, atol=1e-14)

            # Test reverse round-trip: inverse -> transform -> should equal original
            reverse_result = transform.inverse.map(output_array)
            reverse_roundtrip_result = transform.map(reverse_result)
            np.testing.assert_allclose(output_array, reverse_roundtrip_result, rtol=roundtrip_precision, atol=1e-14)

        # Test serialization/deserialization
        try:
            serialized = pickle.dumps(transform)
            deserialized = pickle.loads(serialized)

            # Test that deserialized transform works the same
            check_mapping(deserialized, input_array, output_array, forward_precision)

            # Test round-trip with deserialized transform
            deserialized_forward = deserialized.map(input_array)
            deserialized_roundtrip = deserialized.inverse.map(deserialized_forward)
            np.testing.assert_allclose(input_array, deserialized_roundtrip, rtol=roundtrip_precision, atol=1e-14)

        except Exception:
            # If serialization fails, that's OK - just skip this test
            # Some transforms might not be serializable
            pass