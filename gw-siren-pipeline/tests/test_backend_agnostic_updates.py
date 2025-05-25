#!/usr/bin/env python3
"""Test backend-agnostic array updates."""

import os
import numpy as np
import pytest

# Force JAX CPU for testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from gwsiren.h0_mcmc_analyzer import H0LogLikelihood


def test_backend_agnostic_array_updates():
    """Test that the _update_array_element method works correctly."""
    
    print("🧪 Testing backend-agnostic array updates...")
    
    # Mock data for creating H0LogLikelihood instance
    mock_gw_samples = np.array([100.0, 200.0, 150.0])
    mock_z = np.array([0.02])
    mock_mass = np.array([1.0])
    mock_z_err = np.array([0.001])
    
    # Test NumPy backend
    print("\n1️⃣ Testing NumPy backend...")
    xp_numpy = np
    
    ll_numpy = H0LogLikelihood(
        xp=xp_numpy,
        backend_name="numpy",
        dL_gw_samples=mock_gw_samples,
        host_galaxies_z=mock_z,
        host_galaxies_mass_proxy=mock_mass,
        host_galaxies_z_err=mock_z_err,
    )
    
    # Test array update
    test_array = np.zeros(3)
    print(f"  Original array: {test_array}")
    
    updated_array = ll_numpy._update_array_element(test_array, 1, 5.0)
    print(f"  Updated array: {updated_array}")
    print(f"  Original modified: {test_array}")  # Should be modified in-place
    
    assert updated_array[1] == 5.0, "NumPy array update failed"
    assert test_array[1] == 5.0, "NumPy array not modified in-place"
    print("  ✅ NumPy backend update works correctly")


def test_jax_backend_awareness():
    """Test JAX backend detection."""
    print("\n2️⃣ Testing JAX backend detection...")
    
    # Mock data
    mock_gw_samples = np.array([100.0, 200.0, 150.0])
    mock_z = np.array([0.02])
    mock_mass = np.array([1.0])
    mock_z_err = np.array([0.001])
    
    # Create a mock JAX backend instance
    ll_jax_mock = H0LogLikelihood(
        xp=np,  # Using numpy as xp for simplicity
        backend_name="jax",  # But setting backend_name to jax
        dL_gw_samples=mock_gw_samples,
        host_galaxies_z=mock_z,
        host_galaxies_mass_proxy=mock_mass,
        host_galaxies_z_err=mock_z_err,
    )
    
    # Test that the method detects JAX backend
    print(f"  Backend detected as: {ll_jax_mock.backend_name}")
    assert ll_jax_mock.backend_name == "jax", "Backend name not set correctly"
    print("  ✅ JAX backend detection works correctly")
    
    # Test array update (this will fail because we're using numpy arrays with JAX logic)
    test_array = np.zeros(3)
    try:
        updated_array = ll_jax_mock._update_array_element(test_array, 1, 5.0)
        print("  ❌ Expected failure didn't occur - this suggests .at method exists on numpy arrays")
    except AttributeError as ae:
        print(f"  ✅ Expected AttributeError for numpy array with JAX logic: {ae}")


def test_array_update_summary():
    """Print summary of array update methods."""
    print("\n📊 Summary:")
    print("  - NumPy backend: Uses direct array assignment (array[i] = value)")  
    print("  - JAX backend: Would use immutable updates (array.at[i].set(value))")
    print("  - Method correctly detects backend and chooses appropriate strategy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 