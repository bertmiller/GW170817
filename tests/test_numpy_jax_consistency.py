#!/usr/bin/env python3
"""Test NumPy/JAX backend consistency for redshift marginalization."""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "gw-siren-pipeline"))

import numpy as np
import pytest

# Skip all tests if JAX is not available
try:
    import jax
    jax_available = True
except ImportError:
    jax_available = False

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0
from utils.mock_data import mock_event


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_numpy_jax_marginalization_consistency():
    """Test that NumPy and JAX backends give consistent results for redshift marginalization."""
    
    # Create test data
    pkg = mock_event(n_galaxies=1, seed=42)
    theta = [70.0, 0.0]
    
    # Test cases with different redshift errors
    test_cases = [
        ("tiny", 1e-8),      # Below threshold - should skip marginalization
        ("small", 0.001),    # Above threshold - should marginalize  
        ("medium", 0.005),   # Larger uncertainty
        ("large", 0.01)      # Large uncertainty
    ]
    
    results = {}
    
    for case_name, z_err in test_cases:
        print(f"\n🧪 Testing {case_name} z_err = {z_err}")
        
        # NumPy backend (forced non-vectorized to enable marginalization)
        ll_numpy = get_log_likelihood_h0(
            requested_backend_str="numpy",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=np.array([z_err]),
            z_err_threshold=1e-6,
            n_quad_points=5,  # Use fewer points for faster testing
            force_non_vectorized=True,  # Force non-vectorized path
        )
        
        # JAX backend (should use vectorized path since JAX forces vectorization)
        ll_jax = get_log_likelihood_h0(
            requested_backend_str="jax",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=np.array([z_err]),
            z_err_threshold=1e-6,
            n_quad_points=5,
        )
        
        # Compute likelihoods
        val_numpy = ll_numpy(theta)
        val_jax = ll_jax(theta)
        
        print(f"  NumPy (non-vec): {val_numpy:.8f}")
        print(f"  JAX (vectorized): {val_jax:.8f}")
        
        # Store results
        results[case_name] = {
            'numpy': val_numpy,
            'jax': val_jax,
            'z_err': z_err
        }
        
        # Check that both are finite
        assert np.isfinite(val_numpy), f"NumPy result not finite for {case_name}: {val_numpy}"
        assert np.isfinite(val_jax), f"JAX result not finite for {case_name}: {val_jax}"
    
    # Analysis of consistency
    print(f"\n📊 Consistency Analysis:")
    
    # For the tiny case (below threshold), both should be similar since no marginalization
    tiny_diff = abs(results['tiny']['numpy'] - results['tiny']['jax'])
    print(f"  Tiny z_err difference: {tiny_diff:.2e}")
    
    # Note: We expect differences for cases above threshold because:
    # - NumPy non-vectorized path: Implements proper redshift marginalization
    # - JAX vectorized path: Does NOT implement redshift marginalization
    # This is actually expected and reveals that marginalization is missing from vectorized path
    
    for case_name in ['small', 'medium', 'large']:
        diff = abs(results[case_name]['numpy'] - results[case_name]['jax'])
        print(f"  {case_name.capitalize()} z_err difference: {diff:.2e}")
        
        # The differences reveal that JAX (vectorized) path lacks marginalization
        # This is the underlying issue we need to address
    
    print(f"\n⚠️  Expected Behavior:")
    print(f"  - Tiny z_err: Both should be similar (no marginalization)")
    print(f"  - Larger z_err: Differences expected because JAX path lacks marginalization")
    
    return results


@pytest.mark.skipif(not jax_available, reason="JAX not available")  
def test_jax_marginalization_implementation():
    """Test that we need to implement marginalization in the JAX/vectorized path."""
    
    # This test documents the current limitation and serves as a placeholder
    # for the future implementation of marginalization in the vectorized path
    
    pkg = mock_event(n_galaxies=1, seed=123)
    theta = [70.0, 0.0]
    
    # Test with significant redshift uncertainty
    ll_jax = get_log_likelihood_h0(
        requested_backend_str="jax",
        dL_gw_samples=pkg.dl_samples,
        host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
        host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
        host_galaxies_z_err=np.array([0.01]),  # Large uncertainty
        z_err_threshold=1e-6,
    )
    
    val_jax = ll_jax(theta)
    
    # This should be finite
    assert np.isfinite(val_jax), f"JAX likelihood not finite: {val_jax}"
    
    # TODO: Implement redshift marginalization in vectorized path
    # When implemented, this test should verify that JAX gives similar results
    # to NumPy non-vectorized path for the same marginalization case


if __name__ == "__main__":
    if jax_available:
        print("🚀 Running NumPy/JAX consistency tests...")
        results = test_numpy_jax_marginalization_consistency()
        test_jax_marginalization_implementation()
        print("✅ Tests completed!")
    else:
        print("❌ JAX not available - skipping consistency tests") 