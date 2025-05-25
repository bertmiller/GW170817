#!/usr/bin/env python3
"""Test NumPy/JAX backend consistency for redshift marginalization."""

import os
import numpy as np
import pytest

# Force JAX CPU for testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Skip all tests if JAX is not available
try:
    import jax
    jax_available = True
except ImportError:
    jax_available = False

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0
from .utils.mock_data import mock_event


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
    
    # Now that marginalization is implemented in both paths, they should agree
    for case_name in test_cases:
        case_name_str = case_name[0]
        numpy_val = results[case_name_str]['numpy']
        jax_val = results[case_name_str]['jax']
        
        # Extract scalar values for comparison
        if hasattr(numpy_val, 'item'):
            numpy_val = numpy_val.item()
        if hasattr(jax_val, 'item'):
            jax_val = jax_val.item()
            
        diff = abs(numpy_val - jax_val)
        rel_diff = diff / abs(numpy_val) if abs(numpy_val) > 1e-10 else diff
        
        print(f"  {case_name_str.capitalize()} z_err difference: {diff:.2e} (relative: {rel_diff:.2e})")
        
        # Both backends should now agree within numerical precision
        # Allow for some tolerance due to different computational paths
        tolerance = 1e-10  # Very tight tolerance for consistency
        assert diff < tolerance or rel_diff < 1e-8, \
            f"Backends disagree for {case_name_str}: NumPy={numpy_val:.12f}, JAX={jax_val:.12f}, diff={diff:.2e}"
    
    print(f"\n✅ Success: Both backends now implement consistent redshift marginalization!")
    
    # Don't return results to avoid pytest warning
    # return results


@pytest.mark.skipif(not jax_available, reason="JAX not available")  
def test_vectorized_vs_non_vectorized_consistency():
    """Test that vectorized and non-vectorized paths give the same results within each backend."""
    
    pkg = mock_event(n_galaxies=2, seed=123)
    theta = [70.0, 0.0]
    
    # Test both backends in vectorized and non-vectorized modes
    test_configs = [
        ("numpy", True, "NumPy vectorized"),
        ("numpy", False, "NumPy non-vectorized"),
        ("jax", True, "JAX vectorized"),
        ("jax", False, "JAX non-vectorized (forced)"),
    ]
    
    results = {}
    
    for backend, use_vectorized, label in test_configs:
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=np.array([0.005, 0.002]),  # Different z_err for each galaxy
            z_err_threshold=1e-6,
            n_quad_points=5,
            force_non_vectorized=not use_vectorized,
        )
        
        val = ll(theta)
        if hasattr(val, 'item'):
            val = val.item()
            
        results[label] = val
        print(f"{label}: {val:.8f}")
    
    # Check that vectorized and non-vectorized give consistent results
    numpy_diff = abs(results["NumPy vectorized"] - results["NumPy non-vectorized"])
    jax_diff = abs(results["JAX vectorized"] - results["JAX non-vectorized (forced)"])
    
    print(f"\nNumPy vectorized vs non-vectorized difference: {numpy_diff:.2e}")
    print(f"JAX vectorized vs non-vectorized difference: {jax_diff:.2e}")
    
    # Both should be very close
    tolerance = 1e-8
    assert numpy_diff < tolerance, f"NumPy vectorized/non-vectorized mismatch: {numpy_diff:.2e}"
    assert jax_diff < tolerance, f"JAX vectorized/non-vectorized mismatch: {jax_diff:.2e}"
    
    print("✅ Vectorized and non-vectorized paths are consistent!")


@pytest.mark.skipif(not jax_available, reason="JAX not available")  
def test_marginalization_threshold_behavior():
    """Test that marginalization threshold works correctly."""
    
    pkg = mock_event(n_galaxies=1, seed=456)
    theta = [70.0, 0.0]
    
    # Test with z_err right at the threshold
    threshold = 1e-6
    
    # Case 1: Just below threshold (should not marginalize)
    ll_below = get_log_likelihood_h0(
        requested_backend_str="jax",
        dL_gw_samples=pkg.dl_samples,
        host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
        host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
        host_galaxies_z_err=np.array([threshold * 0.5]),
        z_err_threshold=threshold,
        n_quad_points=5,
    )
    
    # Case 2: Just above threshold (should marginalize)
    ll_above = get_log_likelihood_h0(
        requested_backend_str="jax",
        dL_gw_samples=pkg.dl_samples,
        host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
        host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
        host_galaxies_z_err=np.array([threshold * 2.0]),
        z_err_threshold=threshold,
        n_quad_points=5,
    )
    
    val_below = ll_below(theta)
    val_above = ll_above(theta)
    
    if hasattr(val_below, 'item'):
        val_below = val_below.item()
    if hasattr(val_above, 'item'):
        val_above = val_above.item()
    
    print(f"Below threshold: {val_below:.8f}")
    print(f"Above threshold: {val_above:.8f}")
    
    # Both should be finite
    assert np.isfinite(val_below), f"Below threshold result not finite: {val_below}"
    assert np.isfinite(val_above), f"Above threshold result not finite: {val_above}"
    
    # They should be different due to marginalization effect
    diff = abs(val_below - val_above)
    print(f"Difference: {diff:.6f}")
    
    # Some difference expected, but not too large
    assert diff > 1e-6, f"Marginalization had insufficient effect: {diff:.2e}"
    assert diff < 10.0, f"Marginalization effect too large: {diff:.2e}"
    
    print("✅ Marginalization threshold behavior is correct!")


if __name__ == "__main__":
    if jax_available:
        print("🚀 Running NumPy/JAX consistency tests...")
        pytest.main([__file__, "-v"])
    else:
        print("❌ JAX not available - skipping consistency tests") 

# --- New Test for Full Likelihood Consistency (Stable Core) --- 

@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_full_likelihood_numpy_vs_jax_stable_core():
    """
    Tests the full H0LogLikelihood.__call__ method for numerical consistency
    between NumPy and JAX backends, focusing on paths that use the already
    stabilized core distance functions and bypassing complex marginalization initially.
    """
    if not jax_available: # Should be caught by skipif, but defensive
        pytest.skip("JAX not available")

    # Ensure JAX 64-bit precision (might be set globally, but good to be explicit for the test focus)
    # Note: jax is already imported if jax_available is True
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp # jnp will be available if jax is

    from gwsiren.h0_mcmc_analyzer import (
        H0LogLikelihood, # Using direct class instantiation
        DEFAULT_SIGMA_V_PEC, DEFAULT_C_LIGHT, DEFAULT_OMEGA_M,
        DEFAULT_H0_PRIOR_MIN, DEFAULT_H0_PRIOR_MAX,
        DEFAULT_ALPHA_PRIOR_MIN, DEFAULT_ALPHA_PRIOR_MAX,
        DEFAULT_Z_ERR_THRESHOLD, DEFAULT_QUAD_POINTS,
        DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE,
        DEFAULT_MCMC_BATCH_SIZE
    )
    # from .utils.mock_data import mock_event # Already imported at module level

    # Scenario 1: Minimal marginalization (z_err below threshold)
    print("\n🧪 Testing Scenario 1: Minimal Redshift Marginalization")
    pkg_scenario1 = mock_event(n_galaxies=2, n_samples=10, seed=42)
    pkg_scenario1.candidate_galaxies_df['z_err'] = 1e-8 # Ensure z_err is very small

    common_params_s1 = {
        "dL_gw_samples": pkg_scenario1.dl_samples,
        "host_galaxies_z": pkg_scenario1.candidate_galaxies_df["z"].values,
        "host_galaxies_mass_proxy": pkg_scenario1.candidate_galaxies_df["mass_proxy"].values,
        "host_galaxies_z_err": pkg_scenario1.candidate_galaxies_df["z_err"].values,
        "sigma_v": DEFAULT_SIGMA_V_PEC,
        "c_val": DEFAULT_C_LIGHT,
        "omega_m_val": DEFAULT_OMEGA_M,
        "h0_min": DEFAULT_H0_PRIOR_MIN, "h0_max": DEFAULT_H0_PRIOR_MAX,
        "alpha_min": DEFAULT_ALPHA_PRIOR_MIN, "alpha_max": DEFAULT_ALPHA_PRIOR_MAX,
        "z_err_threshold": DEFAULT_Z_ERR_THRESHOLD,
        "n_quad_points": DEFAULT_QUAD_POINTS, 
        "z_sigma_range": DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE,
        "use_vectorized_likelihood": False, # Force non-vectorized path (looped/batched)
        "batch_size": DEFAULT_MCMC_BATCH_SIZE # Relevant for the batched path when use_vectorized_likelihood=False
    }

    ll_numpy_s1 = H0LogLikelihood(xp=np, backend_name="numpy", **common_params_s1)
    ll_jax_s1 = H0LogLikelihood(xp=jnp, backend_name="jax", **common_params_s1)

    thetas_to_test = [
        [70.0, 0.0], 
        [65.0, 0.1], 
        [75.0, -0.1],
        [DEFAULT_H0_PRIOR_MIN + 1.0, DEFAULT_ALPHA_PRIOR_MIN + 0.05],
        [DEFAULT_H0_PRIOR_MAX - 1.0, DEFAULT_ALPHA_PRIOR_MAX - 0.05]
    ]
    theta_out_of_bounds_h0 = [DEFAULT_H0_PRIOR_MIN - 5.0, 0.0]
    theta_out_of_bounds_alpha = [70.0, DEFAULT_ALPHA_PRIOR_MAX + 0.1]
    all_thetas_s1 = thetas_to_test + [theta_out_of_bounds_h0, theta_out_of_bounds_alpha]

    for theta in all_thetas_s1:
        print(f"  Testing theta: {theta}")
        val_numpy = ll_numpy_s1(theta)
        val_jax_raw = ll_jax_s1(theta)
        val_jax_numpy = np.array(val_jax_raw) 

        is_theta_valid = (DEFAULT_H0_PRIOR_MIN <= theta[0] <= DEFAULT_H0_PRIOR_MAX and
                          DEFAULT_ALPHA_PRIOR_MIN <= theta[1] <= DEFAULT_ALPHA_PRIOR_MAX)

        if is_theta_valid:
            assert np.isfinite(val_numpy), f"S1 NumPy likelihood not finite for valid theta={theta}, got {val_numpy}"
            assert np.isfinite(val_jax_numpy), f"S1 JAX likelihood not finite for valid theta={theta}, got {val_jax_numpy}"
            assert np.allclose(val_numpy, val_jax_numpy, rtol=1e-5, atol=1e-8), \
                f"S1 Likelihood mismatch for theta={theta}: numpy={val_numpy}, jax={val_jax_numpy}"
        else: 
            assert val_numpy == -np.inf, f"S1 NumPy likelihood not -inf for out-of-bounds theta={theta}, got {val_numpy}"
            assert val_jax_numpy == -jnp.inf, f"S1 JAX likelihood not -inf for out-of-bounds theta={theta}, got {val_jax_numpy}"
    print("✅ Scenario 1 (Minimal Marginalization) Passed!")

    # Scenario 2: With Redshift Marginalization
    print("\n🧪 Testing Scenario 2: With Redshift Marginalization")
    pkg_scenario2 = mock_event(n_galaxies=2, n_samples=10, seed=43)
    # Ensure z_err triggers marginalization
    pkg_scenario2.candidate_galaxies_df['z_err'] = np.array([0.001, 0.005]) 

    common_params_s2 = common_params_s1.copy()
    common_params_s2.update({
        "dL_gw_samples": pkg_scenario2.dl_samples,
        "host_galaxies_z": pkg_scenario2.candidate_galaxies_df["z"].values,
        "host_galaxies_mass_proxy": pkg_scenario2.candidate_galaxies_df["mass_proxy"].values,
        "host_galaxies_z_err": pkg_scenario2.candidate_galaxies_df["z_err"].values,
        "n_quad_points": 5 # Using fewer quad points for test speed, as in other consistency tests
    })

    ll_numpy_s2 = H0LogLikelihood(xp=np, backend_name="numpy", **common_params_s2)
    ll_jax_s2 = H0LogLikelihood(xp=jnp, backend_name="jax", **common_params_s2)
    
    # Can reuse thetas_to_test, or define new ones if specific H0 values are more insightful for marginalization
    all_thetas_s2 = thetas_to_test # For now, use the same valid thetas

    for theta in all_thetas_s2:
        print(f"  Testing theta: {theta}")
        val_numpy = ll_numpy_s2(theta)
        val_jax_raw = ll_jax_s2(theta)
        val_jax_numpy = np.array(val_jax_raw)

        # All these thetas are valid for Scenario 2 check
        assert np.isfinite(val_numpy), f"S2 NumPy likelihood not finite for valid theta={theta}, got {val_numpy}"
        assert np.isfinite(val_jax_numpy), f"S2 JAX likelihood not finite for valid theta={theta}, got {val_jax_numpy}"
        # Tolerance might need to be slightly looser when marginalization is involved due to potential small differences
        # in quadrature or float precision accumulation, but start with the same.
        assert np.allclose(val_numpy, val_jax_numpy, rtol=1e-5, atol=1e-7), \
            f"S2 Likelihood mismatch for theta={theta}: numpy={val_numpy}, jax={val_jax_numpy}"
    print("✅ Scenario 2 (With Marginalization, n_quad_points=5) Passed!") 