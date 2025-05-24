#!/usr/bin/env python3
"""Simple JAX compatibility test for gravitational wave analysis."""

import sys
import pathlib
import numpy as np

# Add paths for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "gw-siren-pipeline"))

# Test JAX availability
try:
    import jax
    import jax.numpy as jnp
    jax_available = True
    print(f"✅ JAX {jax.__version__} available")
except ImportError:
    print("❌ JAX not available")
    jax_available = False
    exit(1)

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0
from utils.mock_data import mock_event

def test_jax_basic_functionality():
    """Test basic JAX operations."""
    print("\n🧪 Testing basic JAX functionality...")
    
    # Test basic JAX operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"  JAX array sum: {y}")
    
    # Test on Metal device
    print(f"  Available devices: {jax.devices()}")
    print(f"  Default device: {jax.devices()[0]}")
    
    return True

def test_jax_with_likelihood():
    """Test JAX backend with the likelihood function."""
    print("\n🧪 Testing JAX with likelihood function...")
    
    # Create mock data
    pkg = mock_event(n_galaxies=2, seed=42)
    theta = [70.0, 0.0]
    
    print(f"  Mock data created: {len(pkg.dl_samples)} GW samples, {len(pkg.candidate_galaxies_df)} galaxies")
    
    try:
        # Test JAX backend
        ll_jax = get_log_likelihood_h0(
            requested_backend_str="jax",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        print(f"  JAX likelihood function created successfully")
        print(f"  Backend: {ll_jax.backend_name}")
        print(f"  Vectorized: {ll_jax.use_vectorized_likelihood}")
        
        # Test likelihood evaluation
        val_jax = ll_jax(theta)
        print(f"  JAX likelihood value: {val_jax:.6f}")
        
        assert np.isfinite(val_jax), f"JAX likelihood not finite: {val_jax}"
        print("  ✅ JAX likelihood evaluation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ JAX likelihood test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_vs_jax_comparison():
    """Compare NumPy and JAX backends."""
    print("\n🧪 Comparing NumPy vs JAX backends...")
    
    # Create mock data  
    pkg = mock_event(n_galaxies=1, seed=123)
    theta = [70.0, 0.0]
    
    try:
        # NumPy backend
        ll_numpy = get_log_likelihood_h0(
            requested_backend_str="numpy",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # JAX backend
        ll_jax = get_log_likelihood_h0(
            requested_backend_str="jax",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # Compute likelihoods
        val_numpy = ll_numpy(theta)
        val_jax = ll_jax(theta)
        
        print(f"  NumPy likelihood: {val_numpy:.6f}")
        print(f"  JAX likelihood:   {val_jax:.6f}")
        print(f"  Difference:       {abs(val_numpy - val_jax):.2e}")
        
        # Check both are finite
        assert np.isfinite(val_numpy), f"NumPy result not finite: {val_numpy}"
        assert np.isfinite(val_jax), f"JAX result not finite: {val_jax}"
        
        print("  ✅ Both backends produce finite results")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running JAX compatibility tests on M3 Max...")
    
    all_passed = True
    
    # Test 1: Basic JAX functionality
    all_passed &= test_jax_basic_functionality()
    
    # Test 2: JAX with likelihood function
    all_passed &= test_jax_with_likelihood()
    
    # Test 3: NumPy vs JAX comparison
    all_passed &= test_numpy_vs_jax_comparison()
    
    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed!'}")
    
    if all_passed:
        print("🎉 JAX is working well on your M3 Max MacBook Pro!")
        print("💡 You can use JAX backend for potentially faster computations.")
    else:
        print("⚠️  JAX has some issues on your system.") 