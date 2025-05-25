import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# Try to import JAX and configure, skip JAX tests if not available
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Assuming h0_mcmc_analyzer is structured such that H0LogLikelihood can be imported
# Adjust the import path as necessary based on your project structure.
# For example, if gw-siren-pipeline is the root of a package 'gwsiren':
from gwsiren.h0_mcmc_analyzer import H0LogLikelihood, DEFAULT_C_LIGHT

# Helper function for reference calculation (as provided by user)
def get_reference_comoving_distance_integral(z_scalar, omega_m_val, H0_ref=70.0, c_light_ref=DEFAULT_C_LIGHT):
    """
    Calculates the comoving distance integral using astropy.
    Integral = D_C * H0_ref / c_light_ref
    """
    if not np.isscalar(z_scalar):
        raise TypeError(f"z_scalar must be a scalar. Got {z_scalar} of type {type(z_scalar)}")
    if not np.isscalar(omega_m_val):
        raise TypeError(f"omega_m_val must be a scalar. Got {omega_m_val} of type {type(omega_m_val)}")

    if z_scalar < 1e-10: # Slightly larger threshold for practical zero
        return 0.0
    
    # Ensure H0_ref has units for astropy
    cosmo_astropy = FlatLambdaCDM(H0=H0_ref * u.km / u.s / u.Mpc, Om0=omega_m_val)
    try:
        # Astropy's comoving_distance can sometimes fail for z very close to 0
        # depending on internal precision or if z is exactly 0 in some contexts.
        if z_scalar < 1e-9: # If very close to zero, might return 0 directly to avoid astropy issues.
             comoving_dist_mpc = 0.0
        else:
            comoving_dist_mpc = cosmo_astropy.comoving_distance(z_scalar).to(u.Mpc).value
        
        if comoving_dist_mpc < 0: # Physical comoving distance cannot be negative
            return np.nan

    except Exception as e: # Handle cases where z_scalar might be out of Astropy's typical range
        # print(f"Astropy calculation failed for z={z_scalar}, Om0={omega_m_val}: {e}")
        return np.nan
        
    # Integral = D_C * H0_ref / c_light_ref
    # D_C = (c_light_ref / H0_ref) * integral
    # So, integral = D_C * (H0_ref / c_light_ref)
    reference_integral = comoving_dist_mpc * (H0_ref / c_light_ref)
    return reference_integral

def test_scalar_comoving_distance_integral_numpy_stability():
    """
    Tests the _scalar_comoving_distance_integral method of H0LogLikelihood
    for numerical stability and correctness against an astropy reference,
    using the NumPy backend.
    """
    test_omega_m_val = 0.3
    
    # Minimal valid dummy data for H0LogLikelihood instantiation
    dummy_dL_gw_samples = np.array([100.0, 150.0])
    dummy_host_galaxies_z = np.array([0.01, 0.02])
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([0.001, 0.001])

    # Instantiate H0LogLikelihood with NumPy backend
    analyzer_instance = H0LogLikelihood(
        xp=np,
        backend_name="numpy",
        dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z,
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err,
        omega_m_val=test_omega_m_val,
        # Other parameters can use defaults or minimal valid values
        sigma_v=300,
        c_val=DEFAULT_C_LIGHT, 
        h0_min=10, h0_max=200,
        alpha_min=-1, alpha_max=1,
        use_vectorized_likelihood=False # Not relevant for this specific method test
    )

    # Test z_scalar values
    # Values from near zero up to typical cosmological redshifts.
    z_scalar_values_to_test = [
        0.0, 
        1e-10, # Very close to zero
        1e-9, 
        0.0001, # Small redshift
        0.001, 
        0.01, 
        0.1, 
        0.5, 
        1.0, 
        2.0, 
        2.9, 
        2.99 # Close to cache max
    ]
    
    N_trapz_test_values = [200, 500] # Test with default and higher N_trapz

    for z_val in z_scalar_values_to_test:
        for n_trapz in N_trapz_test_values:
            # Call the pipeline's method
            # Ensure the method uses the instance's omega_m_val
            pipeline_integral_val = analyzer_instance._scalar_comoving_distance_integral(
                z_scalar=z_val, N_trapz=n_trapz
            )

            # Calculate the reference integral value
            reference_integral_val = get_reference_comoving_distance_integral(
                z_scalar=z_val, omega_m_val=test_omega_m_val
            )
            
            print(f"Testing z={z_val}, N_trapz={n_trapz}: Pipeline={pipeline_integral_val}, Reference={reference_integral_val}")

            # Assertions
            assert np.isfinite(pipeline_integral_val), \
                f"Integral is not finite for z={z_val}, N_trapz={n_trapz}. Got: {pipeline_integral_val}"
            
            if np.isfinite(reference_integral_val): # Only compare if reference is valid
                # For very small z_val, the absolute tolerance might be more important
                atol = 1e-7 if z_val < 0.001 else 0 
                assert np.allclose(pipeline_integral_val, reference_integral_val, rtol=1e-5, atol=atol), \
                    f"Integral mismatch for z={z_val}, N_trapz={n_trapz}: pipeline={pipeline_integral_val}, ref={reference_integral_val}"
            elif np.isnan(pipeline_integral_val):
                # If reference is NaN (e.g. Astropy failed), pipeline should ideally also indicate an issue gracefully,
                # though the current pipeline function doesn't explicitly return NaN, it might return large numbers or inf.
                # For this test, if astropy returns nan, we primarily care that our function is finite.
                # The isfinite check above handles this. We could add a specific check if pipeline_integral_val should also be NaN.
                pass
    
    # Test edge case: z_scalar is exactly 0.0
    pipeline_integral_at_zero = analyzer_instance._scalar_comoving_distance_integral(0.0)
    reference_integral_at_zero = get_reference_comoving_distance_integral(0.0, test_omega_m_val)
    assert pipeline_integral_at_zero == 0.0, "Integral at z=0.0 should be 0.0"
    assert reference_integral_at_zero == 0.0, "Reference integral at z=0.0 should be 0.0"
    assert np.allclose(pipeline_integral_at_zero, reference_integral_at_zero), "Mismatch at z=0"

    # Test edge cases for N_trapz
    small_N_trapz_values = [1, 2, 3] # Values that might break trapezoidal rule if not handled
    test_z_for_small_N = 0.1 # A typical redshift
    
    for n_trapz_edge in small_N_trapz_values:
        pipeline_val_edge_N = analyzer_instance._scalar_comoving_distance_integral(
            z_scalar=test_z_for_small_N, N_trapz=n_trapz_edge
        )
        reference_val_edge_N = get_reference_comoving_distance_integral(
            z_scalar=test_z_for_small_N, omega_m_val=test_omega_m_val
        )
        
        print(f"Testing z={test_z_for_small_N}, N_trapz (edge)={n_trapz_edge}: Pipeline={pipeline_val_edge_N}, Reference={reference_val_edge_N}")
        
        assert np.isfinite(pipeline_val_edge_N), \
            f"Integral is not finite for z={test_z_for_small_N}, N_trapz={n_trapz_edge}. Got: {pipeline_val_edge_N}"
        
        # For very small N_trapz, accuracy will be low.
        # The main point here is to check for crashes or NaNs rather than high precision.
        if n_trapz_edge == 1:
            assert pipeline_val_edge_N == 0.0, \
                f"For N_trapz=1, pipeline should return 0.0. Got: {pipeline_val_edge_N}"
        elif np.isfinite(reference_val_edge_N): # For N_trapz > 1 (e.g., 2, 3)
             assert np.allclose(pipeline_val_edge_N, reference_val_edge_N, rtol=0.5), \
                f"Integral mismatch for z={test_z_for_small_N}, N_trapz={n_trapz_edge} (relaxed check): pipeline={pipeline_val_edge_N}, ref={reference_val_edge_N}"

    # Test behavior with Ez potentially becoming zero or very small
    # This would happen if omega_m_val is such that Ez_sq becomes negative or zero.
    # For E(z) = sqrt(omega_m * (1+z)^3 + (1-omega_m)),
    # if omega_m = 0, E(z) = 1.
    # if omega_m = 1, E(z) = (1+z)^(3/2).
    # Problematic if (1-omega_m) is large and negative, but Om0 is constrained [0,1] in FlatLambdaCDM.
    # Test with omega_m = 0 and omega_m = 1
    
    analyzer_instance_om0 = H0LogLikelihood(
        xp=np, backend_name="numpy", dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z, host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err, omega_m_val=0.0
    )
    pipeline_om0 = analyzer_instance_om0._scalar_comoving_distance_integral(z_scalar=0.5, N_trapz=200)
    reference_om0 = get_reference_comoving_distance_integral(z_scalar=0.5, omega_m_val=0.0)
    assert np.isfinite(pipeline_om0), "Integral (Om0=0) not finite"
    assert np.allclose(pipeline_om0, reference_om0, rtol=1e-5), f"Mismatch for Om0=0: P={pipeline_om0}, R={reference_om0}"

    analyzer_instance_om1 = H0LogLikelihood(
        xp=np, backend_name="numpy", dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z, host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err, omega_m_val=1.0
    )
    pipeline_om1 = analyzer_instance_om1._scalar_comoving_distance_integral(z_scalar=0.5, N_trapz=200)
    reference_om1 = get_reference_comoving_distance_integral(z_scalar=0.5, omega_m_val=1.0)
    assert np.isfinite(pipeline_om1), "Integral (Om0=1) not finite"
    assert np.allclose(pipeline_om1, reference_om1, rtol=1e-5), f"Mismatch for Om0=1: P={pipeline_om1}, R={reference_om1}" 

# --- Test for _lum_dist_model_uncached --- 

def get_reference_luminosity_distance(z_values, H0_val, omega_m_val):
    z_array = np.atleast_1d(z_values)
    cosmo_astropy = FlatLambdaCDM(H0=H0_val * u.km / u.s / u.Mpc, Om0=omega_m_val)
    try:
        lum_dist_mpc_array = cosmo_astropy.luminosity_distance(z_array).to(u.Mpc).value
        return lum_dist_mpc_array[0] if np.isscalar(z_values) else lum_dist_mpc_array
    except Exception:
        return np.nan if np.isscalar(z_values) else np.full_like(z_array, np.nan, dtype=float)

def test_lum_dist_model_uncached_numpy_stability_and_accuracy():
    fixed_omega_m_val = 0.3
    fixed_c_val = DEFAULT_C_LIGHT # Using the one from h0_mcmc_analyzer
    
    dummy_dL_gw_samples = np.array([100.0, 150.0])
    dummy_host_galaxies_z = np.array([0.01, 0.02]) # Needs to match omega_m for analyzer instance
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([0.001, 0.001])

    analyzer_instance = H0LogLikelihood(
        xp=np,
        backend_name="numpy",
        dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z,
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err,
        omega_m_val=fixed_omega_m_val, # This omega_m is used by _scalar_comoving_distance_integral
        c_val=fixed_c_val, # This c_val is used by _lum_dist_model_uncached
        h0_min=10, h0_max=200,
        alpha_min=-1, alpha_max=1
    )

    # Test Scalar z input
    z_scalar_inputs = [0.0, 1e-9, 0.01, 0.1, 0.5, 1.0, 2.5]
    H0_inputs = [10.0, 50.0, 70.0, 100.0, 150.0]

    for z_val in z_scalar_inputs:
        for h0_val in H0_inputs:
            pipeline_lum_dist = analyzer_instance._lum_dist_model_uncached(z_val, h0_val)
            reference_lum_dist = get_reference_luminosity_distance(z_val, h0_val, fixed_omega_m_val)
            
            assert np.isscalar(pipeline_lum_dist), f"Pipeline output not scalar for scalar z={z_val}, H0={h0_val}"
            assert np.isfinite(pipeline_lum_dist), f"Lum_dist not finite for z={z_val}, H0={h0_val}. Got {pipeline_lum_dist}"
            if np.isfinite(reference_lum_dist):
                assert np.allclose(pipeline_lum_dist, reference_lum_dist, rtol=1e-5), \
                    f"Lum_dist mismatch z={z_val}, H0={h0_val}: P={pipeline_lum_dist}, R={reference_lum_dist}"

    # Test Array z input
    z_array_input = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 2.5])
    H0_test_val_for_array = 70.0
    pipeline_lum_dist_array = analyzer_instance._lum_dist_model_uncached(z_array_input, H0_test_val_for_array)
    reference_lum_dist_array = get_reference_luminosity_distance(z_array_input, H0_test_val_for_array, fixed_omega_m_val)

    assert pipeline_lum_dist_array.shape == z_array_input.shape, "Shape mismatch for array input"
    assert np.all(np.isfinite(pipeline_lum_dist_array)), "Array lum_dist contains non-finite values"
    if np.all(np.isfinite(reference_lum_dist_array)):
        assert np.allclose(pipeline_lum_dist_array, reference_lum_dist_array, rtol=1e-5), \
            "Array lum_dist mismatch with reference"

    # Test Edge case for H0_val near zero
    z_test_scalar = 0.1
    H0_edge_cases = [1e-10, 0.0] # Testing values that trigger the H0 < 1e-9 condition or are zero

    for h0_edge in H0_edge_cases:
        # Scalar z with edge H0
        pipeline_ld_edge_scalar = analyzer_instance._lum_dist_model_uncached(z_test_scalar, h0_edge)
        assert np.isinf(pipeline_ld_edge_scalar), \
            f"Expected inf for scalar z and H0_edge={h0_edge}, got {pipeline_ld_edge_scalar}"
        
        # Array z with edge H0
        pipeline_ld_edge_array = analyzer_instance._lum_dist_model_uncached(z_array_input, h0_edge)
        assert np.all(np.isinf(pipeline_ld_edge_array)), \
            f"Expected all inf for array z and H0_edge={h0_edge}, got {pipeline_ld_edge_array}"

# --- Test for H0DistanceCache accuracy --- 

def test_h0_distance_cache_accuracy_numpy():
    """
    Tests the H0DistanceCache accuracy via H0LogLikelihood.validate_distance_cache_accuracy.
    Also tests basic hit/miss logic.
    """
    fixed_omega_m_val = 0.3
    fixed_c_val = DEFAULT_C_LIGHT
    
    # Minimal valid dummy data for H0LogLikelihood instantiation
    dummy_dL_gw_samples = np.array([100.0, 150.0])
    dummy_host_galaxies_z = np.array([0.01, 0.02])
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([0.001, 0.001])

    analyzer_instance = H0LogLikelihood(
        xp=np,
        backend_name="numpy",
        dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z,
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err,
        omega_m_val=fixed_omega_m_val,
        c_val=fixed_c_val,
        h0_min=10, h0_max=200,
        alpha_min=-1, alpha_max=1
    )

    # Test cache accuracy
    # The validate_distance_cache_accuracy method uses predefined test_z and test_h0 values.
    # Default test H0 values for validate_distance_cache_accuracy are [50.0, 65.0, 70.0, 75.0, 90.0, 120.0]
    assert analyzer_instance.validate_distance_cache_accuracy(max_relative_error=1e-5) is True, \
        "Cache validation failed for max_relative_error=1e-5"

    # Test Cache Hit/Miss Logic (Optional but Recommended)
    # validate_distance_cache_accuracy would have populated the cache.
    
    stats_before_calls = analyzer_instance.get_distance_cache_stats()
    # These print statements help in debugging if assertions fail.
    # print(f"Stats before calls: {stats_before_calls}")

    # Test cache hit: Use an H0 value that should have been cached.
    # 70.0 is one of the default H0 values used by validate_distance_cache_accuracy.
    # The z_values [0.1, 0.5] are also within the default test_z_values range [0.01, ..., 2.0].
    test_z_for_hit_miss = np.array([0.1, 0.5])
    _ = analyzer_instance._lum_dist_model(z_values=test_z_for_hit_miss, H0_val=70.0)
    stats_after_hit = analyzer_instance.get_distance_cache_stats()
    # print(f"Stats after hit: {stats_after_hit}")
    assert stats_after_hit['cache_hits'] > stats_before_calls['cache_hits'], \
        f"Cache hit count did not increase. Before: {stats_before_calls['cache_hits']}, After: {stats_after_hit['cache_hits']}"

    # Test cache miss: Use a new H0 value not in the default test set for validate_distance_cache_accuracy,
    # and also different from the H0 tolerance used by the cache (default 0.01).
    # Make sure it's different enough from 70.0 to not fall into the same H0 bin if tolerance is used.
    _ = analyzer_instance._lum_dist_model(z_values=test_z_for_hit_miss, H0_val=71.23) 
    stats_after_miss = analyzer_instance.get_distance_cache_stats()
    # print(f"Stats after miss: {stats_after_miss}")
    # A miss can trigger a build, which might also be counted.
    # The key is that a new H0 value should not simply be a hit.
    # We expect either cache_misses to increase or num_interpolators_built to increase.
    assert (stats_after_miss['cache_misses'] > stats_after_hit['cache_misses'] or 
            stats_after_miss['num_interpolators_built'] > stats_after_hit['num_interpolators_built']), \
        f"Neither cache miss nor interpolator build count increased. AfterHit: {stats_after_hit}, AfterMiss: {stats_after_miss}"

    # Log stats for review
    print("Final Cache Stats for test_h0_distance_cache_accuracy_numpy:")
    analyzer_instance.log_distance_cache_stats()
    
    # Clean up for subsequent tests if any
    analyzer_instance.clear_distance_cache() 

# --- Test for _scalar_comoving_distance_integral (JAX Backend) --- 

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed or not configured")
def test_scalar_comoving_distance_integral_jax_stability_and_accuracy():
    """
    Tests _scalar_comoving_distance_integral with JAX backend.
    """
    # Ensure JAX is imported and configured if this test runs
    # This is mostly for clarity, as JAX_AVAILABLE handles the skip
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    # jnp should be defined globally if JAX_AVAILABLE is True
    global jnp

    omega_m_test_jax = 0.3 # Same as NumPy test for this function
    
    # Dummy data for H0LogLikelihood instantiation (NumPy arrays are fine, converted internally)
    dummy_dL_gw_samples = np.array([100.0, 150.0])
    dummy_host_galaxies_z = np.array([0.01, 0.02])
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([0.001, 0.001])

    analyzer_instance_jax = H0LogLikelihood(
        xp=jnp, # Use jax.numpy
        backend_name="jax",
        dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z,
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err,
        omega_m_val=omega_m_test_jax,
        c_val=DEFAULT_C_LIGHT,
        h0_min=10, h0_max=200,
        alpha_min=-1, alpha_max=1
    )

    test_redshifts = [0.0, 1e-9, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 2.9]
    N_trapz_test_values = [200, 500] # Default N_trapz in _scalar_comoving_distance_integral is 200

    for z_val in test_redshifts:
        for n_trapz in N_trapz_test_values:
            pipeline_integral_val_jax = analyzer_instance_jax._scalar_comoving_distance_integral(z_val, N_trapz=n_trapz)
            pipeline_integral_val_np = np.array(pipeline_integral_val_jax) # Convert JAX array to NumPy for assertions
            
            reference_integral_val = get_reference_comoving_distance_integral(z_val, omega_m_test_jax)

            assert np.isfinite(pipeline_integral_val_np), f"JAX integral not finite for z={z_val}, N_trapz={n_trapz}. Got: {pipeline_integral_val_np}"
            if np.isfinite(reference_integral_val):
                atol = 1e-8 if z_val < 0.001 else 1e-7 # Adjusted atol for JAX, can be fine-tuned
                assert np.allclose(pipeline_integral_val_np, reference_integral_val, rtol=1e-5, atol=atol), \
                    f"JAX integral mismatch z={z_val}, N_trapz={n_trapz}: P={pipeline_integral_val_np}, R={reference_integral_val}"

    # Test z_val = 0.0 specifically (should be 0.0)
    pipeline_integral_at_zero_jax = analyzer_instance_jax._scalar_comoving_distance_integral(0.0)
    pipeline_integral_at_zero_np = np.array(pipeline_integral_at_zero_jax)
    assert pipeline_integral_at_zero_np == 0.0, "JAX Integral at z=0.0 should be 0.0"

    # Test N_trapz edge cases
    small_N_trapz_values = [1, 2, 3]
    test_z_for_small_N_jax = 0.1

    for n_trapz_edge in small_N_trapz_values:
        pipeline_val_edge_jax = analyzer_instance_jax._scalar_comoving_distance_integral(test_z_for_small_N_jax, N_trapz=n_trapz_edge)
        pipeline_val_edge_np = np.array(pipeline_val_edge_jax)
        reference_val_edge_N = get_reference_comoving_distance_integral(test_z_for_small_N_jax, omega_m_test_jax)
        
        assert np.isfinite(pipeline_val_edge_np), f"JAX integral not finite z={test_z_for_small_N_jax}, N_trapz={n_trapz_edge}. Got: {pipeline_val_edge_np}"
        if n_trapz_edge == 1:
            # jax.numpy.trapz also returns 0.0 for a single point, similar to numpy.trapz
            assert pipeline_val_edge_np == 0.0, f"JAX N_trapz=1 should return 0.0. Got: {pipeline_val_edge_np}"
        elif np.isfinite(reference_val_edge_N):
            assert np.allclose(pipeline_val_edge_np, reference_val_edge_N, rtol=0.5), \
                f"JAX integral mismatch z={test_z_for_small_N_jax}, N_trapz={n_trapz_edge}: P={pipeline_val_edge_np}, R={reference_val_edge_N}"

# --- Test for _lum_dist_model_uncached (JAX Backend) --- 

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed or not configured")
def test_lum_dist_model_uncached_jax_stability_and_accuracy():
    """
    Tests _lum_dist_model_uncached with JAX backend for stability and accuracy.
    """
    if not JAX_AVAILABLE: # Redundant due to skipif, but good for clarity
        pytest.skip("JAX not available")
    
    global jnp # jnp is defined in the module scope if JAX_AVAILABLE is True

    fixed_omega_m_val = 0.3
    fixed_c_val = DEFAULT_C_LIGHT
    
    # Dummy data for H0LogLikelihood (NumPy arrays are fine, converted internally by the class)
    dummy_dL_gw_samples = np.array([100.0, 150.0])
    dummy_host_galaxies_z = np.array([0.01, 0.02])
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([0.001, 0.001])

    analyzer_instance_jax = H0LogLikelihood(
        xp=jnp, # Use jax.numpy
        backend_name="jax",
        dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z,
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err,
        omega_m_val=fixed_omega_m_val,
        c_val=fixed_c_val,
        h0_min=10, h0_max=200,
        alpha_min=-1, alpha_max=1
    )

    # Test Scalar z input
    z_scalar_inputs = [0.0, 1e-9, 0.001, 0.01, 0.1, 0.5, 1.0, 2.5]
    H0_inputs = [10.0, 50.0, 70.0, 100.0, 150.0]

    for z_val_scalar in z_scalar_inputs:
        for h0_val_scalar in H0_inputs:
            # Pass Python float for z_val_scalar, H0LogLikelihood handles conversion via self.xp
            pipeline_lum_dist_jax = analyzer_instance_jax._lum_dist_model_uncached(z_val_scalar, h0_val_scalar)
            pipeline_lum_dist_np = np.array(pipeline_lum_dist_jax) # Convert JAX array to NumPy
            
            reference_lum_dist = get_reference_luminosity_distance(z_val_scalar, h0_val_scalar, fixed_omega_m_val)
            
            assert np.isscalar(pipeline_lum_dist_np) or pipeline_lum_dist_np.ndim == 0, \
                f"JAX Pipeline output not scalar for scalar z={z_val_scalar}, H0={h0_val_scalar}. Shape: {pipeline_lum_dist_np.shape}"
            assert np.isfinite(pipeline_lum_dist_np), \
                f"JAX Lum_dist not finite for z={z_val_scalar}, H0={h0_val_scalar}. Got {pipeline_lum_dist_np}"
            if np.isfinite(reference_lum_dist):
                assert np.allclose(pipeline_lum_dist_np, reference_lum_dist, rtol=1e-5, atol=1e-8), \
                    f"JAX Lum_dist mismatch z={z_val_scalar}, H0={h0_val_scalar}: P={pipeline_lum_dist_np}, R={reference_lum_dist}"

    # Test Array z input
    z_array_input_np = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 2.5])
    # H0LogLikelihood with JAX backend will convert NumPy input to JAX array internally via self.xp.asarray
    # So, passing z_array_input_np directly is fine.
    H0_test_val_for_array = 70.0
    
    pipeline_lum_dist_jax_array = analyzer_instance_jax._lum_dist_model_uncached(z_array_input_np, H0_test_val_for_array)
    pipeline_lum_dist_np_array = np.array(pipeline_lum_dist_jax_array) # Convert JAX array to NumPy

    reference_lum_dist_array = get_reference_luminosity_distance(z_array_input_np, H0_test_val_for_array, fixed_omega_m_val)

    assert pipeline_lum_dist_np_array.shape == reference_lum_dist_array.shape, "JAX Shape mismatch for array input"
    assert np.all(np.isfinite(pipeline_lum_dist_np_array)), "JAX Array lum_dist contains non-finite values"
    if np.all(np.isfinite(reference_lum_dist_array)):
        assert np.allclose(pipeline_lum_dist_np_array, reference_lum_dist_array, rtol=1e-5, atol=1e-8), \
            "JAX Array lum_dist mismatch with reference"

    # Test Edge case for H0_val near zero
    z_test_scalar_edge = 0.1 
    H0_edge_cases = [1e-10, 0.0] 

    for h0_edge in H0_edge_cases:
        # Scalar z with edge H0 (pass z_test_scalar_edge as Python float)
        pipeline_ld_edge_jax_scalar = analyzer_instance_jax._lum_dist_model_uncached(z_test_scalar_edge, h0_edge)
        pipeline_ld_edge_np_scalar = np.array(pipeline_ld_edge_jax_scalar)
        assert np.isinf(pipeline_ld_edge_np_scalar) and pipeline_ld_edge_np_scalar > 0, \
            f"Expected positive inf for JAX scalar z and H0_edge={h0_edge}, got {pipeline_ld_edge_np_scalar}"
        
        # Array z with edge H0 (pass z_array_input_np as NumPy array)
        pipeline_ld_edge_jax_array = analyzer_instance_jax._lum_dist_model_uncached(z_array_input_np, h0_edge)
        pipeline_ld_edge_np_array = np.array(pipeline_ld_edge_jax_array)
        assert np.all(np.isinf(pipeline_ld_edge_np_array)) and np.all(pipeline_ld_edge_np_array > 0), \
            f"Expected all positive inf for JAX array z and H0_edge={h0_edge}, got {pipeline_ld_edge_np_array}"

# --- Test for H0DistanceCache accuracy (JAX-backed source) --- 

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed or not configured")
def test_h0_distance_cache_accuracy_when_source_is_jax():
    """
    Tests H0DistanceCache accuracy when its source function (_lum_dist_model_uncached)
    is operating with jax.numpy, via H0LogLikelihood.validate_distance_cache_accuracy.
    """
    if not JAX_AVAILABLE: # Defensive skip, though decorator should handle it
        pytest.skip("JAX not available")

    global jnp # jnp is defined in the module scope if JAX_AVAILABLE is True

    fixed_omega_m_val = 0.3
    fixed_c_val = DEFAULT_C_LIGHT
    
    # Minimal valid dummy data for H0LogLikelihood instantiation
    # These are passed as NumPy arrays; H0LogLikelihood converts to self.xp (jnp here)
    dummy_dL_gw_samples = np.array([100.0, 150.0])
    dummy_host_galaxies_z = np.array([0.01, 0.02])
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([0.001, 0.001])

    analyzer_instance_jax = H0LogLikelihood(
        xp=jnp, # Critical: use jax.numpy
        backend_name="jax", # Critical: set backend name to jax
        dL_gw_samples=dummy_dL_gw_samples,
        host_galaxies_z=dummy_host_galaxies_z,
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err,
        omega_m_val=fixed_omega_m_val,
        c_val=fixed_c_val,
        h0_min=10, h0_max=200,
        alpha_min=-1, alpha_max=1
        # The H0LogLikelihood constructor creates and configures the _distance_cache.
        # It sets self._lum_dist_model_uncached as the cache's compute function.
        # Since self.xp is jnp, _lum_dist_model_uncached will operate using jnp.
    )

    # Test cache accuracy
    # validate_distance_cache_accuracy will call the JAX-backed _lum_dist_model_uncached
    # to build interpolators. The H0DistanceCache itself handles converting JAX arrays
    # to NumPy arrays before feeding them to scipy's interp1d.
    max_err = 1e-5
    validation_passed = analyzer_instance_jax.validate_distance_cache_accuracy(max_relative_error=max_err)
    
    assert validation_passed is True, \
        f"Distance cache accuracy validation failed for JAX-backed source function with max_relative_error={max_err}."

    # Optional: Log Cache Statistics for review
    print("Cache Stats for test_h0_distance_cache_accuracy_when_source_is_jax:")
    analyzer_instance_jax.log_distance_cache_stats()
    
    # Clean up for subsequent tests if any
    analyzer_instance_jax.clear_distance_cache() 

# <<< START OF NEW TEST FUNCTION >>>
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed or not configured")
def test_core_static_marginalize_one_galaxy_jax_accuracy():
    """
    Tests the _core_static_marginalize_one_galaxy_jax function for numerical accuracy
    against the NumPy-based looped marginalization for a single galaxy.
    """
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    global jnp # Ensure jnp is accessible
    from gwsiren import h0_mcmc_analyzer # To access module-level jitted functions
    from gwsiren.h0_mcmc_analyzer import _core_static_marginalize_one_galaxy_jax, DEFAULT_C_LIGHT, DEFAULT_OMEGA_M, DEFAULT_SIGMA_V_PEC
    # Assuming CONFIG is available or use default values directly if not easily mockable here
    # from gwsiren import CONFIG # May not be needed if we use defaults

    # Test Parameters
    H0_test = 70.0
    mu_z_test = 0.05
    sigma_z_test = 0.005
    # Using fixed seed for reproducibility of random dL samples
    np.random.seed(42)
    dL_gw_samples_np = np.random.normal(loc=220, scale=20, size=100).astype(np.float64)
    
    c_val_test = DEFAULT_C_LIGHT
    sigma_v_val_test = DEFAULT_SIGMA_V_PEC 
    omega_m_val_test = DEFAULT_OMEGA_M
    # These are often from H0LogLikelihood instance or CONFIG
    # Use H0LogLikelihood defaults if available for n_quad_points and z_sigma_range
    n_quad_points_test = h0_mcmc_analyzer.DEFAULT_QUAD_POINTS 
    z_sigma_range_test = h0_mcmc_analyzer.DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE
    N_trapz_lum_dist_test = 200 # A common default for N_trapz in lum_dist calculations

    # --- Setup NumPy based H0LogLikelihood for reference calculation ---
    # Dummy galaxy data for instantiation, only one galaxy will be effectively tested by calling _marginalize_single_galaxy_redshift_looped
    dummy_host_galaxies_z = np.array([mu_z_test, 0.1]) # Need at least one, can be more
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([sigma_z_test, 0.01])

    analyzer_numpy = H0LogLikelihood(
        xp=np,
        backend_name="numpy",
        dL_gw_samples=dL_gw_samples_np,
        host_galaxies_z=dummy_host_galaxies_z, # mu_z_test is first element
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err, # sigma_z_test is first element
        c_val=c_val_test,
        sigma_v=sigma_v_val_test,
        omega_m_val=omega_m_val_test,
        n_quad_points=n_quad_points_test,
        z_sigma_range=z_sigma_range_test
        # Other params like h0_min/max, alpha_min/max are not directly relevant for this specific internal method call
    )
    # The _marginalize_single_galaxy_redshift_looped uses instance quad nodes/weights
    # and other parameters set during __init__.

    # --- Reference Value from NumPy path ---
    # _marginalize_single_galaxy_redshift_looped internally calls _perform_looped_quadrature_integration
    # It uses the instance's self.dL_gw_samples, self._quad_nodes, self._quad_weights etc.
    # It takes mu_z, sigma_z for *one specific galaxy* and H0.
    reference_logL = analyzer_numpy._marginalize_single_galaxy_redshift_looped(mu_z_test, sigma_z_test, H0_test)
    print(f"Reference NumPy LogL: {reference_logL}")

    # --- Prepare JAX inputs ---
    H0_jnp = jnp.array(H0_test, dtype=jnp.float64)
    mu_z_jnp = jnp.array(mu_z_test, dtype=jnp.float64)
    sigma_z_jnp = jnp.array(sigma_z_test, dtype=jnp.float64)
    dL_gw_samples_jnp = jnp.array(dL_gw_samples_np, dtype=jnp.float64)
    
    # Get quad nodes and weights from the numpy analyzer instance (they are generated in __init__)
    # and convert them to JAX arrays. These are Gauss-Hermite nodes/weights.
    quad_nodes_np = analyzer_numpy._quad_nodes
    quad_weights_np = analyzer_numpy._quad_weights
    quad_nodes_jnp = jnp.array(quad_nodes_np, dtype=jnp.float64)
    quad_weights_jnp = jnp.array(quad_weights_np, dtype=jnp.float64)

    # Get the JITted luminosity distance function and comoving integral function
    # These are module-level functions in h0_mcmc_analyzer.py
    jitted_lum_dist_func = h0_mcmc_analyzer._jitted_static_lum_dist
    jitted_comoving_integral_func = h0_mcmc_analyzer._jitted_static_comoving_integral

    # --- Execute JAX based static function ---
    pipeline_logL_jax = _core_static_marginalize_one_galaxy_jax(
        jnp, 
        H0_jnp, 
        mu_z_jnp, 
        sigma_z_jnp, 
        dL_gw_samples_jnp, 
        quad_nodes_jnp, 
        quad_weights_jnp, 
        c_val_test, 
        sigma_v_val_test, 
        omega_m_val_test, 
        jitted_lum_dist_func, 
        jitted_comoving_integral_func,
        z_sigma_range_test, # Passed for signature, though not used in current JAX func impl
        N_trapz_lum_dist_test
    )
    pipeline_logL_np = np.array(pipeline_logL_jax) # Convert JAX output to NumPy
    print(f"Pipeline JAX LogL (converted to np): {pipeline_logL_np}")

    # --- Assertions ---
    assert np.isfinite(pipeline_logL_np), f"Pipeline JAX logL is not finite: {pipeline_logL_np}"
    assert np.isfinite(reference_logL), f"Reference NumPy logL is not finite: {reference_logL}"
    
    # Check if both are -inf (can happen if all likelihoods are too small)
    if np.isneginf(reference_logL) and np.isneginf(pipeline_logL_np):
        pass # Both are -inf, considered close enough
    else:
        assert np.allclose(pipeline_logL_np, reference_logL, rtol=1e-5, atol=1e-8), \
            f"Mismatch between JAX pipeline LogL and NumPy reference LogL.\n" \
            f"JAX: {pipeline_logL_np}, NumPy: {reference_logL}, Diff: {pipeline_logL_np - reference_logL}"

# <<< END OF NEW TEST FUNCTION >>> 

# <<< START OF NEW TEST FUNCTION FOR JITTED VERSION >>>
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed or not configured")
def test_jitted_core_static_marginalize_one_galaxy_accuracy():
    """
    Tests the JIT-compiled _jitted_core_static_marginalize_one_galaxy function 
    for numerical accuracy against the NumPy-based looped marginalization.
    """
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    global jnp # Ensure jnp is accessible
    from gwsiren import h0_mcmc_analyzer # To access module-level jitted functions
    # Import the JITted version of the core function
    from gwsiren.h0_mcmc_analyzer import (
        _jitted_core_static_marginalize_one_galaxy, 
        DEFAULT_C_LIGHT, 
        DEFAULT_OMEGA_M, 
        DEFAULT_SIGMA_V_PEC,
        _jitted_static_lum_dist, # JITted lum_dist function
        _jitted_static_comoving_integral # JITted comoving integral function
    )

    # Test Parameters (same as the non-JITted test for direct comparison)
    H0_test = 70.0
    mu_z_test = 0.05
    sigma_z_test = 0.005
    np.random.seed(42) # Consistent random samples
    dL_gw_samples_np = np.random.normal(loc=220, scale=20, size=100).astype(np.float64)
    
    c_val_test = DEFAULT_C_LIGHT
    sigma_v_val_test = DEFAULT_SIGMA_V_PEC 
    omega_m_val_test = DEFAULT_OMEGA_M
    n_quad_points_test = h0_mcmc_analyzer.DEFAULT_QUAD_POINTS 
    z_sigma_range_test = h0_mcmc_analyzer.DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE
    N_trapz_lum_dist_test = 200

    # --- Setup NumPy based H0LogLikelihood for reference calculation ---
    dummy_host_galaxies_z = np.array([mu_z_test, 0.1])
    dummy_host_galaxies_mass_proxy = np.array([1.0, 1.0])
    dummy_host_galaxies_z_err = np.array([sigma_z_test, 0.01])

    analyzer_numpy = H0LogLikelihood(
        xp=np,
        backend_name="numpy",
        dL_gw_samples=dL_gw_samples_np,
        host_galaxies_z=dummy_host_galaxies_z,
        host_galaxies_mass_proxy=dummy_host_galaxies_mass_proxy,
        host_galaxies_z_err=dummy_host_galaxies_z_err,
        c_val=c_val_test,
        sigma_v=sigma_v_val_test,
        omega_m_val=omega_m_val_test,
        n_quad_points=n_quad_points_test,
        z_sigma_range=z_sigma_range_test
    )
    reference_logL = analyzer_numpy._marginalize_single_galaxy_redshift_looped(mu_z_test, sigma_z_test, H0_test)
    print(f"Reference NumPy LogL (for JIT test): {reference_logL}")

    # --- Prepare JAX inputs ---
    H0_jnp = jnp.array(H0_test, dtype=jnp.float64)
    mu_z_jnp = jnp.array(mu_z_test, dtype=jnp.float64)
    sigma_z_jnp = jnp.array(sigma_z_test, dtype=jnp.float64)
    dL_gw_samples_jnp = jnp.array(dL_gw_samples_np, dtype=jnp.float64)
    quad_nodes_np = analyzer_numpy._quad_nodes
    quad_weights_np = analyzer_numpy._quad_weights
    quad_nodes_jnp = jnp.array(quad_nodes_np, dtype=jnp.float64)
    quad_weights_jnp = jnp.array(quad_weights_np, dtype=jnp.float64)

    # --- Execute JIT-compiled JAX static function ---
    # Prepare arguments for the JITted function
    # Dynamic args: H0_val_jnp, mu_z_scalar_jnp, sigma_z_scalar_jnp
    # Static args: xp_module, dL_gw_samples_jnp, _quad_nodes_jnp, _quad_weights_jnp, c_val, sigma_v_val, omega_m_val, 
    #              _jitted_lum_dist_calculator_func, _jitted_comoving_integral_func, z_sigma_range, N_trapz_lum_dist

    # Warm-up call for JIT compilation (optional, but good practice for timing or ensuring compilation happens before main call)
    _ = _jitted_core_static_marginalize_one_galaxy(
        jnp, H0_jnp, mu_z_jnp, sigma_z_jnp, # Dynamic args first
        # Static args follow, matching the order expected by the JITted function based on static_argnames
        # The JITted function itself will receive all args, and JAX handles which are static vs dynamic.
        # When calling, we pass all arguments normally.
        dL_gw_samples_jnp, 
        quad_nodes_jnp, 
        quad_weights_jnp, 
        c_val_test, 
        sigma_v_val_test, 
        omega_m_val_test,
        _jitted_static_lum_dist, # Pass the actual JITted lum_dist function
        _jitted_static_comoving_integral, # Pass the actual JITted comoving integral function
        z_sigma_range_test, 
        N_trapz_lum_dist_test
    ).block_until_ready() # Ensure compilation finishes

    # Actual call for testing
    pipeline_logL_jitted_jax = _jitted_core_static_marginalize_one_galaxy(
        jnp, H0_jnp, mu_z_jnp, sigma_z_jnp, 
        dL_gw_samples_jnp, 
        quad_nodes_jnp, 
        quad_weights_jnp, 
        c_val_test, 
        sigma_v_val_test, 
        omega_m_val_test,
        _jitted_static_lum_dist, 
        _jitted_static_comoving_integral,
        z_sigma_range_test, 
        N_trapz_lum_dist_test
    )
    pipeline_logL_np_from_jitted = np.array(pipeline_logL_jitted_jax)
    print(f"Pipeline JITted JAX LogL (converted to np): {pipeline_logL_np_from_jitted}")

    # --- Assertions ---
    assert np.isfinite(pipeline_logL_np_from_jitted), f"Pipeline JITted JAX logL is not finite: {pipeline_logL_np_from_jitted}"
    assert np.isfinite(reference_logL), f"Reference NumPy logL (for JIT test) is not finite: {reference_logL}"
    
    if np.isneginf(reference_logL) and np.isneginf(pipeline_logL_np_from_jitted):
        pass
    else:
        assert np.allclose(pipeline_logL_np_from_jitted, reference_logL, rtol=1e-5, atol=1e-8), \
            f"Mismatch between JITted JAX pipeline LogL and NumPy reference LogL.\n" \
            f"JITted JAX: {pipeline_logL_np_from_jitted}, NumPy: {reference_logL}, Diff: {pipeline_logL_np_from_jitted - reference_logL}"

# <<< END OF NEW TEST FUNCTION FOR JITTED VERSION >>> 

# <<< START OF NEW TEST FUNCTION FOR VMAPPED JITTED VERSION >>>
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed or not configured")
def test_vmapped_jit_marginalize_galaxies_batch_accuracy():
    """
    Tests the vmap'ped and JITted _vmapped_jit_marginalize_galaxies function
    for numerical accuracy over a batch of galaxies against a looped NumPy reference.
    """
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    global jnp
    from gwsiren import h0_mcmc_analyzer
    from gwsiren.h0_mcmc_analyzer import (
        _vmapped_jit_marginalize_galaxies,
        _jitted_static_lum_dist,
        _jitted_static_comoving_integral,
        DEFAULT_C_LIGHT,
        DEFAULT_OMEGA_M,
        DEFAULT_SIGMA_V_PEC
    )

    # Test Parameters
    H0_test = 70.0
    mu_z_batch_np = np.array([0.05, 0.06, 0.07, 0.02], dtype=np.float64)
    sigma_z_batch_np = np.array([0.005, 0.006, 0.004, 0.001], dtype=np.float64)
    np.random.seed(42) # Consistent random samples
    dL_gw_samples_np = np.random.normal(loc=220, scale=20, size=100).astype(np.float64)
    
    c_val_test = DEFAULT_C_LIGHT
    sigma_v_val_test = DEFAULT_SIGMA_V_PEC
    omega_m_val_test = DEFAULT_OMEGA_M
    n_quad_points_test = h0_mcmc_analyzer.DEFAULT_QUAD_POINTS # Used for numpy_analyzer setup
    z_sigma_range_test = h0_mcmc_analyzer.DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE
    N_trapz_lum_dist_test = 200

    # --- Setup NumPy based H0LogLikelihood for reference calculation ---
    # This analyzer will be used to call its single-galaxy looped marginalization.
    # The dL_gw_samples and other parameters should match those used for the JAX path.
    analyzer_numpy = H0LogLikelihood(
        xp=np,
        backend_name="numpy",
        dL_gw_samples=dL_gw_samples_np,
        # Dummy host galaxies z, mass_proxy, z_err for instantiation, not directly used for batch ref.
        host_galaxies_z=mu_z_batch_np, # Can use the batch data here
        host_galaxies_mass_proxy=np.ones_like(mu_z_batch_np),
        host_galaxies_z_err=sigma_z_batch_np, # Can use the batch data here
        c_val=c_val_test,
        sigma_v=sigma_v_val_test,
        omega_m_val=omega_m_val_test,
        n_quad_points=n_quad_points_test,
        z_sigma_range=z_sigma_range_test
    )

    # --- Reference Value Calculation (Looping NumPy single-galaxy marginalization) ---
    reference_logLs_np = np.array([
        analyzer_numpy._marginalize_single_galaxy_redshift_looped(
            mu_z_batch_np[i], sigma_z_batch_np[i], H0_test
        ) for i in range(len(mu_z_batch_np))
    ])
    print(f"Reference NumPy LogLs (batch loop): {reference_logLs_np}")

    # --- Prepare JAX inputs ---
    # Broadcasted arguments (scalars or fixed arrays for the batch)
    H0_jnp = jnp.array(H0_test, dtype=jnp.float64)
    dL_gw_samples_jnp = jnp.array(dL_gw_samples_np, dtype=jnp.float64)
    quad_nodes_np = analyzer_numpy._quad_nodes # Get from numpy instance
    quad_weights_np = analyzer_numpy._quad_weights # Get from numpy instance
    quad_nodes_jnp = jnp.array(quad_nodes_np, dtype=jnp.float64)
    quad_weights_jnp = jnp.array(quad_weights_np, dtype=jnp.float64)
    
    # Mapped arguments (arrays to be iterated over by vmap)
    mu_z_batch_jnp = jnp.array(mu_z_batch_np, dtype=jnp.float64)
    sigma_z_batch_jnp = jnp.array(sigma_z_batch_np, dtype=jnp.float64)

    # --- Execute vmap'ped JAX function ---
    # Arguments for _vmapped_jit_marginalize_galaxies, in order defined by in_axes:
    # (xp_module, H0_val_jnp, mu_z_batch_jnp, sigma_z_batch_jnp, 
    #  dL_gw_samples_jnp, _quad_nodes_jnp, _quad_weights_jnp, 
    #  c_val, sigma_v_val, omega_m_val, 
    #  _jitted_lum_dist_calculator_func, _jitted_comoving_integral_func, 
    #  z_sigma_range, N_trapz_lum_dist)

    pipeline_logLs_vmapped_jax = _vmapped_jit_marginalize_galaxies(
        jnp,                         # xp_module
        H0_jnp,                      # H0_val_jnp
        mu_z_batch_jnp,              # mu_z_batch_jnp (mapped)
        sigma_z_batch_jnp,           # sigma_z_batch_jnp (mapped)
        dL_gw_samples_jnp,           # dL_gw_samples_jnp
        quad_nodes_jnp,              # _quad_nodes_jnp
        quad_weights_jnp,            # _quad_weights_jnp
        c_val_test,                  # c_val
        sigma_v_val_test,            # sigma_v_val
        omega_m_val_test,            # omega_m_val
        _jitted_static_lum_dist,     # _jitted_lum_dist_calculator_func
        _jitted_static_comoving_integral, # _jitted_comoving_integral_func
        z_sigma_range_test,          # z_sigma_range
        N_trapz_lum_dist_test        # N_trapz_lum_dist
    )
    pipeline_logLs_np_from_vmapped = np.array(pipeline_logLs_vmapped_jax)
    print(f"Pipeline vmap'ped JAX LogLs (converted to np): {pipeline_logLs_np_from_vmapped}")

    # --- Assertions ---
    assert pipeline_logLs_np_from_vmapped.shape == reference_logLs_np.shape, \
        f"Shape mismatch: JAX vmapped output shape {pipeline_logLs_np_from_vmapped.shape}, Reference NumPy shape {reference_logLs_np.shape}"
    assert np.all(np.isfinite(pipeline_logLs_np_from_vmapped)), \
        f"Pipeline vmap'ped JAX LogLs not all finite: {pipeline_logLs_np_from_vmapped}"
    assert np.all(np.isfinite(reference_logLs_np)), \
        f"Reference NumPy LogLs not all finite: {reference_logLs_np}"

    # Element-wise comparison, accounting for potential -inf values
    for i in range(len(reference_logLs_np)):
        ref_val = reference_logLs_np[i]
        jax_val = pipeline_logLs_np_from_vmapped[i]
        if np.isneginf(ref_val) and np.isneginf(jax_val):
            continue # Both are -inf, considered close
        assert np.allclose(jax_val, ref_val, rtol=1e-5, atol=1e-8), \
            f"Mismatch at index {i}: JAX vmapped={jax_val}, NumPy ref={ref_val}, Diff={jax_val - ref_val}"

# <<< END OF NEW TEST FUNCTION FOR VMAPPED JITTED VERSION >>> 

# <<< START OF NEW TEST FUNCTION FOR _marginalize_batch_galaxy_redshift (JAX PATH INTEGRATION) >>>
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed or not configured")
def test_marginalize_batch_galaxy_redshift_jax_path_accuracy():
    """
    Tests H0LogLikelihood._marginalize_batch_galaxy_redshift with JAX backend 
    to ensure correct integration of _vmapped_jit_marginalize_galaxies.
    Compares against the NumPy path of the same method.
    """
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    global jnp
    from gwsiren import h0_mcmc_analyzer # For default constants
    # No need to directly import the _vmapped function here, test is via class method

    # Test Parameters
    H0_test = 70.0
    mu_z_batch_np = np.array([0.05, 0.06, 0.07, 0.02, 0.15], dtype=np.float64)
    sigma_z_batch_np = np.array([0.005, 0.006, 0.004, 0.001, 0.015], dtype=np.float64)
    np.random.seed(42) # Consistent random samples
    dL_gw_samples_np = np.random.normal(loc=220, scale=20, size=100).astype(np.float64)
    
    # Parameters for H0LogLikelihood instantiation
    common_params = {
        "dL_gw_samples": dL_gw_samples_np,
        "host_galaxies_z": mu_z_batch_np, # Use batch data for instantiation
        "host_galaxies_mass_proxy": np.ones_like(mu_z_batch_np),
        "host_galaxies_z_err": sigma_z_batch_np,
        "c_val": h0_mcmc_analyzer.DEFAULT_C_LIGHT,
        "sigma_v": h0_mcmc_analyzer.DEFAULT_SIGMA_V_PEC,
        "omega_m_val": h0_mcmc_analyzer.DEFAULT_OMEGA_M,
        "n_quad_points": h0_mcmc_analyzer.DEFAULT_QUAD_POINTS,
        "z_sigma_range": h0_mcmc_analyzer.DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE
    }

    # --- JAX Path Execution ---
    analyzer_jax = H0LogLikelihood(xp=jnp, backend_name="jax", **common_params)
    # Call the method under test
    jax_results_jax_array = analyzer_jax._marginalize_batch_galaxy_redshift(
        H0_test, mu_z_batch_np, sigma_z_batch_np # Pass NumPy arrays, method should convert
    )
    jax_results_np = np.array(jax_results_jax_array)
    print(f"JAX Path _marginalize_batch_galaxy_redshift results: {jax_results_np}")

    # --- NumPy Path Reference Calculation ---
    analyzer_numpy = H0LogLikelihood(xp=np, backend_name="numpy", **common_params)
    # Call the same method, but it will use the NumPy internal path
    numpy_reference_results_np = analyzer_numpy._marginalize_batch_galaxy_redshift(
        H0_test, mu_z_batch_np, sigma_z_batch_np
    )
    print(f"NumPy Path _marginalize_batch_galaxy_redshift results: {numpy_reference_results_np}")

    # --- Assertions ---
    assert jax_results_np.shape == numpy_reference_results_np.shape, \
        f"Shape mismatch: JAX path shape {jax_results_np.shape}, NumPy path shape {numpy_reference_results_np.shape}"
    assert np.all(np.isfinite(jax_results_np)), \
        f"JAX path results not all finite: {jax_results_np}"
    assert np.all(np.isfinite(numpy_reference_results_np)), \
        f"NumPy path reference results not all finite: {numpy_reference_results_np}"

    # Element-wise comparison
    for i in range(len(numpy_reference_results_np)):
        ref_val = numpy_reference_results_np[i]
        jax_val = jax_results_np[i]
        if np.isneginf(ref_val) and np.isneginf(jax_val):
            continue
        assert np.allclose(jax_val, ref_val, rtol=1e-5, atol=1e-8), \
            f"Mismatch for method _marginalize_batch_galaxy_redshift at index {i}: JAX path={jax_val}, NumPy path={ref_val}, Diff={jax_val - ref_val}"

# <<< END OF NEW TEST FUNCTION FOR _marginalize_batch_galaxy_redshift (JAX PATH INTEGRATION) >>> 