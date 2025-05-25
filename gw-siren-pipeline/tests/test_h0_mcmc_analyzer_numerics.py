import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

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