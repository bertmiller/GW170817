import numpy as np
import sys
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import emcee

# Default Cosmological Parameters for Likelihood
DEFAULT_SIGMA_V_PEC = 250.0  # km/s, peculiar velocity uncertainty
DEFAULT_C_LIGHT = 299792.458 # km/s
DEFAULT_OMEGA_M = 0.31

# Default MCMC Parameters
DEFAULT_MCMC_N_DIM = 1
DEFAULT_MCMC_N_WALKERS = 32
DEFAULT_MCMC_N_STEPS = 6000
DEFAULT_MCMC_BURNIN = 1000
DEFAULT_MCMC_THIN_BY = 10
DEFAULT_MCMC_INITIAL_H0_MEAN = 70.0
DEFAULT_MCMC_INITIAL_H0_STD = 10.0
DEFAULT_H0_PRIOR_MIN = 10.0 # km/s/Mpc
DEFAULT_H0_PRIOR_MAX = 200.0 # km/s/Mpc

def get_log_likelihood_h0(
    dL_gw_samples, 
    host_galaxies_z, 
    sigma_v=DEFAULT_SIGMA_V_PEC, 
    c_val=DEFAULT_C_LIGHT, 
    omega_m_val=DEFAULT_OMEGA_M,
    h0_min=DEFAULT_H0_PRIOR_MIN,
    h0_max=DEFAULT_H0_PRIOR_MAX
):
    """
    Returns the log likelihood function for H0, marginalized over GW samples and host galaxies.

    Args:
        dL_gw_samples (np.array): Luminosity distance samples from GW event (N_samples,).
        host_galaxies_z (np.array): Redshifts of candidate host galaxies (N_hosts,).
        sigma_v (float): Peculiar velocity uncertainty (km/s).
        c_val (float): Speed of light (km/s).
        omega_m_val (float): Omega Matter for cosmological calculations.
        h0_min (float): Minimum value for H0 prior.
        h0_max (float): Maximum value for H0 prior.

    Returns:
        function: A log likelihood function `log_likelihood(theta)` where theta is [H0].
    """
    if dL_gw_samples is None or len(dL_gw_samples) == 0:
        raise ValueError("dL_gw_samples cannot be None or empty.")
    if host_galaxies_z is None or len(host_galaxies_z) == 0:
        raise ValueError("host_galaxies_z cannot be None or empty.")

    # Ensure host_galaxies_z is a numpy array for broadcasting
    z_values = np.asarray(host_galaxies_z)
    if z_values.ndim == 0: # if it was a single scalar
        z_values = np.array([z_values])

    def lum_dist_model(z, H0_val):
        cosmo = FlatLambdaCDM(H0=H0_val * u.km / u.s / u.Mpc, Om0=omega_m_val)
        return cosmo.luminosity_distance(z).value # Returns array if z is array

    def log_likelihood(theta):
        H0 = theta[0]
        if not (h0_min <= H0 <= h0_max):
            return -np.inf

        # Calculate model luminosity distances for all host galaxies at this H0
        # model_d will be an array of shape (N_hosts,)
        model_d_for_hosts = lum_dist_model(z_values, H0)
        
        # Calculate sigma_d for each host based on its model_d
        # sigma_d_val_for_hosts will be array of shape (N_hosts,)
        sigma_d_val_for_hosts = (model_d_for_hosts / c_val) * sigma_v
        # Prevent sigma_d from being too small or zero to avoid numerical issues
        sigma_d_val_for_hosts = np.maximum(sigma_d_val_for_hosts, 1e-9) 

        # Reshape dL_gw_samples to (N_samples, 1) for broadcasting
        # Reshape model_d_for_hosts and sigma_d_val_for_hosts to (1, N_hosts) for broadcasting
        # The result of norm.logpdf will be of shape (N_samples, N_hosts)
        # This calculates log P(dL_s | H0, z_i) for each sample s and each host i
        log_pdf_values = norm.logpdf(
            dL_gw_samples[:, None], 
            loc=model_d_for_hosts[None, :], 
            scale=sigma_d_val_for_hosts[None, :]
        )
        
        # Marginalize over GW samples (sum probabilities for each galaxy):
        # log P(data | H0, z_i) = log [ (1/N_samples) * sum_s P(dL_s | H0, z_i) ]
        # This is equivalent to logsumexp(log P(dL_s | H0, z_i)) - log(N_samples)
        # log_sum_over_gw_samples will be of shape (N_hosts,)
        log_sum_over_gw_samples = np.logaddexp.reduce(log_pdf_values, axis=0) - np.log(len(dL_gw_samples))
        
        # Marginalize over galaxies (sum probabilities for a given H0):
        # log P(data | H0) = log [ (1/N_hosts) * sum_i P(data | H0, z_i) ]
        # This is equivalent to logsumexp(log P(data | H0, z_i)) - log(N_hosts)
        # This assumes a uniform prior over the candidate host galaxies.
        total_log_likelihood = np.logaddexp.reduce(log_sum_over_gw_samples) - np.log(len(z_values))
        
        if not np.isfinite(total_log_likelihood):
            # Helps catch numerical issues (e.g., if all log_pdf_values were -inf)
            return -np.inf 
            
        return total_log_likelihood

    return log_likelihood

def run_mcmc_h0(
    log_likelihood_func, 
    event_name, # For logging
    n_walkers=DEFAULT_MCMC_N_WALKERS, 
    n_dim=DEFAULT_MCMC_N_DIM, 
    initial_h0_mean=DEFAULT_MCMC_INITIAL_H0_MEAN, 
    initial_h0_std=DEFAULT_MCMC_INITIAL_H0_STD, 
    n_steps=DEFAULT_MCMC_N_STEPS
):
    """
    Runs the MCMC sampler for H0.

    Args:
        log_likelihood_func (function): The log likelihood function `log_likelihood(theta)`.
        event_name (str): Name of the event for logging.
        n_walkers (int): Number of MCMC walkers.
        n_dim (int): Number of dimensions (should be 1 for H0).
        initial_h0_mean (float): Mean for initializing walker positions.
        initial_h0_std (float): Standard deviation for initializing walker positions.
        n_steps (int): Number of MCMC steps.

    Returns:
        emcee.EnsembleSampler: The MCMC sampler object after running, or None on failure.
    """
    print(f"\\nRunning MCMC for H0 on event {event_name} ({n_steps} steps, {n_walkers} walkers)...")
    # Initial positions for walkers, centered around a plausible H0
    walkers0 = initial_h0_mean + initial_h0_std * np.random.randn(n_walkers, n_dim)

    # Ensure walkers0 is 2D: (n_walkers, n_dim)
    if walkers0.ndim == 1:
        walkers0 = walkers0[:, np.newaxis]
        
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood_func, moves=emcee.moves.StretchMove())
    
    try:
        sampler.run_mcmc(walkers0, n_steps, progress=True)
        print(f"MCMC run completed for {event_name}.")
        return sampler
    except ValueError as ve:
        print(f"❌ ValueError during MCMC for {event_name}: {ve}")
        print("  This can happen if the likelihood consistently returns -inf, check priors or input data.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during MCMC for {event_name}: {e}")
        return None
        
def process_mcmc_samples(sampler, event_name, burnin=DEFAULT_MCMC_BURNIN, thin_by=DEFAULT_MCMC_THIN_BY, n_dim=DEFAULT_MCMC_N_DIM):
    """
    Processes MCMC samples: applies burn-in and thinning.

    Args:
        sampler (emcee.EnsembleSampler): The MCMC sampler object after running.
        event_name (str): Name of the event for logging.
        burnin (int): Number of burn-in steps to discard.
        thin_by (int): Factor to thin the samples by.
        n_dim (int): Expected number of dimensions in the chain.

    Returns:
        np.array: Flattened array of H0 samples after burn-in and thinning, or None if processing fails.
    """
    if sampler is None:
        print(f"⚠️ Sampler object is None for {event_name}. Cannot process MCMC samples.")
        return None

    print(f"Processing MCMC samples for {event_name} (burn-in: {burnin}, thin: {thin_by})...")
    try:
        # get_chain shape: (n_steps, n_walkers, n_dim)
        # flat=True gives shape: (n_steps_after_discard_and_thin * n_walkers, n_dim)
        flat_samples = sampler.get_chain(discard=burnin, thin=thin_by, flat=True)
        
        if flat_samples.shape[0] == 0:
            print(f"⚠️ MCMC for {event_name} resulted in NO valid samples after burn-in ({burnin}) and thinning ({thin_by}).")
            print(f"  Original chain length before discard: {sampler.get_chain().shape[0]}")
            return None

        if n_dim == 1 and flat_samples.ndim == 2 and flat_samples.shape[1] == 1:
             processed_samples = flat_samples[:,0] # Extract the single parameter (H0)
        elif n_dim > 1 and flat_samples.ndim == 2 and flat_samples.shape[1] == n_dim:
            print(f"  Note: MCMC had {n_dim} dimensions. Returning all dimensions after processing.")
            processed_samples = flat_samples # Keep as is if multi-dimensional and correctly shaped
        elif flat_samples.ndim == 1 and n_dim ==1: # Already flat and 1D
            processed_samples = flat_samples
        else:
            print(f"⚠️ Unexpected shape for flat_samples: {flat_samples.shape}. Expected ({'*', n_dim}). Cannot safely extract H0.")
            return None

        print(f"  Successfully processed MCMC samples for {event_name}. Number of samples: {len(processed_samples)}.")
        return processed_samples
        
    except Exception as e:
        print(f"❌ Error processing MCMC chain for {event_name}: {e}")
        return None

if __name__ == '__main__':
    print("--- Testing h0_mcmc_analyzer.py ---")
    
    # Mock data for testing likelihood and MCMC
    mock_event = "GW_MOCK_H0_TEST"
    mock_dL_gw = np.random.normal(loc=700, scale=70, size=1000) # Mpc
    mock_host_zs = np.array([0.1, 0.12, 0.09]) # Redshifts of a few mock galaxies

    # 1. Test get_log_likelihood_h0
    print("\nTest 1: Getting log likelihood function...")
    try:
        log_like_func = get_log_likelihood_h0(mock_dL_gw, mock_host_zs)
        # Test the likelihood function with some H0 values
        h0_test_values = [60.0, 70.0, 80.0]
        print(f"  Log likelihood values for H0={h0_test_values}:")
        for h0_val in h0_test_values:
            ll = log_like_func([h0_val])
            print(f"    H0 = {h0_val:.1f}: logL = {ll:.2f}")
            assert np.isfinite(ll), f"LogL not finite for H0={h0_val}"
        print("  Log likelihood function seems operational.")

        # Test prior boundaries
        ll_low = log_like_func([DEFAULT_H0_PRIOR_MIN - 1])
        assert ll_low == -np.inf, "Prior min boundary failed"
        ll_high = log_like_func([DEFAULT_H0_PRIOR_MAX + 1])
        assert ll_high == -np.inf, "Prior max boundary failed"
        print(f"  Prior boundaries (H0_min={DEFAULT_H0_PRIOR_MIN}, H0_max={DEFAULT_H0_PRIOR_MAX}) correctly applied.")

    except ValueError as ve:
        print(f"  Error in Test 1 (get_log_likelihood_h0): {ve}")
    except Exception as e:
        print(f"  Unexpected error in Test 1 (get_log_likelihood_h0): {e}")

    # 2. Test run_mcmc_h0 and process_mcmc_samples
    print("\nTest 2: Running MCMC and processing samples...")
    # Reduce steps for faster testing
    test_mcmc_steps = 200 
    test_mcmc_burnin = 50
    test_mcmc_walkers = 8 # Fewer walkers for test

    if 'log_like_func' in locals(): # Proceed if likelihood function was created
        sampler = run_mcmc_h0(
            log_like_func, 
            mock_event,
            n_walkers=test_mcmc_walkers,
            n_steps=test_mcmc_steps
        )

        if sampler:
            h0_samples = process_mcmc_samples(
                sampler, 
                mock_event,
                burnin=test_mcmc_burnin, 
                thin_by=2 # Small thin factor for test
            )
            if h0_samples is not None and len(h0_samples) > 0:
                print(f"  Successfully ran MCMC and processed samples for {mock_event}.")
                print(f"  Number of H0 samples obtained: {len(h0_samples)}")
                print(f"  H0 mean: {np.mean(h0_samples):.2f}, H0 std: {np.std(h0_samples):.2f}")
                # Expect H0 mean to be somewhat related to initial mean if likelihood is reasonable
                # assert DEFAULT_MCMC_INITIAL_H0_MEAN - 3*DEFAULT_MCMC_INITIAL_H0_STD < np.mean(h0_samples) < DEFAULT_MCMC_INITIAL_H0_MEAN + 3*DEFAULT_MCMC_INITIAL_H0_STD
            elif h0_samples is not None and len(h0_samples) == 0:
                print("  MCMC processing resulted in zero samples. Check burn-in/thinning or chain length.")
            else:
                print("  MCMC processing failed to return valid samples.")
        else:
            print("  MCMC run failed or returned no sampler. Cannot test processing.")
    else:
        print("  Log likelihood function not available. Skipping MCMC run and processing tests.")

    # Test with problematic inputs for get_log_likelihood_h0
    print("\nTest 3: Edge cases for get_log_likelihood_h0")
    try:
        get_log_likelihood_h0(None, mock_host_zs)
    except ValueError as e:
        print(f"  Correctly caught error for None dL samples: {e}")
    try:
        get_log_likelihood_h0(mock_dL_gw, [])
    except ValueError as e:
        print(f"  Correctly caught error for empty host_zs: {e}")

    print("\n--- Finished testing h0_mcmc_analyzer.py ---") 