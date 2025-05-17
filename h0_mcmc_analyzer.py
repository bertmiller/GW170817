import numpy as np
import sys
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import emcee
import logging
import cProfile
import pstats
import io

logger = logging.getLogger(__name__)

# Default Cosmological Parameters for Likelihood
DEFAULT_SIGMA_V_PEC = 250.0  # km/s, peculiar velocity uncertainty
DEFAULT_C_LIGHT = 299792.458 # km/s
DEFAULT_OMEGA_M = 0.31

# Default MCMC Parameters
DEFAULT_MCMC_N_DIM = 1
DEFAULT_MCMC_N_WALKERS = 14
DEFAULT_MCMC_N_STEPS = 1000
DEFAULT_MCMC_BURNIN = 200
DEFAULT_MCMC_THIN_BY = 10
DEFAULT_MCMC_INITIAL_H0_MEAN = 70.0
DEFAULT_MCMC_INITIAL_H0_STD = 10.0
DEFAULT_H0_PRIOR_MIN = 10.0 # km/s/Mpc
DEFAULT_H0_PRIOR_MAX = 200.0 # km/s/Mpc

class H0LogLikelihood:
    def __init__(self, dL_gw_samples, host_galaxies_z, 
                 sigma_v=DEFAULT_SIGMA_V_PEC, 
                 c_val=DEFAULT_C_LIGHT, 
                 omega_m_val=DEFAULT_OMEGA_M,
                 h0_min=DEFAULT_H0_PRIOR_MIN,
                 h0_max=DEFAULT_H0_PRIOR_MAX,
                 use_vectorized_likelihood=False):
        
        if dL_gw_samples is None or len(dL_gw_samples) == 0:
            raise ValueError("dL_gw_samples cannot be None or empty.")
        if host_galaxies_z is None or len(host_galaxies_z) == 0:
            raise ValueError("host_galaxies_z cannot be None or empty.")

        self.dL_gw_samples = np.asarray(dL_gw_samples)
        # Ensure host_galaxies_z is a numpy array for broadcasting
        self.z_values = np.asarray(host_galaxies_z)
        if self.z_values.ndim == 0: # if it was a single scalar
            self.z_values = np.array([self.z_values])
        
        self.sigma_v = sigma_v
        self.c_val = c_val
        self.omega_m_val = omega_m_val
        self.h0_min = h0_min
        self.h0_max = h0_max
        self.use_vectorized_likelihood = use_vectorized_likelihood

    def _lum_dist_model(self, z, H0_val):
        cosmo = FlatLambdaCDM(H0=H0_val * u.km / u.s / u.Mpc, Om0=self.omega_m_val)
        return cosmo.luminosity_distance(z).value

    def __call__(self, theta):
        H0 = theta[0]
        # logger.debug(f"H0LogLikelihood.__call__ with H0 = {H0}, vectorized = {self.use_vectorized_likelihood}")

        if not (self.h0_min <= H0 <= self.h0_max):
            # logger.debug(f"H0 = {H0} is outside prior range ({self.h0_min}, {self.h0_max}). Returning -inf.")
            return -np.inf

        try: 
            model_d_for_hosts = self._lum_dist_model(self.z_values, H0)
            # logger.debug(f"model_d_for_hosts (first 5 if available): {model_d_for_hosts[:min(5, len(model_d_for_hosts))]}") 
            if np.any(~np.isfinite(model_d_for_hosts)): 
                # logger.debug(f"Non-finite values in model_d_for_hosts for H0 = {H0}. Example: {model_d_for_hosts[~np.isfinite(model_d_for_hosts)][:min(5, len(model_d_for_hosts[~np.isfinite(model_d_for_hosts)]))]}") 
                return -np.inf 
        except Exception as e: 
            # logger.debug(f"EXCEPTION in _lum_dist_model for H0 = {H0}: {e}") 
            return -np.inf 
        
        sigma_d_val_for_hosts = (model_d_for_hosts / self.c_val) * self.sigma_v
        sigma_d_val_for_hosts = np.maximum(sigma_d_val_for_hosts, 1e-9) 
        # logger.debug(f"sigma_d_val_for_hosts (first 5 if available): {sigma_d_val_for_hosts[:min(5, len(sigma_d_val_for_hosts))]}") 
        if np.any(~np.isfinite(sigma_d_val_for_hosts)): 
            # logger.debug(f"Non-finite values in sigma_d_val_for_hosts for H0 = {H0}.") 
            return -np.inf

        log_sum_over_gw_samples = np.array([-np.inf]) # Placeholder for now

        if self.use_vectorized_likelihood:
            # --- Fully Vectorized Alternative ---
            # logger.debug("Using fully vectorized likelihood path.")
            try:
                # Reshape for broadcasting: 
                # self.dL_gw_samples: (N_samples,) -> (N_samples, 1)
                # model_d_for_hosts: (N_hosts,) -> (1, N_hosts)
                # sigma_d_val_for_hosts: (N_hosts,) -> (1, N_hosts)
                log_pdf_values_full = norm.logpdf(
                    self.dL_gw_samples[:, np.newaxis],
                    loc=model_d_for_hosts[np.newaxis, :],
                    scale=sigma_d_val_for_hosts[np.newaxis, :]
                ) # Shape: (N_samples, N_hosts)
            except Exception as e:
                # logger.debug(f"EXCEPTION in norm.logpdf (vectorized) for H0 = {H0}: {e}")
                # If the entire logpdf calculation fails, it likely means a broad issue
                # (e.g. H0 way off, leading to bad model_d_for_hosts for ALL hosts)
                # In this case, returning -np.inf for the whole likelihood is appropriate.
                return -np.inf

            # Handle potential NaNs by converting them to -np.inf.
            # This allows logaddexp.reduce to correctly ignore them unless an entire column is -np.inf.
            log_pdf_values_full[np.isnan(log_pdf_values_full)] = -np.inf

            # Check if any galaxy (column) resulted in all -np.inf values for its log_pdf terms.
            # This can happen if a specific galaxy's model_d or sigma_d was problematic.
            # all_inf_columns = np.all(log_pdf_values_full == -np.inf, axis=0)

            log_sum_over_gw_samples = np.logaddexp.reduce(log_pdf_values_full, axis=0) - np.log(len(self.dL_gw_samples))
            
            # After reduction, if any element in log_sum_over_gw_samples is -np.inf 
            # (e.g., because its corresponding column in log_pdf_values_full was all -np.inf, 
            # or because logaddexp.reduce itself resulted in -inf due to underflow with very small numbers),
            # it will correctly propagate.
            # No specific check for all_inf_columns is strictly needed here because logaddexp.reduce handles it.
            
            # logger.debug(f"log_sum_over_gw_samples (vectorized) shape: {log_sum_over_gw_samples.shape}, any non-finite: {np.any(~np.isfinite(log_sum_over_gw_samples))}")
            # Example: Check first 5 values if problematic
            if np.any(~np.isfinite(log_sum_over_gw_samples)):
                # logger.debug(f"  Non-finite log_sum_over_gw_samples (vectorized): {log_sum_over_gw_samples[~np.isfinite(log_sum_over_gw_samples)][:5]}")
                return -np.inf

        else:
            # --- Current Memory-Efficient Loop ---
            # logger.debug("Using memory-efficient loop path.")
            log_P_data_H0_zi_terms = np.zeros(len(self.z_values))
            for i in range(len(self.z_values)):
                current_model_d = model_d_for_hosts[i]
                current_sigma_d = sigma_d_val_for_hosts[i]
                try:
                    log_pdf_for_one_galaxy = norm.logpdf(
                        self.dL_gw_samples,
                        loc=current_model_d,
                        scale=current_sigma_d
                    )
                except Exception as e:
                    # logger.debug(f"EXCEPTION in norm.logpdf (loop) for H0 = {H0}, galaxy_idx = {i}: {e}")
                    log_pdf_for_one_galaxy = np.full(len(self.dL_gw_samples), -np.inf)

                if np.any(~np.isfinite(log_pdf_for_one_galaxy)):
                    log_P_data_H0_zi_terms[i] = -np.inf
                else:
                    log_P_data_H0_zi_terms[i] = np.logaddexp.reduce(log_pdf_for_one_galaxy) - np.log(len(self.dL_gw_samples))
            log_sum_over_gw_samples = log_P_data_H0_zi_terms
            # logger.debug(f"log_sum_over_gw_samples (iterative per galaxy) shape: {log_sum_over_gw_samples.shape}, any non-finite: {np.any(~np.isfinite(log_sum_over_gw_samples))}")

        total_log_likelihood = np.logaddexp.reduce(log_sum_over_gw_samples) - np.log(len(self.z_values))
        # logger.debug(f"total_log_likelihood = {total_log_likelihood}")
        
        if not np.isfinite(total_log_likelihood):
            # logger.debug(f"total_log_likelihood is not finite ({total_log_likelihood}) for H0 = {H0}. Returning -inf.")
            return -np.inf 
            
        return total_log_likelihood

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
    Returns an instance of the H0LogLikelihood class, dynamically choosing
    between looped and vectorized likelihood calculation based on data size.

    Args:
        dL_gw_samples (np.array): Luminosity distance samples from GW event (N_samples,).
        host_galaxies_z (np.array): Redshifts of candidate host galaxies (N_hosts,).
        sigma_v (float): Peculiar velocity uncertainty (km/s).
        c_val (float): Speed of light (km/s).
        omega_m_val (float): Omega Matter for cosmological calculations.
        h0_min (float): Minimum value for H0 prior.
        h0_max (float): Maximum value for H0 prior.

    Returns:
        H0LogLikelihood: An instance of the H0LogLikelihood class.
    """
    # Dynamic switch for vectorization based on memory
    # 4 GB threshold = 4 * 1024^3 bytes
    # float64 takes 8 bytes
    MEMORY_THRESHOLD_BYTES = 4 * (1024**3)
    BYTES_PER_ELEMENT = 8 # for float64
    max_elements_for_vectorization = MEMORY_THRESHOLD_BYTES / BYTES_PER_ELEMENT

    n_gw_samples = len(dL_gw_samples)
    # Ensure host_galaxies_z is treated as an array for len()
    _host_galaxies_z = np.asarray(host_galaxies_z)
    if _host_galaxies_z.ndim == 0:
        n_hosts = 1
    else:
        n_hosts = len(_host_galaxies_z)
        
    current_elements = n_gw_samples * n_hosts

    should_use_vectorized = current_elements <= max_elements_for_vectorization

    if should_use_vectorized:
        logger.info(
            f"Using VECTORIZED likelihood: N_samples ({n_gw_samples}) * N_hosts ({n_hosts}) = {current_elements} elements. "
            f"Threshold: {max_elements_for_vectorization:.0f} elements ({MEMORY_THRESHOLD_BYTES / (1024**3):.0f} GB)."
        )
    else:
        logger.info(
            f"Using LOOPED likelihood (memory efficient): N_samples ({n_gw_samples}) * N_hosts ({n_hosts}) = {current_elements} elements. "
            f"Exceeds threshold of {max_elements_for_vectorization:.0f} elements ({MEMORY_THRESHOLD_BYTES / (1024**3):.0f} GB)."
        )

    return H0LogLikelihood(
        dL_gw_samples, host_galaxies_z, 
        sigma_v, c_val, omega_m_val, 
        h0_min, h0_max,
        use_vectorized_likelihood=should_use_vectorized 
    )

def get_log_likelihood_h0_vectorized( # Helper to specifically get vectorized
    dL_gw_samples,
    host_galaxies_z,
    sigma_v=DEFAULT_SIGMA_V_PEC,
    c_val=DEFAULT_C_LIGHT,
    omega_m_val=DEFAULT_OMEGA_M,
    h0_min=DEFAULT_H0_PRIOR_MIN,
    h0_max=DEFAULT_H0_PRIOR_MAX
):
    return H0LogLikelihood(
        dL_gw_samples, host_galaxies_z,
        sigma_v, c_val, omega_m_val,
        h0_min, h0_max,
        use_vectorized_likelihood=True
    )

def run_mcmc_h0(
    log_likelihood_func, 
    event_name, # For logging
    n_walkers=DEFAULT_MCMC_N_WALKERS, 
    n_dim=DEFAULT_MCMC_N_DIM, 
    initial_h0_mean=DEFAULT_MCMC_INITIAL_H0_MEAN, 
    initial_h0_std=DEFAULT_MCMC_INITIAL_H0_STD, 
    n_steps=DEFAULT_MCMC_N_STEPS,
    pool=None  # Add new pool parameter
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
        pool (object, optional): A pool object for parallelization (e.g., from multiprocessing or emcee.interruptible_pool).

    Returns:
        emcee.EnsembleSampler: The MCMC sampler object after running, or None on failure.
    """
    logger.info(f"Running MCMC for H0 on event {event_name} ({n_steps} steps, {n_walkers} walkers)...")
    # Initial positions for walkers, centered around a plausible H0
    walkers0 = initial_h0_mean + initial_h0_std * np.random.randn(n_walkers, n_dim)

    # Ensure walkers0 is 2D: (n_walkers, n_dim)
    if walkers0.ndim == 1:
        walkers0 = walkers0[:, np.newaxis]
        
    sampler = emcee.EnsembleSampler(
        n_walkers, 
        n_dim, 
        log_likelihood_func, 
        moves=emcee.moves.StretchMove(),
        pool=pool  # Pass the pool to the sampler
    )
    
    try:
        sampler.run_mcmc(walkers0, n_steps, progress=True)
        logger.info(f"MCMC run completed for {event_name}.")
        return sampler
    except ValueError as ve:
        logger.error(f"❌ ValueError during MCMC for {event_name}: {ve}")
        logger.error("  This can happen if the likelihood consistently returns -inf, check priors or input data.")
        return None
    except Exception as e:
        logger.exception(f"❌ An unexpected error occurred during MCMC for {event_name}: {e}")
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
        logger.warning(f"⚠️ Sampler object is None for {event_name}. Cannot process MCMC samples.")
        return None

    logger.info(f"Processing MCMC samples for {event_name} (burn-in: {burnin}, thin: {thin_by})...")
    try:
        # get_chain shape: (n_steps, n_walkers, n_dim)
        # flat=True gives shape: (n_steps_after_discard_and_thin * n_walkers, n_dim)
        flat_samples = sampler.get_chain(discard=burnin, thin=thin_by, flat=True)
        
        if flat_samples.shape[0] == 0:
            logger.warning(f"⚠️ MCMC for {event_name} resulted in NO valid samples after burn-in ({burnin}) and thinning ({thin_by}).")
            logger.warning(f"  Original chain length before discard: {sampler.get_chain().shape[0]}")
            return None

        if n_dim == 1 and flat_samples.ndim == 2 and flat_samples.shape[1] == 1:
             processed_samples = flat_samples[:,0] # Extract the single parameter (H0)
        elif n_dim > 1 and flat_samples.ndim == 2 and flat_samples.shape[1] == n_dim:
            logger.info(f"  Note: MCMC had {n_dim} dimensions. Returning all dimensions after processing.")
            processed_samples = flat_samples # Keep as is if multi-dimensional and correctly shaped
        elif flat_samples.ndim == 1 and n_dim ==1: # Already flat and 1D
            processed_samples = flat_samples
        else:
            logger.warning(f"⚠️ Unexpected shape for flat_samples: {flat_samples.shape}. Expected ({'*', n_dim}). Cannot safely extract H0.")
            return None

        logger.info(f"  Successfully processed MCMC samples for {event_name}. Number of samples: {len(processed_samples)}.")
        return processed_samples
        
    except Exception as e:
        logger.exception(f"❌ Error processing MCMC chain for {event_name}: {e}")
        return None

def profile_likelihood_call(log_like_func_to_profile, h0_values_to_test, n_calls=10):
    """Helper function to profile the likelihood call multiple times."""
    logger.info(f"Profiling {log_like_func_to_profile.__class__.__name__} ({'vectorized' if getattr(log_like_func_to_profile, 'use_vectorized_likelihood', False) else 'looped'})...")
    for h0_val in h0_values_to_test:
        for _ in range(n_calls):
            _ = log_like_func_to_profile([h0_val]) # Call the likelihood

if __name__ == '__main__':
    # Configure basic logging for standalone testing of this module
    logging.basicConfig(
        level=logging.INFO, # Changed from DEBUG to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.info("--- Testing h0_mcmc_analyzer.py ---")
    
    # Mock data for testing likelihood and MCMC
    mock_event = "GW_MOCK_H0_TEST"
    mock_dL_gw = np.random.normal(loc=700, scale=70, size=1000) # Mpc
    mock_host_zs = np.array([0.1, 0.12, 0.09, 0.15, 0.11] * 20) # N_hosts = 100

    h0_test_values_for_profiling = [65.0, 70.0, 75.0]

    # 1. Test get_log_likelihood_h0
    logger.info("\nTest 1: Getting log likelihood function...")
    try:
        log_like_func = get_log_likelihood_h0(mock_dL_gw, mock_host_zs) # Now uses dynamic switching
        # Test the likelihood function with some H0 values
        h0_test_values = [60.0, 70.0, 80.0]
        logger.info(f"  Log likelihood values for H0={h0_test_values}:")
        for h0_val in h0_test_values:
            ll = log_like_func([h0_val])
            logger.info(f"    H0 = {h0_val:.1f}: logL = {ll:.2f}")
            assert np.isfinite(ll), f"LogL not finite for H0={h0_val}"
        logger.info("  Log likelihood function seems operational.")

        # Test prior boundaries
        ll_low = log_like_func([DEFAULT_H0_PRIOR_MIN - 1])
        assert ll_low == -np.inf, "Prior min boundary failed"
        ll_high = log_like_func([DEFAULT_H0_PRIOR_MAX + 1])
        assert ll_high == -np.inf, "Prior max boundary failed"
        logger.info(f"  Prior boundaries (H0_min={DEFAULT_H0_PRIOR_MIN}, H0_max={DEFAULT_H0_PRIOR_MAX}) correctly applied.")

        # Test with explicitly vectorized version for comparison if needed, or rely on dynamic one
        log_like_func_explicit_vec = get_log_likelihood_h0_vectorized(mock_dL_gw, mock_host_zs)
        ll_vec_explicit = log_like_func_explicit_vec([70.0])
        ll_dynamic = log_like_func([70.0]) # log_like_func should be vectorized for mock data size
        assert np.isclose(ll_vec_explicit, ll_dynamic), \
            f"Explicitly vectorized ({ll_vec_explicit}) and dynamically chosen vectorized ({ll_dynamic}) paths should give same result"
        logger.info(f"  Explicitly vectorized path gives logL={ll_vec_explicit:.2f}, Dynamically chosen path gives logL={ll_dynamic:.2f} (vectorized expected for this data size)")

    except ValueError as ve:
        logger.error(f"  Error in Test 1 (get_log_likelihood_h0): {ve}")
    except Exception as e:
        logger.exception(f"  Unexpected error in Test 1 (get_log_likelihood_h0): {e}")

    # 2. Test run_mcmc_h0 and process_mcmc_samples
    logger.info("\nTest 2: Running MCMC and processing samples...")
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
                logger.info(f"  Successfully ran MCMC and processed samples for {mock_event}.")
                logger.info(f"  Number of H0 samples obtained: {len(h0_samples)}")
                logger.info(f"  H0 mean: {np.mean(h0_samples):.2f}, H0 std: {np.std(h0_samples):.2f}")
                # Expect H0 mean to be somewhat related to initial mean if likelihood is reasonable
                # assert DEFAULT_MCMC_INITIAL_H0_MEAN - 3*DEFAULT_MCMC_INITIAL_H0_STD < np.mean(h0_samples) < DEFAULT_MCMC_INITIAL_H0_MEAN + 3*DEFAULT_MCMC_INITIAL_H0_STD
            elif h0_samples is not None and len(h0_samples) == 0:
                logger.warning("  MCMC processing resulted in zero samples. Check burn-in/thinning or chain length.")
            else:
                logger.error("  MCMC processing failed to return valid samples.")
        else:
            logger.error("  MCMC run failed or returned no sampler. Cannot test processing.")
    else:
        logger.error("  Log likelihood function not available. Skipping MCMC run and processing tests.")

    # Test with problematic inputs for get_log_likelihood_h0
    logger.info("\nTest 3: Edge cases for get_log_likelihood_h0")
    try:
        get_log_likelihood_h0(None, mock_host_zs)
    except ValueError as e:
        logger.info(f"  Correctly caught error for None dL samples: {e}")
    try:
        get_log_likelihood_h0(mock_dL_gw, [])
    except ValueError as e:
        logger.info(f"  Correctly caught error for empty host_zs: {e}")

    # 4. Profiling Section
    logger.info("\nTest 4: Profiling the __call__ method...")
    if 'log_like_func' in locals():
        # Profile the original looped version
        pr_loop = cProfile.Profile()
        pr_loop.enable()
        profile_likelihood_call(log_like_func, h0_test_values_for_profiling, n_calls=5)
        pr_loop.disable()
        
        s_loop = io.StringIO()
        ps_loop = pstats.Stats(pr_loop, stream=s_loop).sort_stats('cumulative')
        ps_loop.print_stats(30) # Print top 30 cumulative time consumers
        logger.info("\n--- cProfile results for Looped Likelihood ---")
        print(s_loop.getvalue())

        # Profile the dynamically chosen version (expected to be vectorized for mock data)
        # log_like_func_vectorized = get_log_likelihood_h0_vectorized(mock_dL_gw, mock_host_zs) # Keep this for explicit vectorization test
        # For profiling, we rely on log_like_func which is now dynamic.
        # To ensure we profile both paths, we might need to force one to be looped if mock data is small.
        # For now, let's assume mock_dL_gw * mock_host_zs is small enough for vectorization.
        # And we need an instance that is FORCED to be looped for comparison in profiling.

        log_like_func_looped_for_profiling = H0LogLikelihood(
            mock_dL_gw, mock_host_zs, 
            use_vectorized_likelihood=False # Force loop for this profiling instance
        )
        log_like_func_vectorized_for_profiling = H0LogLikelihood(
            mock_dL_gw, mock_host_zs,
            use_vectorized_likelihood=True # Force vectorization for this profiling instance
        )

        logger.info("Profiling dynamically chosen (expected vectorized) likelihood path...")
        pr_vec = cProfile.Profile()
        pr_vec.enable()
        profile_likelihood_call(log_like_func, h0_test_values_for_profiling, n_calls=5) # log_like_func is dynamic
        pr_vec.disable()

        s_vec = io.StringIO()
        ps_vec = pstats.Stats(pr_vec, stream=s_vec).sort_stats('cumulative')
        ps_vec.print_stats(30)
        logger.info("\n--- cProfile results for Dynamically Chosen (expected Vectorized) Likelihood ---")
        print(s_vec.getvalue())
    else:
        logger.error("  Log likelihood function not available. Skipping profiling.")

    # 5. Timeit Benchmarking Section
    logger.info("\nTest 5: Benchmarking with timeit...")
    if 'log_like_func' in locals() and 'log_like_func_vectorized' in locals():
        import timeit
        
        # For timeit, we want to compare an explicitly looped vs explicitly vectorized (or dynamically vectorized)
        # log_like_func is now dynamic. Let's create specific instances for timeit.
        timeit_looped_instance = H0LogLikelihood(mock_dL_gw, mock_host_zs, use_vectorized_likelihood=False)
        timeit_vectorized_instance = H0LogLikelihood(mock_dL_gw, mock_host_zs, use_vectorized_likelihood=True)
        # Or use the dynamic one for vectorized, assuming it picks vectorization for mock data:
        # timeit_vectorized_instance = get_log_likelihood_h0(mock_dL_gw, mock_host_zs)

        n_timeit_runs = 100 # Number of times to execute the statement for timing
        n_timeit_repeat = 5 # Number of times to repeat the timing trial

        # Setup for timeit: make sure the objects are accessible
        # We'll call the __call__ method on existing instances.
        # The `glob` argument to timeit.repeat makes these available.
        timeit_globals = {
            "log_like_func_looped": timeit_looped_instance, 
            "log_like_func_vec": timeit_vectorized_instance, 
            "h0_test_val": [70.0] # A sample H0 value
        }

        logger.info(f"  Timing looped version ({n_timeit_runs} calls, {n_timeit_repeat} repeats)...")
        looped_times = timeit.repeat(
            "log_like_func_looped(h0_test_val)", 
            globals=timeit_globals,
            number=n_timeit_runs, 
            repeat=n_timeit_repeat
        )
        min_looped_time = min(looped_times) / n_timeit_runs
        logger.info(f"    Min time per call (looped): {min_looped_time*1e6:.2f} microseconds")

        logger.info(f"  Timing vectorized version ({n_timeit_runs} calls, {n_timeit_repeat} repeats)...")
        vectorized_times = timeit.repeat(
            "log_like_func_vec(h0_test_val)", 
            globals=timeit_globals,
            number=n_timeit_runs, 
            repeat=n_timeit_repeat
        )
        min_vectorized_time = min(vectorized_times) / n_timeit_runs
        logger.info(f"    Min time per call (vectorized): {min_vectorized_time*1e6:.2f} microseconds")
        
        if min_vectorized_time < min_looped_time:
            logger.info(f"    Vectorized is approx {min_looped_time/min_vectorized_time:.2f}x faster.")
        else:
            logger.info(f"    Looped is approx {min_vectorized_time/min_looped_time:.2f}x faster (or similar speed).")
            
    else:
        logger.error("  Log likelihood functions not available for timeit benchmarking.")

    # 6. Profiling Full MCMC Run
    logger.info("\nTest 6: Profiling full MCMC runs...")
    if True: # Simpler condition for now, assuming mock data setup is always done
        # For MCMC profiling, we need one instance that is definitely looped and one that is 
        # definitely vectorized (or dynamically chooses vectorization, which it should for mock data)
        
        mcmc_log_like_looped = H0LogLikelihood(
            mock_dL_gw, mock_host_zs, 
            use_vectorized_likelihood=False # Force loop
        )
        # The main log_like_func from Test 1 is already dynamically determined.
        # For mock_dL_gw (1000) and mock_host_zs (100), 1000*100 = 100,000 elements.
        # 4GB / 8 bytes = 536,870,912 elements. So it SHOULD be vectorized.
        mcmc_log_like_dynamic_vectorized = get_log_likelihood_h0(mock_dL_gw, mock_host_zs)

        # Profile MCMC with Looped Likelihood
        logger.info("  Profiling MCMC with (forced) looped likelihood...")
        pr_mcmc_loop = cProfile.Profile()
        pr_mcmc_loop.enable()
        sampler_loop = run_mcmc_h0(
            mcmc_log_like_looped, 
            mock_event + "_LoopedProf",
            n_walkers=test_mcmc_walkers,
            n_steps=test_mcmc_steps
        )
        pr_mcmc_loop.disable()
        if sampler_loop:
            logger.info(f"    MCMC with looped likelihood finished. Processing {mock_event}_LoopedProf samples...")
            _ = process_mcmc_samples(sampler_loop, mock_event + "_LoopedProf", burnin=test_mcmc_burnin, thin_by=2)
        s_mcmc_loop = io.StringIO()
        ps_mcmc_loop = pstats.Stats(pr_mcmc_loop, stream=s_mcmc_loop).sort_stats('cumulative')
        ps_mcmc_loop.print_stats(30) # Print top 30 cumulative time consumers
        logger.info("\n--- cProfile results for MCMC with Looped Likelihood ---")
        print(s_mcmc_loop.getvalue())

        # Profile MCMC with Vectorized Likelihood
        logger.info("  Profiling MCMC with (dynamically chosen, expected vectorized) likelihood...")
        pr_mcmc_vec = cProfile.Profile()
        pr_mcmc_vec.enable()
        sampler_vec = run_mcmc_h0(
            mcmc_log_like_dynamic_vectorized, 
            mock_event + "_VectorizedProf",
            n_walkers=test_mcmc_walkers,
            n_steps=test_mcmc_steps
        )
        pr_mcmc_vec.disable()
        if sampler_vec:
            logger.info(f"    MCMC with vectorized likelihood finished. Processing {mock_event}_VectorizedProf samples...")
            _ = process_mcmc_samples(sampler_vec, mock_event + "_VectorizedProf", burnin=test_mcmc_burnin, thin_by=2)
        s_mcmc_vec = io.StringIO()
        ps_mcmc_vec = pstats.Stats(pr_mcmc_vec, stream=s_mcmc_vec).sort_stats('cumulative')
        ps_mcmc_vec.print_stats(30)
        logger.info("\n--- cProfile results for MCMC with Vectorized Likelihood ---")
        print(s_mcmc_vec.getvalue())
    else:
        logger.error("  Log likelihood functions not available for MCMC profiling.")

    logger.info("\n--- Finished testing h0_mcmc_analyzer.py ---") 