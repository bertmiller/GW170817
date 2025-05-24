import numpy as np
import sys
# from scipy.stats import norm # Replaced by logpdf_normal_xp
from numpy.polynomial.hermite import hermgauss
# from scipy.special import logsumexp # Will be replaced by logsumexp_xp
from astropy.cosmology import FlatLambdaCDM # Removed
from astropy import units as u # Removed
import emcee
import logging
import cProfile
import pstats
import io
from gwsiren import CONFIG # Keep for default values, but not for backend selection in get_log_likelihood_h0
from gwsiren.backends import get_xp, logpdf_normal_xp, logsumexp_xp, trapz_xp 

logger = logging.getLogger(__name__)

# Module-level constants for memory management in NumPy backend path
MEMORY_THRESHOLD_BYTES = 4 * (1024**3)  # 4 GB
BYTES_PER_ELEMENT = 8  # for float64 (numpy.float64)

# Default Cosmological Parameters for Likelihood
DEFAULT_SIGMA_V_PEC = CONFIG.cosmology["sigma_v_pec"]  # km/s, peculiar velocity uncertainty
DEFAULT_C_LIGHT = CONFIG.cosmology["c_light"]
DEFAULT_OMEGA_M = CONFIG.cosmology["omega_m"]

# Default MCMC Parameters
DEFAULT_MCMC_N_DIM = 2
DEFAULT_MCMC_N_WALKERS = CONFIG.mcmc["walkers"]
DEFAULT_MCMC_N_STEPS = CONFIG.mcmc["steps"]
DEFAULT_MCMC_BURNIN = CONFIG.mcmc["burnin"]
DEFAULT_MCMC_THIN_BY = CONFIG.mcmc["thin_by"]
DEFAULT_MCMC_INITIAL_H0_MEAN = 70.0
DEFAULT_MCMC_INITIAL_H0_STD = 10.0
DEFAULT_H0_PRIOR_MIN = CONFIG.mcmc["prior_h0_min"]  # km/s/Mpc
DEFAULT_H0_PRIOR_MAX = CONFIG.mcmc["prior_h0_max"]  # km/s/Mpc
DEFAULT_ALPHA_PRIOR_MIN = CONFIG.mcmc.get("prior_alpha_min", -1.0)
DEFAULT_ALPHA_PRIOR_MAX = CONFIG.mcmc.get("prior_alpha_max", 1.0)
DEFAULT_MCMC_INITIAL_ALPHA_MEAN = 0.0
DEFAULT_MCMC_INITIAL_ALPHA_STD = 0.5

# Redshift marginalization parameters
DEFAULT_Z_ERR_THRESHOLD = 1e-6  # Much smaller threshold - only skip for essentially zero uncertainty
DEFAULT_QUAD_POINTS = 7  # More quadrature points for better accuracy
DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE = 4.0  # How many sigma to integrate over

class H0LogLikelihood:
    """Log-likelihood for joint inference of ``H0`` and ``alpha``.

    Args:
        dL_gw_samples: Array of GW luminosity distance samples.
        host_galaxies_z: Redshifts of candidate host galaxies.
        host_galaxies_mass_proxy: Positive mass proxy values for the galaxies.
        host_galaxies_z_err: Redshift uncertainties for the galaxies.
        sigma_v: Peculiar velocity dispersion in km/s.
        c_val: Speed of light in km/s.
        omega_m_val: Matter density parameter.
        h0_min: Lower prior bound on ``H0``.
        h0_max: Upper prior bound on ``H0``.
        alpha_min: Lower prior bound on ``alpha``.
        alpha_max: Upper prior bound on ``alpha``.
        use_vectorized_likelihood: Whether to use the vectorised likelihood
            implementation.
    """
    def __init__(
        self,
        xp,  # Added
        backend_name, # Added
        # device_name, # Optional to store, but good to receive
        dL_gw_samples,
        host_galaxies_z,
        host_galaxies_mass_proxy,
        host_galaxies_z_err,
        sigma_v=DEFAULT_SIGMA_V_PEC,
        c_val=DEFAULT_C_LIGHT,
        omega_m_val=DEFAULT_OMEGA_M,
        h0_min=DEFAULT_H0_PRIOR_MIN,
        h0_max=DEFAULT_H0_PRIOR_MAX,
        alpha_min=DEFAULT_ALPHA_PRIOR_MIN,
        alpha_max=DEFAULT_ALPHA_PRIOR_MAX,
        use_vectorized_likelihood=False, # Keep for now
        z_err_threshold=DEFAULT_Z_ERR_THRESHOLD,  # New parameter
        n_quad_points=DEFAULT_QUAD_POINTS,  # New parameter
        z_sigma_range=DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE,  # New parameter
    ):
        self.xp = xp
        self.backend_name = backend_name
        # self.device_name = device_name

        # Initial validation using NumPy as it's familiar and robust for this
        if dL_gw_samples is None or len(np.asarray(dL_gw_samples)) == 0: # Use np for initial check
            raise ValueError("dL_gw_samples cannot be None or empty.")
        if host_galaxies_z is None or len(np.asarray(host_galaxies_z)) == 0:
            raise ValueError("host_galaxies_z cannot be None or empty.")
        if host_galaxies_mass_proxy is None or len(np.asarray(host_galaxies_mass_proxy)) == 0:
            raise ValueError("host_galaxies_mass_proxy cannot be None or empty.")
        if host_galaxies_z_err is None or len(np.asarray(host_galaxies_z_err)) == 0:
            raise ValueError("host_galaxies_z_err cannot be None or empty.")

        # Convert to self.xp arrays after validation
        self.dL_gw_samples = self.xp.asarray(dL_gw_samples)
        _z_values_np = np.asarray(host_galaxies_z) # Use np for initial manipulation if needed
        _mass_proxy_values_np = np.asarray(host_galaxies_mass_proxy, dtype=float)
        _z_err_values_np = np.asarray(host_galaxies_z_err, dtype=float)

        if _z_values_np.ndim == 0: # if it was a single scalar
            _z_values_np = np.array([_z_values_np])
        if _mass_proxy_values_np.ndim == 0:
            _mass_proxy_values_np = np.array([_mass_proxy_values_np])
        if _z_err_values_np.ndim == 0:
            _z_err_values_np = np.array([_z_err_values_np])
        
        self.z_values = self.xp.asarray(_z_values_np)
        self.mass_proxy_values = self.xp.asarray(_mass_proxy_values_np)
        self.z_err_values = self.xp.asarray(_z_err_values_np)

        if len(self.mass_proxy_values) != len(self.z_values):
            raise ValueError("host_galaxies_mass_proxy and host_galaxies_z must have the same length")
        if len(self.z_err_values) != len(self.z_values):
            raise ValueError("host_galaxies_z_err and host_galaxies_z must have the same length")
        
        self.sigma_v = sigma_v
        self.c_val = c_val
        self.omega_m_val = omega_m_val
        self.h0_min = h0_min
        self.h0_max = h0_max
        self.use_vectorized_likelihood = use_vectorized_likelihood
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.z_err_threshold = z_err_threshold
        self.n_quad_points = n_quad_points
        self.z_sigma_range = z_sigma_range

        _quad_nodes_np, _quad_weights_np = hermgauss(self.n_quad_points) # Still use numpy for this
        self._quad_nodes = self.xp.asarray(_quad_nodes_np) # Then convert
        self._quad_weights = self.xp.asarray(_quad_weights_np) # Then convert

        # Conditionally JIT compile __call__ if using JAX
        if self.backend_name == "jax":
            try:
                import jax
                # JIT compile the __call__ method.
                # JAX will treat 'self' as a static argument by default when JITting a method.
                # This means that the attributes of 'self' that are JAX arrays will be traced as
                # dynamic inputs, while Python primitive attributes of 'self' might be treated as static.
                # This is generally the desired behavior here.
                # self._original_call = self.__call__ # Optional: keep a reference for debugging
                self.__call__ = jax.jit(self.__call__)
                logger.debug(f"H0LogLikelihood.__call__ method has been JIT-compiled for event processing with JAX.")
            except ImportError:
                # This case should ideally be caught by get_xp, but as a fallback:
                logger.error("JAX backend was selected, but JAX module could not be imported for JIT. Likelihood will not be JITted.")
            except Exception as e:
                logger.error(f"Error during JAX JIT compilation of __call__: {e}")
                # Proceed with non-JITted version for JAX if JIT fails
                logger.warning("Proceeding with non-JITted JAX likelihood due to JIT compilation error.")


    def _scalar_comoving_distance_integral(self, z_scalar, N_trapz=200):
        """Computes integral(dz / E(z)) for a single scalar z_scalar using backend-agnostic trapz."""
        if z_scalar < 1e-9: # Threshold for being effectively zero
            return 0.0
        
        # z_steps must be generated by self.xp for trapz_xp
        z_steps = self.xp.linspace(0.0, z_scalar, N_trapz) 
        one_plus_z_steps = 1.0 + z_steps
        omega_m_val_f = self.xp.asarray(self.omega_m_val, dtype=self.xp.float64) # Ensure float for JAX
        
        Ez_sq = omega_m_val_f * (one_plus_z_steps**3) + (1.0 - omega_m_val_f)
        # Ensure Ez_sq is positive before sqrt, for numerical stability.
        Ez = self.xp.sqrt(self.xp.maximum(1e-18, Ez_sq)) 
        
        # Avoid division by zero if any Ez element is zero
        integrand = 1.0 / self.xp.maximum(1e-18, Ez)
        
        # Use backend-agnostic trapz instead of self.xp.trapz
        integral = trapz_xp(self.xp, integrand, x=z_steps)
        return integral

    def _lum_dist_model(self, z_values, H0_val, N_trapz=200):
        """Compute luminosity distance for z_values (array or scalar) and H0_val."""
        is_scalar_input = (self.xp.asarray(z_values).ndim == 0)
        z_input_arr = self.xp.atleast_1d(z_values)

        if H0_val < 1e-9: # Avoid division by zero for H0
            return self.xp.full_like(z_input_arr, self.xp.inf, dtype=self.xp.float64)

        dc_integrals_list = [self._scalar_comoving_distance_integral(z_val, N_trapz) for z_val in z_input_arr]
        dc_integrals = self.xp.asarray(dc_integrals_list, dtype=self.xp.float64)

        lum_dist = (self.c_val / H0_val) * (1.0 + z_input_arr) * dc_integrals
        
        if is_scalar_input and lum_dist.ndim > 0: # Ensure output matches input shape
            return lum_dist[0]
        return lum_dist

    def __call__(self, theta):
        H0 = theta[0]
        # logger.debug(f"H0LogLikelihood.__call__ with H0 = {H0}, vectorized = {self.use_vectorized_likelihood}")

        if not (self.h0_min <= H0 <= self.h0_max):
            # logger.debug(f"H0 = {H0} is outside prior range ({self.h0_min}, {self.h0_max}). Returning -inf.")
            return -self.xp.inf

        try: 
            model_d_for_hosts = self._lum_dist_model(self.z_values, H0) # self.z_values is now xp array
            # logger.debug(f"model_d_for_hosts (first 5 if available): {model_d_for_hosts[:min(5, len(model_d_for_hosts))]}") 
            if self.xp.any(~self.xp.isfinite(model_d_for_hosts)): 
                # logger.debug(f"Non-finite values in model_d_for_hosts for H0 = {H0}. Example: {model_d_for_hosts[~self.xp.isfinite(model_d_for_hosts)][:min(5, len(model_d_for_hosts[~self.xp.isfinite(model_d_for_hosts)]))]}") 
                return -self.xp.inf 
        except Exception as e: 
            # logger.debug(f"EXCEPTION in _lum_dist_model for H0 = {H0}: {e}") 
            return -self.xp.inf 
        
        sigma_d_val_for_hosts = (model_d_for_hosts / self.c_val) * self.sigma_v
        sigma_d_val_for_hosts = self.xp.maximum(sigma_d_val_for_hosts, 1e-9) 
        # logger.debug(f"sigma_d_val_for_hosts (first 5 if available): {sigma_d_val_for_hosts[:min(5, len(sigma_d_val_for_hosts))]}") 
        if self.xp.any(~self.xp.isfinite(sigma_d_val_for_hosts)): 
            # logger.debug(f"Non-finite values in sigma_d_val_for_hosts for H0 = {H0}.") 
            return -self.xp.inf

        log_sum_over_gw_samples = self.xp.array([-self.xp.inf]) # Placeholder for now

        if self.use_vectorized_likelihood:
            # --- Fully Vectorized Alternative ---
            # logger.debug("Using fully vectorized likelihood path.")
            try:
                # Reshape for broadcasting: 
                # self.dL_gw_samples: (N_samples,) -> (N_samples, 1)
                # model_d_for_hosts: (N_hosts,) -> (1, N_hosts)
                # sigma_d_val_for_hosts: (N_hosts,) -> (1, N_hosts)
                log_pdf_values_full = logpdf_normal_xp(
                    self.xp,
                    self.dL_gw_samples[:, self.xp.newaxis],
                    loc=model_d_for_hosts[self.xp.newaxis, :],
                    scale=sigma_d_val_for_hosts[self.xp.newaxis, :]
                ) # Shape: (N_samples, N_hosts)
            except Exception as e:
                # logger.debug(f"EXCEPTION in logpdf_normal_xp (vectorized) for H0 = {H0}: {e}")
                # If the entire logpdf calculation fails, it likely means a broad issue
                # (e.g. H0 way off, leading to bad model_d_for_hosts for ALL hosts)
                # In this case, returning -self.xp.inf for the whole likelihood is appropriate.
                return -self.xp.inf

            # Handle potential NaNs by converting them to -self.xp.inf.
            # This allows logaddexp.reduce to correctly ignore them unless an entire column is -self.xp.inf.
            log_pdf_values_full = self.xp.where(self.xp.isnan(log_pdf_values_full), -self.xp.inf, log_pdf_values_full)


            # Check if any galaxy (column) resulted in all -self.xp.inf values for its log_pdf terms.
            # This can happen if a specific galaxy's model_d or sigma_d was problematic.
            # all_inf_columns = self.xp.all(log_pdf_values_full == -self.xp.inf, axis=0)

            # Use the new logsumexp_xp helper function
            log_sum_over_gw_samples = logsumexp_xp(self.xp, log_pdf_values_full, axis=0) - self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
            
            # After reduction, if any element in log_sum_over_gw_samples is -self.xp.inf 
            # (e.g., because its corresponding column in log_pdf_values_full was all -self.xp.inf, 
            # or because logaddexp.reduce itself resulted in -self.xp.inf due to underflow with very small numbers),
            # it will correctly propagate.
            # No specific check for all_inf_columns is strictly needed here because logaddexp.reduce handles it.
            
            # logger.debug(f"log_sum_over_gw_samples (vectorized) shape: {log_sum_over_gw_samples.shape}, any non-finite: {self.xp.any(~self.xp.isfinite(log_sum_over_gw_samples))}")
            # Example: Check first 5 values if problematic
            if self.xp.any(~self.xp.isfinite(log_sum_over_gw_samples)):
                # logger.debug(f"  Non-finite log_sum_over_gw_samples (vectorized): {log_sum_over_gw_samples[~self.xp.isfinite(log_sum_over_gw_samples)][:5]}")
                return -self.xp.inf

        else:
            # --- Memory-Efficient Loop with Redshift Marginalization ---
            log_P_data_H0_zi_terms = self.xp.zeros(len(self.z_values))
            for i in range(len(self.z_values)):
                mu_z = self.z_values[i]
                sigma_z = self.z_err_values[i]

                # Use the new, much smaller threshold for skipping marginalization
                if sigma_z < self.z_err_threshold:
                    # Only skip marginalization for essentially zero uncertainty
                    current_model_d = model_d_for_hosts[i]
                    current_sigma_d = sigma_d_val_for_hosts[i]
                    try:
                        log_pdf_for_one_galaxy = logpdf_normal_xp(
                            self.xp,
                            self.dL_gw_samples,
                            loc=current_model_d,
                            scale=current_sigma_d,
                        )
                    except Exception:
                        log_pdf_for_one_galaxy = self.xp.full(len(self.dL_gw_samples), -self.xp.inf)

                    if self.xp.any(~self.xp.isfinite(log_pdf_for_one_galaxy)):
                        log_P_data_H0_zi_terms = self._update_array_element(log_P_data_H0_zi_terms, i, -self.xp.inf)
                    else:
                        reduced_log_pdf = logsumexp_xp(self.xp, log_pdf_for_one_galaxy)
                        term_val = reduced_log_pdf - self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
                        log_P_data_H0_zi_terms = self._update_array_element(log_P_data_H0_zi_terms, i, term_val)
                    continue

                # Perform proper redshift marginalization
                # Integrate over a wider range for better accuracy
                z_min = self.xp.maximum(mu_z - self.z_sigma_range * sigma_z, 1e-6)  # Ensure positive
                z_max = mu_z + self.z_sigma_range * sigma_z
                
                # Check if the integration range is meaningful
                if z_max <= z_min:
                    # Fallback to point estimate if range is degenerate
                    current_model_d = model_d_for_hosts[i]
                    current_sigma_d = sigma_d_val_for_hosts[i]
                    try:
                        log_pdf_for_one_galaxy = logpdf_normal_xp(
                            self.xp,
                            self.dL_gw_samples,
                            loc=current_model_d,
                            scale=current_sigma_d,
                        )
                        reduced_log_pdf = logsumexp_xp(self.xp, log_pdf_for_one_galaxy)
                        term_val = reduced_log_pdf - self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
                        log_P_data_H0_zi_terms = self._update_array_element(log_P_data_H0_zi_terms, i, term_val)
                    except Exception:
                        log_P_data_H0_zi_terms = self._update_array_element(log_P_data_H0_zi_terms, i, -self.xp.inf)
                    continue

                # Gaussian quadrature integration
                log_integrand_values = []
                valid_evaluations = 0
                
                for node, weight in zip(self._quad_nodes, self._quad_weights):
                    # Transform quadrature point to redshift domain
                    z_j = mu_z + self.xp.sqrt(2.0) * sigma_z * node
                    
                    # Skip negative redshifts
                    if z_j <= 0:
                        continue
                        
                    try:
                        # Compute luminosity distance for this redshift
                        model_d = self._lum_dist_model(z_j, H0)
                        
                        # Check if model distance is reasonable
                        if not self.xp.isfinite(model_d) or model_d <= 0:
                            continue
                            
                        sigma_d = self.xp.maximum((model_d / self.c_val) * self.sigma_v, 1e-9)
                        
                        # Compute likelihood for this redshift
                        log_pdf = logpdf_normal_xp(
                            self.xp,
                            self.dL_gw_samples,
                            loc=model_d,
                            scale=sigma_d,
                        )
                        
                        if self.xp.any(~self.xp.isfinite(log_pdf)):
                            continue
                        
                        # Sum over GW samples
                        current_logsumexp = logsumexp_xp(self.xp, log_pdf)
                        log_likelihood_at_z = current_logsumexp - self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
                        
                        # Store log(weight * likelihood) for numerical stability
                        log_weight = self.xp.log(weight / self.xp.sqrt(self.xp.pi))
                        log_integrand_values.append(log_weight + log_likelihood_at_z)
                        valid_evaluations += 1
                        
                    except Exception:
                        # Skip this quadrature point if evaluation fails
                        continue

                # Combine the integration results
                if valid_evaluations > 0 and len(log_integrand_values) > 0:
                    # Use logsumexp for numerical stability
                    log_integrand_array = self.xp.array(log_integrand_values)
                    log_integrated_prob = logsumexp_xp(self.xp, log_integrand_array)
                    log_P_data_H0_zi_terms = self._update_array_element(log_P_data_H0_zi_terms, i, log_integrated_prob)
                else:
                    # Integration failed - set to -inf
                    log_P_data_H0_zi_terms = self._update_array_element(log_P_data_H0_zi_terms, i, -self.xp.inf)
            
            log_sum_over_gw_samples = log_P_data_H0_zi_terms

        alpha = theta[1]
        if not (self.alpha_min <= alpha <= self.alpha_max):
            return -self.xp.inf

        if self.xp.isclose(alpha, 0.0):
            weights = self.xp.full(len(self.mass_proxy_values), 1.0 / len(self.mass_proxy_values))
        else:
            powered = self.mass_proxy_values ** alpha
            if self.xp.any(powered <= 0) or not self.xp.all(self.xp.isfinite(powered)):
                return -self.xp.inf
            denom = powered.sum()
            if not self.xp.isfinite(denom) or denom <= 0:
                return -self.xp.inf
            weights = powered / denom

        valid_mask = weights > 0
        if not self.xp.any(valid_mask):
            return -self.xp.inf
        
        terms_to_sum = self.xp.log(weights[valid_mask]) + log_sum_over_gw_samples[valid_mask]
        total_log_likelihood = logsumexp_xp(self.xp, terms_to_sum)

        if not self.xp.isfinite(total_log_likelihood):
            return -self.xp.inf

        return total_log_likelihood

    def _update_array_element(self, array, index, value):
        """Backend-agnostic array element update."""
        if self.backend_name == "jax":
            # JAX uses immutable arrays, so we use .at[].set()
            return array.at[index].set(value)
        else:
            # NumPy uses mutable arrays
            array[index] = value
            return array

def get_log_likelihood_h0(
    requested_backend_str: str, # Added: explicit backend request
    dL_gw_samples,
    host_galaxies_z,
    host_galaxies_mass_proxy,
    host_galaxies_z_err,
    sigma_v=DEFAULT_SIGMA_V_PEC,
    c_val=DEFAULT_C_LIGHT,
    omega_m_val=DEFAULT_OMEGA_M,
    h0_min=DEFAULT_H0_PRIOR_MIN,
    h0_max=DEFAULT_H0_PRIOR_MAX,
    alpha_min=DEFAULT_ALPHA_PRIOR_MIN,
    alpha_max=DEFAULT_ALPHA_PRIOR_MAX,
    z_err_threshold=DEFAULT_Z_ERR_THRESHOLD,  # New parameter
    n_quad_points=DEFAULT_QUAD_POINTS,  # New parameter
    z_sigma_range=DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE,  # New parameter
    force_non_vectorized=False,  # New parameter to override vectorization decision
    # use_vectorized_likelihood is determined below based on backend_name
):
    """
    Returns an instance of the H0LogLikelihood class, configured for the specified backend.
    
    Args:
        requested_backend_str (str): The backend to use ("auto", "numpy", "jax").
        dL_gw_samples (np.array): Luminosity distance samples...
        host_galaxies_z (np.array): Redshifts of candidate host galaxies (N_hosts,).
        host_galaxies_mass_proxy (np.array): Positive mass proxy values for the galaxies.
        host_galaxies_z_err (np.array): Redshift uncertainties for the galaxies.
        sigma_v (float): Peculiar velocity uncertainty (km/s).
        c_val (float): Speed of light (km/s).
        omega_m_val (float): Omega Matter for cosmological calculations.
        h0_min (float): Minimum value for H0 prior.
        h0_max (float): Maximum value for H0 prior.
        alpha_min (float): Minimum value for alpha prior.
        alpha_max (float): Maximum value for alpha prior.

    Returns:
        H0LogLikelihood: An instance of the H0LogLikelihood class.
    """
    # At the beginning of the function:
    xp_module, backend_name, device_name = get_xp(requested_backend_str) # Use the passed string

    # Ensure dL_gw_samples is array-like for len() before passing to H0LogLikelihood
    # which will convert it to xp_module.asarray()
    # H0LogLikelihood's __init__ does initial checks with np.asarray for robustness.
    n_gw_samples = len(np.asarray(dL_gw_samples))
    
    # Ensure host_galaxies_z is treated as an array for len()
    _host_galaxies_z_np_for_len = np.asarray(host_galaxies_z)
    if _host_galaxies_z_np_for_len.ndim == 0:
        n_hosts = 1
    else:
        n_hosts = len(_host_galaxies_z_np_for_len)

    if backend_name == "jax":
        # JAX can also be forced to use non-vectorized path for testing/debugging  
        if force_non_vectorized:
            should_use_vectorized = False
            logger.info(f"JAX backend FORCED to non-vectorized for testing/debugging purposes.")
        else:
            should_use_vectorized = True # JAX path will be inherently vectorized/optimized
            logger.info(f"Using JAX backend (on {device_name}). Will use JAX-optimized likelihood path.")
    else: # numpy backend
        # Existing memory-based decision for NumPy
        # n_gw_samples, n_hosts are calculated above this block in the provided snippet
        current_elements = n_gw_samples * n_hosts
        # MEMORY_THRESHOLD_BYTES and BYTES_PER_ELEMENT are now module-level constants
        max_elements_for_vectorization = MEMORY_THRESHOLD_BYTES / BYTES_PER_ELEMENT
        should_use_vectorized = current_elements <= max_elements_for_vectorization
        
        # Override if explicitly requested
        if force_non_vectorized:
            should_use_vectorized = False
            logger.info(f"FORCED non-vectorized likelihood for testing/debugging purposes.")
        
        if should_use_vectorized:
            logger.info(
                f"Using NumPy backend, VECTORIZED likelihood: N_samples ({n_gw_samples}) * N_hosts ({n_hosts}) = {current_elements} elements. "
                f"Threshold: {max_elements_for_vectorization:.0f} elements ({MEMORY_THRESHOLD_BYTES / (1024**3):.0f} GB)."
            )
        else:
            logger.info(
                f"Using NumPy backend, LOOPED likelihood (memory efficient): N_samples ({n_gw_samples}) * N_hosts ({n_hosts}) = {current_elements} elements. "
                f"{'Forced by force_non_vectorized=True' if force_non_vectorized else f'Exceeds threshold of {max_elements_for_vectorization:.0f} elements ({MEMORY_THRESHOLD_BYTES / (1024**3):.0f} GB).'}."
            )

    return H0LogLikelihood(
        xp_module, # Pass xp
        backend_name, # Pass backend_name
        # device_name, # Pass device_name (optional to store in H0LogLikelihood)
        dL_gw_samples,
        host_galaxies_z,
        host_galaxies_mass_proxy,
        host_galaxies_z_err,
        sigma_v,
        c_val,
        omega_m_val,
        h0_min,
        h0_max,
        alpha_min,
        alpha_max,
        use_vectorized_likelihood=should_use_vectorized,
        z_err_threshold=z_err_threshold,
        n_quad_points=n_quad_points,
        z_sigma_range=z_sigma_range,
    )

# It might be useful to keep get_log_likelihood_h0_vectorized for specific testing,
# but ensure it also gets the xp context. Or remove if get_log_likelihood_h0 covers all.
# For now, let's assume it's removed or refactored if kept.
# For this subtask, I will remove it as get_log_likelihood_h0 now handles backend.
# def get_log_likelihood_h0_vectorized(...): -> This function is removed.

def run_mcmc_h0(
    log_likelihood_func,
    event_name,  # For logging
    n_walkers=DEFAULT_MCMC_N_WALKERS,
    n_dim=DEFAULT_MCMC_N_DIM,
    initial_h0_mean=DEFAULT_MCMC_INITIAL_H0_MEAN,
    initial_h0_std=DEFAULT_MCMC_INITIAL_H0_STD,
    alpha_prior_min=DEFAULT_ALPHA_PRIOR_MIN,
    alpha_prior_max=DEFAULT_ALPHA_PRIOR_MAX,
    n_steps=DEFAULT_MCMC_N_STEPS,
    pool=None,
):
    """
    Runs the MCMC sampler for H0.

    Args:
        log_likelihood_func (function): The log likelihood function `log_likelihood(theta)`.
        event_name (str): Name of the event for logging.
        n_walkers (int): Number of MCMC walkers.
        n_dim (int): Number of dimensions (2 for ``H0`` and ``alpha``).
        initial_h0_mean (float): Mean for initializing ``H0`` walker positions.
        initial_h0_std (float): Standard deviation for initializing ``H0`` positions.
        alpha_prior_min (float): Lower bound for ``alpha`` initialization.
        alpha_prior_max (float): Upper bound for ``alpha`` initialization.
        n_steps (int): Number of MCMC steps.
        pool (object, optional): A pool object for parallelization (e.g., from multiprocessing or emcee.interruptible_pool).

    Returns:
        emcee.EnsembleSampler: The MCMC sampler object after running, or None on failure.
    """
    logger.info(
        f"Running MCMC for H0 and alpha on event {event_name} ({n_steps} steps, {n_walkers} walkers)..."
    )
    pos_H0 = initial_h0_mean + initial_h0_std * np.random.randn(n_walkers, 1)
    pos_alpha = np.random.uniform(alpha_prior_min, alpha_prior_max, size=(n_walkers, 1))
    walkers0 = np.hstack((pos_H0, pos_alpha))
        
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
    # Determine backend for testing - e.g. use "auto" or a specific one like "numpy"
    test_backend_choice = "auto" # Or "numpy", "jax"
    # If using "auto", it might pick JAX if available. For more controlled tests, specify "numpy" or "jax".
    # test_backend_choice = CONFIG.computation.backend # Or read from config for test consistency
    
    try:
        # Ensure mock data for mass_proxy and z_err are provided
        mock_mass_proxy = np.random.rand(len(mock_host_zs)) * 100 + 1 # Example positive values
        mock_z_err = np.random.rand(len(mock_host_zs)) * 0.01 # Example small errors

        log_like_func = get_log_likelihood_h0(
            test_backend_choice, # Pass the chosen backend string for testing
            mock_dL_gw, 
            mock_host_zs,
            mock_mass_proxy,
            mock_z_err
        )
        # Test the likelihood function with some H0 values
        h0_test_values = [60.0, 70.0, 80.0]
        logger.info(f"  Log likelihood values for H0={h0_test_values} using backend '{log_like_func.backend_name}':")
        for h0_val in h0_test_values:
            # Assuming alpha is the second parameter, provide a default like 0.0
            ll = log_like_func([h0_val, 0.0]) 
            logger.info(f"    H0 = {h0_val:.1f}, alpha = 0.0: logL = {ll:.2f}")
            # Using xp.isfinite from the likelihood's own backend context
            assert log_like_func.xp.isfinite(ll), f"LogL not finite for H0={h0_val}"
        logger.info("  Log likelihood function seems operational.")

        # Test prior boundaries
        ll_low = log_like_func([DEFAULT_H0_PRIOR_MIN - 1, 0.0])
        assert ll_low == -log_like_func.xp.inf, "Prior min boundary failed"
        ll_high = log_like_func([DEFAULT_H0_PRIOR_MAX + 1, 0.0])
        assert ll_high == -log_like_func.xp.inf, "Prior max boundary failed"
        logger.info(f"  Prior boundaries (H0_min={DEFAULT_H0_PRIOR_MIN}, H0_max={DEFAULT_H0_PRIOR_MAX}) correctly applied.")

        ll_dynamic = log_like_func([70.0, 0.0]) # Pass alpha too
        logger.info(f"  Dynamically chosen path (backend: {log_like_func.backend_name if hasattr(log_like_func, 'backend_name') else 'N/A'}) gives logL={ll_dynamic:.2f}")


    except ValueError as ve:
        logger.error(f"  Error in Test 1 (get_log_likelihood_h0): {ve}", exc_info=True)
    except Exception as e:
        logger.exception(f"  Unexpected error in Test 1 (get_log_likelihood_h0): {e}", exc_info=True)

    # 2. Test run_mcmc_h0 and process_mcmc_samples
    logger.info("\nTest 2: Running MCMC and processing samples...")
    # Reduce steps for faster testing
    test_mcmc_steps = 200
    test_mcmc_burnin = 50
    test_mcmc_walkers = 8 # Fewer walkers for test

    # Get the backend once for the test script main section, if still needed outside get_log_likelihood_h0
    # xp_test, backend_name_test, device_name_test = get_xp(test_backend_choice) 
    # logger.info(f"--- Main test script using backend: {backend_name_test} on {device_name_test} for general tests ---")


    if 'log_like_func' in locals(): # Proceed if likelihood function was created
        try: # Add try block here
            # log_like_func is already obtained using test_backend_choice
            # try:
                # log_like_func = get_log_likelihood_h0( # This would re-create it, not needed if already created above
                #     test_backend_choice,
                #     mock_dL_gw, 
                #     mock_host_zs, 
                #     mock_mass_proxy, 
                #     mock_z_err
                # )
                # logger.info(f"  Log likelihood function re-obtained with backend: {log_like_func.backend_name if hasattr(log_like_func, 'backend_name') else 'N/A'}")

            sampler = run_mcmc_h0(
                    log_like_func,
                    mock_event,
                    n_walkers=test_mcmc_walkers,
                    n_steps=test_mcmc_steps
                )

            if sampler:
                    # Process MCMC samples (this function remains largely numpy-based for now from emcee output)
                    h0_samples_processed = process_mcmc_samples(
                        sampler,
                        mock_event,
                        burnin=test_mcmc_burnin,
                        thin_by=2 # Small thin factor for test
                    )
                    if h0_samples_processed is not None and len(h0_samples_processed) > 0:
                        logger.info(f"  Successfully ran MCMC and processed samples for {mock_event}.")
                        logger.info(f"  Number of H0 samples obtained: {len(h0_samples_processed)}")
                        # np.mean and np.std are fine here as h0_samples_processed is a NumPy array from emcee
                        logger.info(f"  H0 mean: {np.mean(h0_samples_processed[:,0]):.2f}, H0 std: {np.std(h0_samples_processed[:,0]):.2f}")
                        if h0_samples_processed.shape[1] > 1:
                             logger.info(f"  Alpha mean: {np.mean(h0_samples_processed[:,1]):.2f}, Alpha std: {np.std(h0_samples_processed[:,1]):.2f}")

                    elif h0_samples_processed is not None and len(h0_samples_processed) == 0:
                        logger.warning("  MCMC processing resulted in zero samples. Check burn-in/thinning or chain length.")
                    else:
                        logger.error("  MCMC processing failed to return valid samples.")
            else:
                    logger.error("  MCMC run failed or returned no sampler. Cannot test processing.")
        except Exception as e_mcmc_test: # This except now correctly pairs with the try above
            logger.exception(f"  Error during MCMC test setup or execution with new backend logic: {e_mcmc_test}")

    else:
        logger.error("  Log likelihood function (log_like_func) not available. Skipping MCMC run and processing tests.")


    # Test with problematic inputs for get_log_likelihood_h0
    logger.info("\nTest 3: Edge cases for get_log_likelihood_h0")
    try:
        get_log_likelihood_h0(None, mock_host_zs, np.ones_like(mock_host_zs), np.full_like(mock_host_zs, 0.001))
    except ValueError as e:
        logger.info(f"  Correctly caught error for None dL samples: {e}")
    try:
        get_log_likelihood_h0(mock_dL_gw, [], [], [])
    except ValueError as e:
        logger.info(f"  Correctly caught error for empty host_zs: {e}")

    # 4. Profiling Section - This will need careful adaptation for xp
    logger.info("\nTest 4: Profiling the __call__ method...")
    if 'log_like_func' in locals() and hasattr(log_like_func, 'xp'):
        
        # To profile both paths (numpy-looped, numpy-vectorized, jax), we need to ensure
        # we can create instances of H0LogLikelihood with specific backends and vectorization modes.
        
        # Numpy Looped
        xp_numpy, _, _ = get_xp("numpy")
        log_like_numpy_looped = H0LogLikelihood(
            xp_numpy, "numpy", mock_dL_gw, mock_host_zs, np.ones_like(mock_host_zs), np.full_like(mock_host_zs, 0.001),
            use_vectorized_likelihood=False
        )
        logger.info("Profiling NumPy Looped Likelihood...")
        pr_np_loop = cProfile.Profile()
        pr_np_loop.enable()
        profile_likelihood_call(log_like_numpy_looped, h0_test_values_for_profiling, n_calls=5)
        pr_np_loop.disable()
        s_np_loop = io.StringIO()
        pstats.Stats(pr_np_loop, stream=s_np_loop).sort_stats('cumulative').print_stats(30)
        logger.info("\n--- cProfile results for NumPy Looped Likelihood ---")
        print(s_np_loop.getvalue())

        # Numpy Vectorized
        log_like_numpy_vec = H0LogLikelihood(
            xp_numpy, "numpy", mock_dL_gw, mock_host_zs, np.ones_like(mock_host_zs), np.full_like(mock_host_zs, 0.001),
            use_vectorized_likelihood=True # This should be chosen if data size allows by get_log_likelihood_h0
        )
        logger.info("Profiling NumPy Vectorized Likelihood...")
        pr_np_vec = cProfile.Profile()
        pr_np_vec.enable()
        profile_likelihood_call(log_like_numpy_vec, h0_test_values_for_profiling, n_calls=5)
        pr_np_vec.disable()
        s_np_vec = io.StringIO()
        pstats.Stats(pr_np_vec, stream=s_np_vec).sort_stats('cumulative').print_stats(30)
        logger.info("\n--- cProfile results for NumPy Vectorized Likelihood ---")
        print(s_np_vec.getvalue())

        # JAX (if available)
        try:
            xp_jax, _, _ = get_xp("jax")
            # For JAX, use_vectorized_likelihood is effectively True
            log_like_jax = H0LogLikelihood(
                xp_jax, "jax", mock_dL_gw, mock_host_zs, np.ones_like(mock_host_zs), np.full_like(mock_host_zs, 0.001),
                use_vectorized_likelihood=True 
            )
            logger.info("Profiling JAX Likelihood...")
            # JAX specific: compile call for fair comparison if needed
            # _ = log_like_jax([h0_test_values_for_profiling[0]]).block_until_ready() # JIT compile
            
            pr_jax = cProfile.Profile()
            pr_jax.enable()
            profile_likelihood_call(log_like_jax, h0_test_values_for_profiling, n_calls=5)
            # for h0_val in h0_test_values_for_profiling: # Manual loop for block_until_ready
            #    for _ in range(5):
            #        _ = log_like_jax([h0_val]).block_until_ready()

            pr_jax.disable()
            s_jax = io.StringIO()
            pstats.Stats(pr_jax, stream=s_jax).sort_stats('cumulative').print_stats(30)
            logger.info("\n--- cProfile results for JAX Likelihood ---")
            print(s_jax.getvalue())
        except Exception as e_jax_profile:
            logger.warning(f"Could not profile JAX backend (likely not available or error during setup): {e_jax_profile}")
            
    else:
        logger.error("  Log likelihood function not available for profiling with xp context.")

    # 5. Timeit Benchmarking Section - Adapt similarly to profiling
    logger.info("\nTest 5: Benchmarking with timeit...")
    if 'log_like_numpy_looped' in locals() and 'log_like_numpy_vec' in locals(): # Check if numpy versions were created
        import timeit
        
        n_timeit_runs = 100
        n_timeit_repeat = 5
        timeit_globals_np_loop = {"func": log_like_numpy_looped, "val": [70.0, 0.0]}
        timeit_globals_np_vec = {"func": log_like_numpy_vec, "val": [70.0, 0.0]}

        logger.info(f"  Timing NumPy looped version ({n_timeit_runs} calls, {n_timeit_repeat} repeats)...")
        looped_times = timeit.repeat("func(val)", globals=timeit_globals_np_loop, number=n_timeit_runs, repeat=n_timeit_repeat)
        min_looped_time = min(looped_times) / n_timeit_runs
        logger.info(f"    Min time per call (NumPy looped): {min_looped_time*1e6:.2f} microseconds")

        logger.info(f"  Timing NumPy vectorized version ({n_timeit_runs} calls, {n_timeit_repeat} repeats)...")
        vectorized_times = timeit.repeat("func(val)", globals=timeit_globals_np_vec, number=n_timeit_runs, repeat=n_timeit_repeat)
        min_vectorized_time = min(vectorized_times) / n_timeit_runs
        logger.info(f"    Min time per call (NumPy vectorized): {min_vectorized_time*1e6:.2f} microseconds")
        
        if min_vectorized_time < min_looped_time:
            logger.info(f"    NumPy Vectorized is approx {min_looped_time/min_vectorized_time:.2f}x faster.")
        else:
            logger.info(f"    NumPy Looped is approx {min_vectorized_time/min_looped_time:.2f}x faster (or similar speed).")

        if 'log_like_jax' in locals():
            # JAX timeit needs careful handling of compilation
            # It's often better to benchmark JAX by calling a jitted function multiple times
            # and using block_until_ready(). timeit might not give the full picture due to JIT overhead on first calls.
            # For a simple comparison here:
            from functools import partial
            
            @partial(xp_jax.jit, static_argnums=(0,))
            def jax_call_for_timeit(func_obj, val_jax):
                return func_obj(val_jax)

            # Warm-up / JIT compilation
            # test_val_jax = xp_jax.array([70.0, 0.0])
            # _ = jax_call_for_timeit(log_like_jax, test_val_jax).block_until_ready()
            
            # timeit_globals_jax = {"func_jitted": partial(jax_call_for_timeit, log_like_jax) , "val_j": test_val_jax}
            # logger.info(f"  Timing JAX version ({n_timeit_runs} calls, {n_timeit_repeat} repeats)...")
            # jax_times = timeit.repeat("func_jitted(val_j).block_until_ready()", globals=timeit_globals_jax, number=n_timeit_runs, repeat=n_timeit_repeat)
            # min_jax_time = min(jax_times) / n_timeit_runs
            # logger.info(f"    Min time per call (JAX): {min_jax_time*1e6:.2f} microseconds")
            # if min_jax_time < min_vectorized_time:
            #    logger.info(f"    JAX is approx {min_vectorized_time/min_jax_time:.2f}x faster than NumPy vectorized.")
            # else:
            #    logger.info(f"    NumPy Vectorized is approx {min_jax_time/min_vectorized_time:.2f}x faster than JAX (or similar speed).")
            logger.info("    (JAX timeit requires careful setup with JIT compilation and block_until_ready, simplified here)")


    else:
        logger.error("  Log likelihood functions (NumPy versions) not available for timeit benchmarking.")

    # 6. Profiling Full MCMC Run - This would also need adaptation
    logger.info("\nTest 6: Profiling full MCMC runs (SKIPPING for this refactoring stage due to complexity of adapting MCMC part)...")
    # The MCMC part uses emcee, which expects numpy arrays for initial positions.
    # The log_likelihood_func passed to emcee would now be xp-aware.
    # If JAX is used, emcee might run slower if not integrated carefully (e.g., passing JAX arrays directly).
    # This requires more thought on how emcee interacts with JAX functions (e.g. `jax.experimental.host_callback`).
    # For now, the main goal is to make the likelihood function itself backend-agnostic.

    logger.info("\n--- Finished testing h0_mcmc_analyzer.py (with backend adaptations) ---")