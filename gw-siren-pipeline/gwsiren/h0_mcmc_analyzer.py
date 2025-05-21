import numpy as np
import sys
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp as scipy_logsumexp # Renamed for clarity
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from gwsiren.backends import log_gaussian, get_xp # Added get_xp
# Conditional JAX import for type hinting and eventual JIT
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
import emcee
import logging
import cProfile
import pstats
import io
from gwsiren import CONFIG

logger = logging.getLogger(__name__)

# Placeholder for the dL_ વો constant mentioned in the prompt for sigma_d_val_for_hosts
# Assuming it's a relative error component on dL. Let's define it as a constant for now.
# If this should be configurable, it would need to be passed in or accessed from CONFIG.
DL_REL_ERROR_FOR_SIGMA_D = 0.05 # This is the "dL_ વો" or similar from the prompt

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


# --- Pure function for JIT compilation ---
def _calculate_vectorized_log_likelihood_core(
    theta, 
    dL_gw_samples, 
    mass_proxy_values, 
    _base_dl_for_hosts, 
    sigma_v_pec, 
    c_light, 
    h0_min, 
    h0_max, 
    alpha_min, 
    alpha_max, 
    xp, # numpy or jax.numpy
    log_gaussian_func, 
    num_gw_samples,
    dl_rel_error_for_sigma_d # Added based on formula interpretation
    ):
    """
    Core calculation of the vectorized log likelihood.
    Designed to be JIT-compatible with JAX.
    All inputs must be passed explicitly.
    """
    H0, alpha_g = theta[0], theta[1] # Assuming theta is [H0, alpha_g]

    # Prior Checks
    if not (h0_min <= H0 <= h0_max):
        return -xp.inf 
    if not (alpha_min <= alpha_g <= alpha_max):
        return -xp.inf

    # Cosmological calculations
    # Ensure H0 is not zero or negative before division
    if H0 <= 0:
        return -xp.inf
    model_d_for_hosts = _base_dl_for_hosts / H0
    
    # sigma_d_val_for_hosts calculation based on the formula:
    # xp.sqrt(xp.square(model_d_for_hosts * sigma_v_pec / c_light) + xp.square(model_d_for_hosts * 0.05))
    # The 0.05 term is an assumption for dL_ વો / dL error term.
    # It should be model_d_for_hosts * dL_ વો, where dL_ વો is the relative uncertainty.
    # Let's assume dL_ વો is a constant relative error for now.
    term1_sq = xp.square(model_d_for_hosts * sigma_v_pec / c_light)
    term2_sq = xp.square(model_d_for_hosts * dl_rel_error_for_sigma_d) 
    sigma_d_val_for_hosts = xp.sqrt(term1_sq + term2_sq)
    # Ensure sigma_d is not zero or negative
    sigma_d_val_for_hosts = xp.maximum(sigma_d_val_for_hosts, 1e-9) # Floor value

    # Reshape for Broadcasting:
    # dL_gw_samples: (N_gw_samples,) -> (N_gw_samples, 1) for log_gaussian
    # model_d_for_hosts: (N_hosts,) -> (1, N_hosts) for log_gaussian
    # sigma_d_val_for_hosts: (N_hosts,) -> (1, N_hosts) for log_gaussian
    # log_pdf_matrix will be (N_gw_samples, N_hosts)
    #
    # The prompt had:
    # dL_gw_reshaped = xp.reshape(dL_gw_samples, (1, num_gw_samples)) -> This seems transposed for (sample, host) matrix
    # mass_proxy_reshaped = xp.reshape(mass_proxy_values, (1, num_gw_samples)) -> This is likely per host, not per GW sample
    # model_d_for_hosts_reshaped = xp.reshape(model_d_for_hosts, (-1, 1))
    # sigma_d_val_for_hosts_reshaped = xp.reshape(sigma_d_val_for_hosts, (-1, 1))
    #
    # Corrected reshaping for (dL_gw_samples_i, model_d_host_j)
    # dL_gw_samples should be (num_gw_samples, 1)
    # model_d_for_hosts should be (1, num_hosts)
    # This makes log_pdf_matrix shape (num_gw_samples, num_hosts)
    
    dL_gw_reshaped = xp.reshape(dL_gw_samples, (num_gw_samples, 1))
    model_d_for_hosts_reshaped = xp.reshape(model_d_for_hosts, (1, -1)) # -1 infers N_hosts
    sigma_d_val_for_hosts_reshaped = xp.reshape(sigma_d_val_for_hosts, (1, -1)) # -1 infers N_hosts

    log_pdf_matrix = log_gaussian_func(
        xp, 
        dL_gw_reshaped, # x values
        model_d_for_hosts_reshaped, # mu values
        sigma_d_val_for_hosts_reshaped # sigma values
    ) # Shape: (num_gw_samples, num_hosts)

    # Averaging likelihood over GW samples (logsumexp then subtract log(N))
    # This is P(dL_gw | H0, z_host_j) = (1/N_gw_samples) * sum_i P(dL_gw_i | H0, z_host_j)
    # log P(dL_gw | H0, z_host_j) = logsumexp_i ( log P(dL_gw_i | H0, z_host_j) ) - log(N_gw_samples)
    
    # Determine which logsumexp to use
    if hasattr(xp, 'scipy') and hasattr(xp.scipy, 'special') and hasattr(xp.scipy.special, 'logsumexp'):
        logsumexp_to_use = xp.scipy.special.logsumexp # For JAX
    else:
        logsumexp_to_use = scipy_logsumexp # For NumPy (imported as scipy_logsumexp)

    # log_likelihood_per_host should be of shape (num_hosts,)
    log_likelihood_per_host = logsumexp_to_use(log_pdf_matrix, axis=0) - xp.log(xp.array(num_gw_samples, dtype=xp.float64))

    # Weighting by mass proxy
    # P(H0, alpha|D) = sum_j P(H0|D,G_j) * P(G_j|alpha)
    # log P(H0,alpha|D) = logsumexp_j (log P(H0|D,G_j) + log P(G_j|alpha))
    # Assuming mass_proxy_values are M_j^alpha / sum_k M_k^alpha (already normalized or direct weights)
    # The prompt says "mass_proxy_reshaped = xp.reshape(mass_proxy_values, (1, num_gw_samples))"
    # this is incorrect, mass_proxy_values are per host.
    # And "weighted_log_pdf = log_pdf_matrix + mass_proxy_reshaped"
    # This implies mass_proxy_values are already log(weights).
    # Let's assume mass_proxy_values are M_j. We need to calculate weights P(G_j|alpha).
    # P(G_j|alpha) = M_j^alpha / sum_k M_k^alpha
    # log P(G_j|alpha) = alpha * log(M_j) - logsumexp_k (alpha * log(M_k))
    
    # Handle alpha = 0 case for weights
    num_hosts = model_d_for_hosts.shape[0]
    if xp.isclose(alpha_g, 0.0):
        # If alpha is close to 0, weights are uniform 1/num_hosts
        log_host_weights = -xp.log(xp.array(num_hosts, dtype=xp.float64))
    else:
        # Ensure mass_proxy_values are positive for log
        safe_mass_proxy = xp.maximum(mass_proxy_values, 1e-30) # Floor to avoid log(0)
        log_mass_proxy = xp.log(safe_mass_proxy)
        
        weighted_log_mass_proxy = alpha_g * log_mass_proxy
        log_sum_weighted_mass_proxy = logsumexp_to_use(weighted_log_mass_proxy)
        log_host_weights = weighted_log_mass_proxy - log_sum_weighted_mass_proxy

    # Combine likelihood per host with host weights
    # log_likelihood_per_host is log P(D|H0, G_j)
    # log_host_weights is log P(G_j|alpha)
    total_log_likelihood = logsumexp_to_use(log_likelihood_per_host + log_host_weights)
    
    # Check for NaN/Inf in final result before returning
    if not xp.isfinite(total_log_likelihood):
        return -xp.inf
        
    return total_log_likelihood


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
        use_vectorized_likelihood=False, # This will be controlled by backend choice/memory limit
        xp_module: object = None, # Added: will be required
        backend_name: str = None, # Added: will be required
    ):
        if xp_module is None or backend_name is None:
            # This case should ideally not be reached if factory is used.
            # If direct instantiation is allowed, this check is important.
            raise ValueError("xp_module and backend_name must be provided to H0LogLikelihood.")

        if dL_gw_samples is None or len(dL_gw_samples) == 0:
            raise ValueError("dL_gw_samples cannot be None or empty.")
        if host_galaxies_z is None or len(host_galaxies_z) == 0:
            raise ValueError("host_galaxies_z cannot be None or empty.")
        if host_galaxies_mass_proxy is None or len(host_galaxies_mass_proxy) == 0:
            raise ValueError("host_galaxies_mass_proxy cannot be None or empty.")
        if host_galaxies_z_err is None or len(host_galaxies_z_err) == 0:
            raise ValueError("host_galaxies_z_err cannot be None or empty.")

        self.dL_gw_samples = np.asarray(dL_gw_samples)
        # Ensure host_galaxies_z is a numpy array for broadcasting
        self.z_values = np.asarray(host_galaxies_z)
        self.mass_proxy_values = np.asarray(host_galaxies_mass_proxy, dtype=float)
        self.z_err_values = np.asarray(host_galaxies_z_err, dtype=float)
        if self.z_values.ndim == 0: # if it was a single scalar
            self.z_values = np.array([self.z_values])
        if self.mass_proxy_values.ndim == 0:
            self.mass_proxy_values = np.array([self.mass_proxy_values])
        if self.z_err_values.ndim == 0:
            self.z_err_values = np.array([self.z_err_values])
        if len(self.mass_proxy_values) != len(self.z_values):
            raise ValueError("host_galaxies_mass_proxy and host_galaxies_z must have the same length")
        if len(self.z_err_values) != len(self.z_values):
            raise ValueError("host_galaxies_z_err and host_galaxies_z must have the same length")
        
        self.sigma_v = sigma_v
        self.c_val = c_val
        self.omega_m_val = omega_m_val
        self.h0_min = h0_min
        self.h0_max = h0_max
        # self.use_vectorized_likelihood will be determined by backend choice later
        # For now, we assume it might be passed or set by a future task (J-4 related)
        self.use_vectorized_likelihood = use_vectorized_likelihood 
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Backend selection and JIT compilation will depend on Task J-4.
        # For now, let's assume self.xp and self.backend_name are set by __init__ args (to be added in J-4)
        self.xp = xp_module
        self.backend_name = backend_name

        # Pre-create a base cosmology object
        self._base_cosmo = FlatLambdaCDM(
            H0=1.0 * u.km / u.s / u.Mpc, Om0=self.omega_m_val
        )
        # Pre-calculate base luminosity distances for hosts (H0=1)
        self._base_dl_for_hosts = self.xp.asarray(self._base_cosmo.luminosity_distance(self.z_values).value)
        
        # Store log_gaussian function
        self._log_gaussian_func = log_gaussian
        self._num_gw_samples = len(self.dL_gw_samples)

        # Store other parameters needed by the core function if they are fixed per event
        self.sigma_v_pec = sigma_v # DEFAULT_SIGMA_V_PEC
        self.c_light = c_val # DEFAULT_C_LIGHT
        # The dl_rel_error_for_sigma_d is new based on the formula
        self.dl_rel_error_for_sigma_d = DL_REL_ERROR_FOR_SIGMA_D


        self._n_quad_points = 5 # For looped likelihood
        self._quad_nodes, self._quad_weights = hermgauss(self._n_quad_points)

        # JIT compilation for JAX backend
        self._jitted_likelihood_core = None
        if self.backend_name == "jax" and jax is not None and self.xp == jnp: # Use self.backend_name and self.xp directly
            try:
                # Arguments for _calculate_vectorized_log_likelihood_core:
                # theta, dL_gw_samples, mass_proxy_values, _base_dl_for_hosts, 
                # sigma_v_pec, c_light, h0_min, h0_max, alpha_min, alpha_max, 
                # xp, log_gaussian_func, num_gw_samples, dl_rel_error_for_sigma_d
                static_arg_names_list = [
                    # "dL_gw_samples", "mass_proxy_values", "_base_dl_for_hosts", # These are arrays, not static
                    "sigma_v_pec", "c_light", "h0_min", "h0_max",
                    "alpha_min", "alpha_max", "xp", "log_gaussian_func", 
                    "num_gw_samples", "dl_rel_error_for_sigma_d"
                ]
                # Ensure all static args are indeed hashable/valid static arguments for JAX JIT
                # xp module and functions can be tricky if not handled carefully.
                # JAX usually traces through functions passed as args unless told otherwise.
                # Let's assume log_gaussian_func is JAX-compatible if xp is jax.numpy
                
                # Forcing xp and log_gaussian_func to be static might be problematic if they are meant to be dynamic.
                # However, for a given H0LogLikelihood instance, these are fixed.
                # JAX can often trace through functions if they are pure.
                # Let's try without them in static_argnames first if issues arise.
                # For now, following the prompt that implies they could be static.
                
                self._jitted_likelihood_core = self.xp.jit(
                    _calculate_vectorized_log_likelihood_core,
                    static_argnames=tuple(static_arg_names_list)
                )
                logger.info("JAX JIT compilation of _calculate_vectorized_log_likelihood_core successful.")
            except Exception as e:
                logger.error(f"JAX JIT compilation failed: {e}. Will use non-JIT path for JAX.")
                self._jitted_likelihood_core = None # Ensure it's None if JIT fails
        elif self.backend_name == "jax" and (jax is None or self.xp != jnp): # Use self.backend_name and self.xp directly
             logger.warning("JAX backend selected, but JAX or jax.numpy (jnp) not available. Cannot JIT.")


    def _lum_dist_model(self, z, H0_val):
        """Compute luminosity distance for ``z`` and ``H0_val``.

        This method reuses a pre-created cosmology instance for efficiency.

        Args:
            z: Redshift value(s).
            H0_val: Hubble constant value.

        Returns:
            Luminosity distance in Mpc units.
        """
        # Ensure H0_val is positive for physical distances
        if isinstance(H0_val, (int, float)) and H0_val <= 0: # Check for scalar H0_val
             # For array H0_val, division by zero or negative will be handled by xp, resulting in inf/nan
            return self.xp.inf # or some other indicator of invalidity
        
        base_distance = self._base_cosmo.luminosity_distance(z).value # This is always numpy from astropy
        # Convert to the current backend type if necessary, though division might handle it
        base_distance_xp = self.xp.asarray(base_distance)
        return base_distance_xp / H0_val

    def __call__(self, theta):
        # theta is expected to be [H0, alpha_g]
        # Ensure theta is an array type compatible with self.xp
        theta_xp = self.xp.asarray(theta)

        # Check if JAX JIT compiled version should be used
        if self.backend_name == "jax" and self._jitted_likelihood_core is not None and self.xp == jnp:
            try:
                return self._jitted_likelihood_core(
                    theta_xp,
                    self.xp.asarray(self.dL_gw_samples), # Ensure it's JAX array if xp is JAX
                    self.xp.asarray(self.mass_proxy_values),
                    self._base_dl_for_hosts, # Already xp.array from init
                    self.sigma_v_pec,
                    self.c_light,
                    self.h0_min,
                    self.h0_max,
                    self.alpha_min,
                    self.alpha_max,
                    self.xp, # jax.numpy
                    self._log_gaussian_func,
                    self._num_gw_samples,
                    self.dl_rel_error_for_sigma_d
                )
            except Exception as e:
                logger.error(f"Error calling JAX JITted function: {e}. Falling back to non-JIT/standard path if possible.")
                # Fallback to standard core calculation or looped depending on config
                # This error path suggests a problem with JIT or its inputs beyond initial compilation
                pass # Will proceed to numpy or looped path based on use_vectorized_likelihood

        # If not JAX JIT, or JIT failed, proceed with NumPy or non-JIT JAX
        # self.use_vectorized_likelihood flag will determine if we use the core func or loop
        
        # Common prior checks (already inside _calculate_vectorized_log_likelihood_core, but good for early exit)
        H0 = theta_xp[0]
        alpha_g = theta_xp[1]
        if not (self.h0_min <= H0 <= self.h0_max): return -self.xp.inf
        if not (self.alpha_min <= alpha_g <= self.alpha_max): return -self.xp.inf
        if H0 <= 0: return -self.xp.inf


        if self.use_vectorized_likelihood:
            # Use the core function directly (either with NumPy or non-JIT JAX)
            return _calculate_vectorized_log_likelihood_core(
                theta_xp,
                self.xp.asarray(self.dL_gw_samples),
                self.xp.asarray(self.mass_proxy_values),
                self._base_dl_for_hosts,
                self.sigma_v_pec,
                self.c_light,
                self.h0_min,
                self.h0_max,
                self.alpha_min,
                self.alpha_max,
                self.xp,
                self._log_gaussian_func,
                self._num_gw_samples,
                self.dl_rel_error_for_sigma_d
            )
        else:
            # --- Memory-Efficient Loop with Redshift Marginalization ---
            # This path needs to be updated to use self.xp and self._log_gaussian_func
            
            # Initial H0 related calculations (formerly outside the loop)
            # model_d_for_hosts_loop = self._lum_dist_model(self.z_values, H0) # Uses self.xp via _lum_dist_model
            # sigma_d_val_for_hosts_loop = (model_d_for_hosts_loop / self.c_light) * self.sigma_v_pec
            # sigma_d_val_for_hosts_loop = self.xp.maximum(sigma_d_val_for_hosts_loop, 1e-9)
            # The above seems too broad if z_err is large. The old code calculated model_d inside quad loop.

            log_P_data_H0_zi_terms = self.xp.zeros(len(self.z_values)) # Use self.xp
            
            # Determine which logsumexp to use
            if hasattr(self.xp, 'scipy') and hasattr(self.xp.scipy, 'special') and hasattr(self.xp.scipy.special, 'logsumexp'):
                logsumexp_to_use_loop = self.xp.scipy.special.logsumexp 
            else:
                logsumexp_to_use_loop = scipy_logsumexp

            for i in range(len(self.z_values)):
                mu_z = self.z_values[i] # This is numpy array from init
                sigma_z = self.z_err_values[i] # This is numpy array from init

                if sigma_z < 1e-4: # If redshift uncertainty is negligible
                    # Calculate dL(z_host_i, H0)
                    # _base_dl_for_hosts is dL(z_host_i, H0=1)
                    current_model_d = self._base_dl_for_hosts[i] / H0
                    if not self.xp.isfinite(current_model_d) or current_model_d <= 0:
                        log_P_data_H0_zi_terms[i] = -self.xp.inf
                        continue
                    
                    # Calculate sigma_d for this specific host
                    term1_sq_loop = self.xp.square(current_model_d * self.sigma_v_pec / self.c_light)
                    term2_sq_loop = self.xp.square(current_model_d * self.dl_rel_error_for_sigma_d)
                    current_sigma_d = self.xp.sqrt(term1_sq_loop + term2_sq_loop)
                    current_sigma_d = self.xp.maximum(current_sigma_d, 1e-9)

                    if not self.xp.isfinite(current_sigma_d) or current_sigma_d <= 0:
                        log_P_data_H0_zi_terms[i] = -self.xp.inf
                        continue

                    # P(dL_gw | H0, z_host_i) using self._log_gaussian_func
                    # self.dL_gw_samples is numpy, ensure it's compatible with self.xp
                    log_pdf_for_one_galaxy = self._log_gaussian_func(
                        self.xp,
                        self.xp.asarray(self.dL_gw_samples), # x
                        current_model_d,    # mu
                        current_sigma_d,    # sigma
                    )
                    
                    # Average over GW samples
                    log_P_data_H0_zi_terms[i] = (
                        logsumexp_to_use_loop(log_pdf_for_one_galaxy) 
                        - self.xp.log(self.xp.array(self._num_gw_samples, dtype=self.xp.float64))
                    )
                else:
                    # Integrate over redshift uncertainty using Gaussian quadrature
                    integrated_prob = 0.0
                    for node, weight in zip(self._quad_nodes, self._quad_weights): # _quad_nodes/weights are numpy
                        z_j = mu_z + self.xp.sqrt(2.0) * sigma_z * node
                        if z_j <= 0:
                            continue
                        
                        # Calculate dL(z_j, H0)
                        # model_d_j = self._lum_dist_model(z_j, H0) # This uses astropy, then self.xp
                        # For efficiency, if _base_cosmo.luminosity_distance can take array z_j:
                        # This part of the loop is tricky for full JAX compatibility if z_j becomes a JAX tracer
                        # For now, _lum_dist_model should handle it correctly by using self.xp internally
                        # However, _base_cosmo is astropy, not JAX. This loop is problematic for JAX.
                        # This non-vectorized path is NOT intended for JAX JIT.
                        # So, direct calls to astropy here are acceptable if self.xp is numpy.
                        # If self.xp is JAX, this path should ideally not be taken if JIT is goal.
                        
                        # Re-implement _lum_dist_model logic here directly for clarity with xp
                        # Astropy's luminosity_distance returns a Quantity, get .value for float
                        base_dl_j = self.xp.asarray(self._base_cosmo.luminosity_distance(z_j).value)
                        model_d_j = base_dl_j / H0

                        if not self.xp.isfinite(model_d_j) or model_d_j <= 0:
                            continue

                        # Calculate sigma_d for this z_j
                        term1_sq_j = self.xp.square(model_d_j * self.sigma_v_pec / self.c_light)
                        term2_sq_j = self.xp.square(model_d_j * self.dl_rel_error_for_sigma_d)
                        sigma_d_j = self.xp.sqrt(term1_sq_j + term2_sq_j)
                        sigma_d_j = self.xp.maximum(sigma_d_j, 1e-9)

                        if not self.xp.isfinite(sigma_d_j) or sigma_d_j <= 0:
                            continue

                        log_pdf_j = self._log_gaussian_func(
                            self.xp,
                            self.xp.asarray(self.dL_gw_samples), # x
                            model_d_j,      # mu
                            sigma_d_j,      # sigma
                        )
                        
                        # P_event is sum_samples P(dL_sample | z_j, H0) / N_samples
                        # log P_event = logsumexp(log_pdf_j) - log(N_samples)
                        # P_event = exp(logsumexp(log_pdf_j) - log(N_samples))
                        log_P_event_j = logsumexp_to_use_loop(log_pdf_j) - self.xp.log(self.xp.array(self._num_gw_samples, dtype=self.xp.float64))
                        P_event_j = self.xp.exp(log_P_event_j)
                        
                        integrated_prob += (weight / self.xp.sqrt(self.xp.pi)) * P_event_j

                    if integrated_prob > 1e-30: # Avoid log(0)
                        log_P_data_H0_zi_terms[i] = self.xp.log(integrated_prob)
                    else:
                        log_P_data_H0_zi_terms[i] = -self.xp.inf
            
            # At this point, log_P_data_H0_zi_terms contains log P(D|H0, G_j) for each host G_j
            # This is equivalent to log_likelihood_per_host from the vectorized version.
            log_sum_over_gw_samples = log_P_data_H0_zi_terms


        # Combine with mass proxy weighting (same logic as in _calculate_vectorized_log_likelihood_core)
        # alpha_g = theta_xp[1] # Already defined
        num_hosts_loop = len(self.mass_proxy_values)
        if self.xp.isclose(alpha_g, 0.0):
            log_host_weights_loop = -self.xp.log(self.xp.array(num_hosts_loop, dtype=self.xp.float64))
        else:
            safe_mass_proxy_loop = self.xp.maximum(self.xp.asarray(self.mass_proxy_values), 1e-30)
            log_mass_proxy_loop = self.xp.log(safe_mass_proxy_loop)
            weighted_log_mass_proxy_loop = alpha_g * log_mass_proxy_loop
            log_sum_weighted_mass_proxy_loop = logsumexp_to_use_loop(weighted_log_mass_proxy_loop)
            log_host_weights_loop = weighted_log_mass_proxy_loop - log_sum_weighted_mass_proxy_loop
        
        # Filter out hosts with -inf likelihood before final sum to avoid propagating NaNs from log(-inf + weight)
        # Or ensure log_host_weights corresponding to -inf log_sum_over_gw_samples don't cause issues.
        # logsumexp handles -inf terms correctly.
        
        valid_terms_mask = self.xp.isfinite(log_sum_over_gw_samples) & self.xp.isfinite(log_host_weights_loop)

        if not self.xp.any(valid_terms_mask):
            return -self.xp.inf
            
        total_log_likelihood = logsumexp_to_use_loop(
            log_sum_over_gw_samples[valid_terms_mask] + log_host_weights_loop[valid_terms_mask]
        )

        if not self.xp.isfinite(total_log_likelihood):
            return -self.xp.inf

        return total_log_likelihood

def get_log_likelihood_h0(
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
    backend_preference: str = "auto", # Added backend_preference
):
    """
    Returns an instance of the H0LogLikelihood class, dynamically choosing
    between looped and vectorized likelihood calculation based on data size.

    Args:
        dL_gw_samples (np.array): Luminosity distance samples from GW event (N_samples,).
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
    # Select backend
    xp_module, backend_name = get_xp(preferred_backend=backend_preference)
    logger.info(f"H0LogLikelihood will use backend: {backend_name}")

    # Determine if vectorized likelihood path should be used
    if backend_name == "jax":
        # JAX always uses the "vectorized" core function (which is JITted)
        should_use_vectorized = True
        logger.info("Using JAX backend: employing JIT-compiled vectorized core function.")
    else: # numpy
        # Dynamic switch for vectorization based on memory for NumPy
        MEMORY_THRESHOLD_BYTES = CONFIG.mcmc.get("vectorized_likelihood_memory_threshold_gb", 4) * (1024**3)
        BYTES_PER_ELEMENT = 8  # for float64 (numpy default)
        
        # Ensure dL_gw_samples is a numpy array for this calculation if coming from JAX
        # However, at this stage, dL_gw_samples is expected to be numpy array as per typical input.
        n_gw_samples = len(np.asarray(dL_gw_samples))
        _host_galaxies_z = np.asarray(host_galaxies_z)
        if _host_galaxies_z.ndim == 0:
            n_hosts = 1
        else:
            n_hosts = len(_host_galaxies_z)
        
        current_elements = n_gw_samples * n_hosts
        max_elements_for_vectorization = MEMORY_THRESHOLD_BYTES / BYTES_PER_ELEMENT
        should_use_vectorized = current_elements <= max_elements_for_vectorization

        if should_use_vectorized:
            logger.info(
                f"Using NumPy backend with VECTORIZED likelihood: N_samples ({n_gw_samples}) * N_hosts ({n_hosts}) = {current_elements} elements. "
                f"Threshold: {max_elements_for_vectorization:.0f} elements ({MEMORY_THRESHOLD_BYTES / (1024**3):.0f} GB)."
            )
        else:
            logger.info(
                f"Using NumPy backend with LOOPED likelihood (memory efficient): N_samples ({n_gw_samples}) * N_hosts ({n_hosts}) = {current_elements} elements. "
                f"Exceeds threshold of {max_elements_for_vectorization:.0f} elements ({MEMORY_THRESHOLD_BYTES / (1024**3):.0f} GB)."
            )

    return H0LogLikelihood(
        dL_gw_samples=dL_gw_samples, # Pass as is, H0LogLikelihood will convert to its xp type if needed
        host_galaxies_z=host_galaxies_z,
        host_galaxies_mass_proxy=host_galaxies_mass_proxy,
        host_galaxies_z_err=host_galaxies_z_err,
        sigma_v=sigma_v,
        c_val=c_val,
        omega_m_val=omega_m_val,
        h0_min=h0_min,
        h0_max=h0_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        use_vectorized_likelihood=should_use_vectorized,
        xp_module=xp_module,
        backend_name=backend_name,
    )

# Removing get_log_likelihood_h0_vectorized as its functionality is covered by
# get_log_likelihood_h0 with backend_preference="jax" or memory-based choice for numpy.
# If a user explicitly wants vectorized numpy and it's above memory limit, that's a user choice issue.
# The default "auto" backend with memory check for numpy is safer.

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