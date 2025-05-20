import numpy as np
import sys
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import emcee
import logging
import cProfile
import pstats
import io
from gwsiren import CONFIG
from gwsiren.backends import log_gaussian, get_xp

logger = logging.getLogger(__name__)

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


def _calculate_vectorized_log_likelihood_core(
    theta: tuple[float, float],
    dL_gw_samples: object,
    mass_proxy_values: object,
    base_dl_for_hosts: object,
    *,
    sigma_v_pec: float,
    c_light: float,
    h0_min: float,
    h0_max: float,
    alpha_min: float,
    alpha_max: float,
    xp: object,
    log_gaussian_func: callable,
    num_gw_samples: int,
) -> float:
    """Core vectorized likelihood calculation.

    This function performs the heavy numerical work of the likelihood using the
    provided backend ``xp``. It contains no side effects and is therefore
    suitable for JIT compilation with JAX.

    Args:
        theta: Tuple of ``(H0, alpha)`` parameters.
        dL_gw_samples: Array of GW luminosity distance samples.
        mass_proxy_values: Mass proxy values for the host galaxies.
        base_dl_for_hosts: Precomputed luminosity distance at ``H0=1`` for each
            host galaxy redshift.
        sigma_v_pec: Peculiar velocity dispersion in km/s.
        c_light: Speed of light in km/s.
        h0_min: Minimum allowed value of ``H0``.
        h0_max: Maximum allowed value of ``H0``.
        alpha_min: Minimum allowed value of ``alpha``.
        alpha_max: Maximum allowed value of ``alpha``.
        xp: Numerical backend module (``numpy`` or ``jax.numpy``).
        log_gaussian_func: Callable implementing the log Gaussian PDF.
        num_gw_samples: Number of GW samples.

    Returns:
        The log-likelihood value. ``-xp.inf`` if parameters are outside the
        allowed range or numerical issues occur.
    """

    H0, alpha = theta

    if not (h0_min <= H0 <= h0_max):
        return -xp.inf
    if not (alpha_min <= alpha <= alpha_max):
        return -xp.inf

    dL_gw_samples = xp.asarray(dL_gw_samples)
    mass_proxy_values = xp.asarray(mass_proxy_values)
    base_dl_for_hosts = xp.asarray(base_dl_for_hosts)

    model_d_for_hosts = base_dl_for_hosts / H0
    if xp.any(~xp.isfinite(model_d_for_hosts)):
        return -xp.inf

    sigma_d_val_for_hosts = (model_d_for_hosts / c_light) * sigma_v_pec
    sigma_d_val_for_hosts = xp.maximum(sigma_d_val_for_hosts, 1e-9)
    if xp.any(~xp.isfinite(sigma_d_val_for_hosts)):
        return -xp.inf

    log_pdf_values_full = log_gaussian_func(
        xp,
        dL_gw_samples[:, xp.newaxis],
        model_d_for_hosts[xp.newaxis, :],
        sigma_d_val_for_hosts[xp.newaxis, :],
    )
    log_pdf_values_full = xp.nan_to_num(log_pdf_values_full, nan=-xp.inf)

    log_sum_over_gw_samples = (
        xp.logaddexp.reduce(log_pdf_values_full, axis=0)
        - xp.log(num_gw_samples)
    )

    if xp.isclose(alpha, 0.0):
        weights = xp.full(mass_proxy_values.shape[0], 1.0 / mass_proxy_values.shape[0])
    else:
        powered = mass_proxy_values ** alpha
        if xp.any(powered <= 0) or xp.any(~xp.isfinite(powered)):
            return -xp.inf
        denom = powered.sum()
        if denom <= 0 or ~xp.isfinite(denom):
            return -xp.inf
        weights = powered / denom

    valid_mask = (weights > 0) & xp.isfinite(log_sum_over_gw_samples)
    if xp.sum(valid_mask) == 0:
        return -xp.inf

    total_log_likelihood = xp.logaddexp.reduce(
        xp.log(weights[valid_mask]) + log_sum_over_gw_samples[valid_mask]
    )

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
        xp: Numerical backend module to use (defaults to ``numpy``).
        backend_name: Name of the selected backend.
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
        use_vectorized_likelihood=False,
        *,
        xp=np,
        backend_name="numpy",
    ):
        
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
        self.use_vectorized_likelihood = use_vectorized_likelihood
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.xp = xp
        self.backend_name = backend_name

        # Pre-create a base cosmology object to avoid repeated construction
        # during likelihood evaluations. Using ``H0=1`` allows scaling
        # distances for arbitrary ``H0`` values without recomputing the
        # cosmology internals.
        self._base_cosmo = FlatLambdaCDM(
            H0=1.0 * u.km / u.s / u.Mpc, Om0=self.omega_m_val
        )
        self._base_dl_for_hosts = self._base_cosmo.luminosity_distance(self.z_values).value

        self._log_gaussian_func = log_gaussian
        self._num_gw_samples = len(self.dL_gw_samples)

        if self.backend_name == "jax" and hasattr(self.xp, "jit"):
            static_arg_names = (
                "sigma_v_pec",
                "c_light",
                "h0_min",
                "h0_max",
                "alpha_min",
                "alpha_max",
                "xp",
                "log_gaussian_func",
                "num_gw_samples",
            )
            self._jitted_likelihood_core = self.xp.jit(
                _calculate_vectorized_log_likelihood_core,
                static_argnames=static_arg_names,
            )
        else:
            self._jitted_likelihood_core = None

        self._n_quad_points = 5
        self._quad_nodes, self._quad_weights = hermgauss(self._n_quad_points)

    def _lum_dist_model(self, z, H0_val):
        """Compute luminosity distance for ``z`` and ``H0_val``.

        This method reuses a pre-created cosmology instance for efficiency.

        Args:
            z: Redshift value(s).
            H0_val: Hubble constant value.

        Returns:
            Luminosity distance in Mpc units.
        """
        base_distance = self._base_cosmo.luminosity_distance(z).value
        return base_distance / H0_val

    def __call__(self, theta):
        if self.backend_name == "jax" and self._jitted_likelihood_core is not None:
            return self._jitted_likelihood_core(
                theta=theta,
                dL_gw_samples=self.dL_gw_samples,
                mass_proxy_values=self.mass_proxy_values,
                base_dl_for_hosts=self._base_dl_for_hosts,
                sigma_v_pec=self.sigma_v,
                c_light=self.c_val,
                h0_min=self.h0_min,
                h0_max=self.h0_max,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                xp=self.xp,
                log_gaussian_func=self._log_gaussian_func,
                num_gw_samples=self._num_gw_samples,
            )

        if self.use_vectorized_likelihood:
            return _calculate_vectorized_log_likelihood_core(
                theta=theta,
                dL_gw_samples=self.dL_gw_samples,
                mass_proxy_values=self.mass_proxy_values,
                base_dl_for_hosts=self._base_dl_for_hosts,
                sigma_v_pec=self.sigma_v,
                c_light=self.c_val,
                h0_min=self.h0_min,
                h0_max=self.h0_max,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                xp=self.xp,
                log_gaussian_func=self._log_gaussian_func,
                num_gw_samples=self._num_gw_samples,
            )

        # --- Memory-Efficient Loop with Redshift Marginalization ---
        H0, alpha = theta
        if not (self.h0_min <= H0 <= self.h0_max):
            return -self.xp.inf

        try:
            model_d_for_hosts = self._lum_dist_model(self.z_values, H0)
            if self.xp.any(~self.xp.isfinite(model_d_for_hosts)):
                return -self.xp.inf
        except Exception:
            return -self.xp.inf

        sigma_d_val_for_hosts = (model_d_for_hosts / self.c_val) * self.sigma_v
        sigma_d_val_for_hosts = self.xp.maximum(sigma_d_val_for_hosts, 1e-9)
        if self.xp.any(~self.xp.isfinite(sigma_d_val_for_hosts)):
            return -self.xp.inf

        log_P_data_H0_zi_terms = self.xp.zeros(len(self.z_values))
        for i in range(len(self.z_values)):
            mu_z = self.z_values[i]
            sigma_z = self.z_err_values[i]

            if sigma_z < 1e-4:
                current_model_d = model_d_for_hosts[i]
                current_sigma_d = sigma_d_val_for_hosts[i]
                log_pdf_for_one_galaxy = self._log_gaussian_func(
                    self.xp,
                    self.dL_gw_samples,
                    current_model_d,
                    current_sigma_d,
                )
                if self.xp.any(~self.xp.isfinite(log_pdf_for_one_galaxy)):
                    log_P_data_H0_zi_terms[i] = -self.xp.inf
                else:
                    log_P_data_H0_zi_terms[i] = (
                        self.xp.logaddexp.reduce(log_pdf_for_one_galaxy)
                        - self.xp.log(self._num_gw_samples)
                    )
                continue

            integrated_prob = 0.0
            for node, weight in zip(self._quad_nodes, self._quad_weights):
                z_j = mu_z + self.xp.sqrt(2.0) * sigma_z * node
                if z_j <= 0:
                    continue
                model_d = self._lum_dist_model(z_j, H0)
                sigma_d = self.xp.maximum((model_d / self.c_val) * self.sigma_v, 1e-9)
                log_pdf = self._log_gaussian_func(
                    self.xp,
                    self.dL_gw_samples,
                    model_d,
                    sigma_d,
                )
                if self.xp.any(~self.xp.isfinite(log_pdf)):
                    continue
                P_event = self.xp.exp(
                    self.xp.logaddexp.reduce(log_pdf) - self.xp.log(self._num_gw_samples)
                )
                integrated_prob += (weight / self.xp.sqrt(self.xp.pi)) * P_event

            if integrated_prob > 0:
                log_P_data_H0_zi_terms[i] = self.xp.log(integrated_prob)
            else:
                log_P_data_H0_zi_terms[i] = -self.xp.inf

        log_sum_over_gw_samples = log_P_data_H0_zi_terms

        alpha = theta[1]
        if not (self.alpha_min <= alpha <= self.alpha_max):
            return -self.xp.inf

        if self.xp.isclose(alpha, 0.0):
            weights = self.xp.full(
                len(self.mass_proxy_values), 1.0 / len(self.mass_proxy_values)
            )
        else:
            powered = self.mass_proxy_values ** alpha
            if self.xp.any(powered <= 0) or self.xp.any(~self.xp.isfinite(powered)):
                return -self.xp.inf
            denom = powered.sum()
            if denom <= 0 or not np.isfinite(denom):
                return -self.xp.inf
            weights = powered / denom

        valid_mask = (weights > 0) & self.xp.isfinite(log_sum_over_gw_samples)
        if not self.xp.any(valid_mask):
            return -self.xp.inf

        total_log_likelihood = self.xp.logaddexp.reduce(
            self.xp.log(weights[valid_mask]) + log_sum_over_gw_samples[valid_mask]
        )

        if not self.xp.isfinite(total_log_likelihood):
            return -self.xp.inf

        return float(total_log_likelihood)

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
    *,
    backend_preference: str = "auto",
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

    xp_mod, backend_name = get_xp(backend_preference)

    return H0LogLikelihood(
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
        xp=xp_mod,
        backend_name=backend_name,
    )

def get_log_likelihood_h0_vectorized(
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
):
    return H0LogLikelihood(
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
        use_vectorized_likelihood=True,
    )

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