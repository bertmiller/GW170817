import numpy as np
import sys
from numpy.polynomial.hermite import hermgauss
import emcee
import logging
from gwsiren import CONFIG
from gwsiren.backends import get_xp, logpdf_normal_xp, logsumexp_xp, trapz_xp
from gwsiren.distance_cache import create_distance_cache
from dataclasses import dataclass

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
DEFAULT_MCMC_INITIAL_H0_MEAN = CONFIG.mcmc_initial_positions.h0_mean
DEFAULT_MCMC_INITIAL_H0_STD = CONFIG.mcmc_initial_positions.h0_std
DEFAULT_H0_PRIOR_MIN = CONFIG.mcmc["prior_h0_min"]  # km/s/Mpc
DEFAULT_H0_PRIOR_MAX = CONFIG.mcmc["prior_h0_max"]  # km/s/Mpc
DEFAULT_ALPHA_PRIOR_MIN = CONFIG.mcmc.get("prior_alpha_min", -1.0)
DEFAULT_ALPHA_PRIOR_MAX = CONFIG.mcmc.get("prior_alpha_max", 1.0)
DEFAULT_MCMC_INITIAL_ALPHA_MEAN = CONFIG.mcmc_initial_positions.alpha_mean
DEFAULT_MCMC_INITIAL_ALPHA_STD = CONFIG.mcmc_initial_positions.alpha_std

# Redshift marginalization parameters - now from config
DEFAULT_Z_ERR_THRESHOLD = CONFIG.redshift_marginalization.z_err_threshold
DEFAULT_QUAD_POINTS = CONFIG.redshift_marginalization.n_quad_points
DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE = CONFIG.redshift_marginalization.sigma_range

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
        xp,
        backend_name,
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
        z_err_threshold=DEFAULT_Z_ERR_THRESHOLD,
        n_quad_points=DEFAULT_QUAD_POINTS,
        z_sigma_range=DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE,
    ):
        """Initialize H0LogLikelihood with backend-agnostic computation setup.
        
        Args:
            xp: Backend computation module (numpy or jax.numpy)
            backend_name: Name of the backend ("numpy" or "jax")
            dL_gw_samples: GW luminosity distance samples
            host_galaxies_z: Host galaxy redshifts
            host_galaxies_mass_proxy: Galaxy mass proxy values
            host_galaxies_z_err: Redshift uncertainties
            ... (other parameters as before)
        """
        # 1. Store backend configuration
        self._setup_backend_config(xp, backend_name)
        
        # 2. Validate and process input data
        self._setup_input_data(
            dL_gw_samples, host_galaxies_z, host_galaxies_mass_proxy, host_galaxies_z_err
        )
        
        # 3. Store computational parameters
        self._setup_computational_parameters(
            sigma_v, c_val, omega_m_val, h0_min, h0_max, alpha_min, alpha_max,
            use_vectorized_likelihood, z_err_threshold, n_quad_points, z_sigma_range
        )
        
        # 4. Initialize quadrature integration
        self._setup_quadrature_integration(n_quad_points)
        
        # 5. Apply backend-specific optimizations
        self._apply_backend_optimizations()

    def _setup_backend_config(self, xp, backend_name):
        """Store backend configuration for computation.
        
        Args:
            xp: Backend computation module
            backend_name: Name of the backend
        """
        self.xp = xp
        self.backend_name = backend_name

    def _setup_input_data(self, dL_gw_samples, host_galaxies_z, host_galaxies_mass_proxy, host_galaxies_z_err):
        """Validate and convert input data to backend arrays.
        
        Args:
            dL_gw_samples: GW distance samples
            host_galaxies_z: Host redshifts  
            host_galaxies_mass_proxy: Mass proxy values
            host_galaxies_z_err: Redshift uncertainties
        """
        # Validate inputs using NumPy for robustness
        self._validate_input_data(dL_gw_samples, host_galaxies_z, host_galaxies_mass_proxy, host_galaxies_z_err)
        
        # Convert to backend arrays
        self.dL_gw_samples = self.xp.asarray(dL_gw_samples)
        
        # Process galaxy data with proper dimensionality handling
        self.z_values, self.mass_proxy_values, self.z_err_values = self._process_galaxy_data(
            host_galaxies_z, host_galaxies_mass_proxy, host_galaxies_z_err
        )
        
        # Validate data consistency
        self._validate_data_consistency()

    def _validate_input_data(self, dL_gw_samples, host_galaxies_z, host_galaxies_mass_proxy, host_galaxies_z_err):
        """Validate input data before processing.
        
        Args:
            dL_gw_samples: GW distance samples
            host_galaxies_z: Host redshifts
            host_galaxies_mass_proxy: Mass proxy values
            host_galaxies_z_err: Redshift uncertainties
            
        Raises:
            ValueError: If any input is invalid
        """
        if dL_gw_samples is None or len(np.asarray(dL_gw_samples)) == 0:
            raise ValueError("dL_gw_samples cannot be None or empty.")
        if host_galaxies_z is None or len(np.asarray(host_galaxies_z)) == 0:
            raise ValueError("host_galaxies_z cannot be None or empty.")
        if host_galaxies_mass_proxy is None or len(np.asarray(host_galaxies_mass_proxy)) == 0:
            raise ValueError("host_galaxies_mass_proxy cannot be None or empty.")
        if host_galaxies_z_err is None or len(np.asarray(host_galaxies_z_err)) == 0:
            raise ValueError("host_galaxies_z_err cannot be None or empty.")

    def _process_galaxy_data(self, host_galaxies_z, host_galaxies_mass_proxy, host_galaxies_z_err):
        """Process galaxy data ensuring proper dimensionality.
        
        Args:
            host_galaxies_z: Host redshifts
            host_galaxies_mass_proxy: Mass proxy values  
            host_galaxies_z_err: Redshift uncertainties
            
        Returns:
            tuple: (z_values, mass_proxy_values, z_err_values) as backend arrays
        """
        # Convert to NumPy first for reliable shape handling
        z_values_np = np.asarray(host_galaxies_z, dtype=float)
        mass_proxy_values_np = np.asarray(host_galaxies_mass_proxy, dtype=float)
        z_err_values_np = np.asarray(host_galaxies_z_err, dtype=float)

        # Ensure at least 1D arrays
        if z_values_np.ndim == 0:
            z_values_np = np.array([z_values_np])
        if mass_proxy_values_np.ndim == 0:
            mass_proxy_values_np = np.array([mass_proxy_values_np])
        if z_err_values_np.ndim == 0:
            z_err_values_np = np.array([z_err_values_np])
        
        # Convert to backend arrays
        z_values = self.xp.asarray(z_values_np)
        mass_proxy_values = self.xp.asarray(mass_proxy_values_np)
        z_err_values = self.xp.asarray(z_err_values_np)
        
        return z_values, mass_proxy_values, z_err_values

    def _validate_data_consistency(self):
        """Validate that all galaxy data arrays have consistent lengths.
        
        Raises:
            ValueError: If array lengths are inconsistent
        """
        if len(self.mass_proxy_values) != len(self.z_values):
            raise ValueError("host_galaxies_mass_proxy and host_galaxies_z must have the same length")
        if len(self.z_err_values) != len(self.z_values):
            raise ValueError("host_galaxies_z_err and host_galaxies_z must have the same length")

    def _setup_computational_parameters(self, sigma_v, c_val, omega_m_val, h0_min, h0_max, 
                                      alpha_min, alpha_max, use_vectorized_likelihood,
                                      z_err_threshold, n_quad_points, z_sigma_range):
        """Store computational parameters for likelihood evaluation.
        
        Args:
            sigma_v: Peculiar velocity dispersion
            c_val: Speed of light
            omega_m_val: Matter density parameter
            h0_min, h0_max: H0 prior bounds
            alpha_min, alpha_max: Alpha prior bounds
            use_vectorized_likelihood: Vectorization flag
            z_err_threshold: Redshift error threshold for marginalization
            n_quad_points: Number of quadrature points
            z_sigma_range: Range for redshift integration
        """
        # Physical parameters
        self.sigma_v = sigma_v
        self.c_val = c_val
        self.omega_m_val = omega_m_val
        
        # Prior bounds
        self.h0_min = h0_min
        self.h0_max = h0_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Computational settings
        self.use_vectorized_likelihood = use_vectorized_likelihood
        self.z_err_threshold = z_err_threshold
        self.n_quad_points = n_quad_points
        self.z_sigma_range = z_sigma_range

    def _setup_quadrature_integration(self, n_quad_points):
        """Initialize Gaussian quadrature nodes and weights.
        
        Args:
            n_quad_points: Number of quadrature points
        """
        # Generate quadrature nodes and weights using NumPy
        quad_nodes_np, quad_weights_np = hermgauss(n_quad_points)
        
        # Convert to backend arrays
        self._quad_nodes = self.xp.asarray(quad_nodes_np)
        self._quad_weights = self.xp.asarray(quad_weights_np)

    def _apply_backend_optimizations(self):
        """Apply backend-specific optimizations like JIT compilation.
        
        For JAX backend, this applies JIT compilation to the __call__ method.
        For NumPy backend, this is a no-op.
        """
        # Initialize distance cache for performance optimization
        self._distance_cache = create_distance_cache(
            max_cache_size=100,  # Cache up to 100 H0 values
            z_range=(0.001, 3.0),  # Reasonable redshift range
            z_points=2000,  # High resolution for accuracy
            h0_tolerance=0.01  # Group H0 values within 0.01 km/s/Mpc
        )
        
        # Set up the uncached distance function for the cache to use
        self._distance_cache.set_distance_function(self._lum_dist_model_uncached)
        
        if self.backend_name == "jax":
            self._apply_jax_optimizations()

    def _apply_jax_optimizations(self):
        """Apply JAX-specific optimizations like JIT compilation."""
        if self.backend_name == "jax":
            # For JAX backend, we can JIT compile the main computation
            # This is a placeholder for future JAX-specific optimizations
            pass
    
    def _lum_dist_model_uncached(self, z_values, H0_val, N_trapz=200):
        """Uncached version of luminosity distance computation.
        
        This is the original computation that the cache will use to build
        interpolation tables. Should not be called directly - use _lum_dist_model instead.
        """
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

    def _lum_dist_model(self, z_values, H0_val, N_trapz=200):
        """Compute luminosity distance for z_values (array or scalar) and H0_val.
        
        Uses distance cache for performance optimization. This replaces the original
        distance computation with a cached version that builds interpolation tables.
        """
        # Use the distance cache for optimized computation
        result = self._distance_cache.get_distances(z_values, H0_val)
        
        # Convert result to backend array if needed
        # The distance cache returns NumPy arrays/scalars, we need to convert to backend arrays
        if self.backend_name == 'jax':
            # For JAX, always convert to JAX arrays
            if self.xp.isscalar(z_values):
                return self.xp.array(result)
            else:
                return self.xp.asarray(result)
        else:
            # For NumPy, check if conversion is needed
            if hasattr(result, '__array__') and hasattr(result, 'dtype'):
                # Already an array-like object
                return self.xp.asarray(result)
            else:
                # Scalar or other type
                if self.xp.isscalar(z_values):
                    return self.xp.array(result)
                else:
                    return self.xp.asarray(result)

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

    def __call__(self, theta):
        """Compute log-likelihood for H0 and alpha parameters."""
        H0, alpha = theta[0], theta[1]
        
        # 1. Validate parameters
        if not self._validate_parameters(H0, alpha):
            return -self.xp.inf
            
        # 2. Compute distances and uncertainties  
        distances_result = self._compute_distances_and_uncertainties(H0)
        if distances_result is None:
            return -self.xp.inf
        model_distances, distance_uncertainties = distances_result
            
        # 3. Compute likelihood contributions from all galaxies
        galaxy_log_likelihoods = self._compute_galaxy_likelihoods(
            model_distances, distance_uncertainties, H0
        )
        if galaxy_log_likelihoods is None:
            return -self.xp.inf
            
        # 4. Apply mass weighting based on alpha
        galaxy_weights = self._compute_galaxy_weights(alpha)  
        if galaxy_weights is None:
            return -self.xp.inf
            
        # 5. Combine weighted likelihoods
        return self._combine_weighted_likelihoods(galaxy_log_likelihoods, galaxy_weights)

    def _validate_parameters(self, H0, alpha):
        """Validate H0 and alpha are within their respective prior bounds.
        
        Args:
            H0 (float): Hubble constant value to validate
            alpha (float): Mass bias parameter to validate
            
        Returns:
            bool: True if both parameters are within bounds, False otherwise
        """
        h0_valid = self.h0_min <= H0 <= self.h0_max
        alpha_valid = self.alpha_min <= alpha <= self.alpha_max
        return h0_valid and alpha_valid
        
    def _compute_distances_and_uncertainties(self, H0):
        """Compute model luminosity distances and their uncertainties.
        
        Args:
            H0 (float): Hubble constant value
            
        Returns:
            tuple: (model_distances, distance_uncertainties) or None if computation fails
        """
        try:
            # Compute luminosity distances for all host galaxies
            model_d_for_hosts = self._lum_dist_model(self.z_values, H0)
            
            # Check for finite values
            if self.xp.any(~self.xp.isfinite(model_d_for_hosts)):
                return None
                
            # Compute distance uncertainties from peculiar velocity
            sigma_d_val_for_hosts = (model_d_for_hosts / self.c_val) * self.sigma_v
            sigma_d_val_for_hosts = self.xp.maximum(sigma_d_val_for_hosts, 1e-9)
            
            # Check for finite uncertainties
            if self.xp.any(~self.xp.isfinite(sigma_d_val_for_hosts)):
                return None
                
            return model_d_for_hosts, sigma_d_val_for_hosts
            
        except Exception:
            return None

    def _compute_galaxy_likelihoods(self, model_distances, distance_uncertainties, H0):
        """Compute likelihood contributions from all galaxies.
        
        Dispatches to either vectorized or looped implementation based on 
        the use_vectorized_likelihood setting.
        
        Args:
            model_distances: Luminosity distances for each galaxy
            distance_uncertainties: Distance uncertainties for each galaxy  
            H0 (float): Hubble constant value
            
        Returns:
            array: Log-likelihood contribution for each galaxy, or None if computation fails
        """
        if self.use_vectorized_likelihood:
            return self._compute_galaxy_likelihoods_vectorized(
                model_distances, distance_uncertainties, H0
            )
        else:
            return self._compute_galaxy_likelihoods_looped(
                model_distances, distance_uncertainties, H0
            )

    def _compute_galaxy_likelihoods_vectorized(self, model_distances, distance_uncertainties, H0):
        """Compute galaxy likelihoods using vectorized operations for memory efficiency.
        
        Args:
            model_distances: Luminosity distances for each galaxy
            distance_uncertainties: Distance uncertainties for each galaxy
            H0 (float): Hubble constant value
            
        Returns:
            array: Log-likelihood contribution for each galaxy, or None if computation fails
        """
        # Check which galaxies need redshift marginalization
        needs_marginalization = self.z_err_values >= self.z_err_threshold
        
        if not self.xp.any(needs_marginalization):
            # All galaxies below threshold - use simple vectorized computation
            return self._compute_simple_vectorized_likelihoods(
                model_distances, distance_uncertainties
            )
        else:
            # Some galaxies need marginalization
            return self._compute_mixed_vectorized_likelihoods(
                model_distances, distance_uncertainties, H0, needs_marginalization
            )

    def _compute_simple_vectorized_likelihoods(self, model_distances, distance_uncertainties):
        """Compute likelihoods when no redshift marginalization is needed.
        
        Args:
            model_distances: Luminosity distances for each galaxy
            distance_uncertainties: Distance uncertainties for each galaxy
            
        Returns:
            array: Log-likelihood contribution for each galaxy, or None if computation fails
        """
        try:
            # Ensure arrays are properly shaped for broadcasting
            # Convert to backend arrays and ensure they're at least 1D
            model_distances = self.xp.atleast_1d(self.xp.asarray(model_distances))
            distance_uncertainties = self.xp.atleast_1d(self.xp.asarray(distance_uncertainties))
            dL_gw_samples = self.xp.atleast_1d(self.xp.asarray(self.dL_gw_samples))
            
            # Create properly shaped arrays for broadcasting
            # Shape: (N_samples, 1) and (1, N_hosts)
            gw_samples_expanded = dL_gw_samples[:, self.xp.newaxis]
            model_distances_expanded = model_distances[self.xp.newaxis, :]
            distance_uncertainties_expanded = distance_uncertainties[self.xp.newaxis, :]
            
            log_pdf_values_full = logpdf_normal_xp(
                self.xp,
                gw_samples_expanded,
                loc=model_distances_expanded,
                scale=distance_uncertainties_expanded
            )  # Shape: (N_samples, N_hosts)
        except Exception:
            return None

        # Handle potential NaNs by converting them to -inf
        log_pdf_values_full = self.xp.where(
            self.xp.isnan(log_pdf_values_full), -self.xp.inf, log_pdf_values_full
        )

        # Sum over GW samples and normalize
        log_sum_over_gw_samples = logsumexp_xp(self.xp, log_pdf_values_full, axis=0) - \
                                 self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
        
        return log_sum_over_gw_samples

    def _compute_mixed_vectorized_likelihoods(self, model_distances, distance_uncertainties, H0, needs_marginalization):
        """Compute likelihoods when some galaxies need redshift marginalization.
        
        Args:
            model_distances: Luminosity distances for each galaxy
            distance_uncertainties: Distance uncertainties for each galaxy  
            H0 (float): Hubble constant value
            needs_marginalization: Boolean mask indicating which galaxies need marginalization
            
        Returns:
            array: Log-likelihood contribution for each galaxy, or None if computation fails
        """
        log_sum_over_gw_samples = self.xp.zeros(len(self.z_values))
        
        # Process galaxies that don't need marginalization
        no_marg_mask = ~needs_marginalization
        if self.xp.any(no_marg_mask):
            no_marg_result = self._process_non_marginalized_galaxies_vectorized(
                model_distances, distance_uncertainties, no_marg_mask
            )
            if no_marg_result is not None:
                log_sum_no_marg, no_marg_indices = no_marg_result
                # Update results for non-marginalized galaxies
                for i, idx in enumerate(no_marg_indices):
                    log_sum_over_gw_samples = self._update_array_element(
                        log_sum_over_gw_samples, idx, log_sum_no_marg[i]
                    )
            else:
                # Set non-marginalized galaxies to -inf if computation fails
                for idx in self.xp.where(no_marg_mask)[0]:
                    log_sum_over_gw_samples = self._update_array_element(
                        log_sum_over_gw_samples, idx, -self.xp.inf
                    )
        
        # Process galaxies that need marginalization
        marginalized_result = self._process_marginalized_galaxies_vectorized(
            model_distances, distance_uncertainties, H0, needs_marginalization
        )
        if marginalized_result is not None:
            marg_likelihoods, marg_indices = marginalized_result
            for i, idx in enumerate(marg_indices):
                log_sum_over_gw_samples = self._update_array_element(
                    log_sum_over_gw_samples, idx, marg_likelihoods[i]
                )
        
        # Check for any non-finite values in final result
        if self.xp.any(~self.xp.isfinite(log_sum_over_gw_samples)):
            return None
            
        return log_sum_over_gw_samples

    def _process_non_marginalized_galaxies_vectorized(self, model_distances, distance_uncertainties, no_marg_mask):
        """Process galaxies that don't need redshift marginalization in vectorized mode.
        
        Args:
            model_distances: Luminosity distances for each galaxy
            distance_uncertainties: Distance uncertainties for each galaxy
            no_marg_mask: Boolean mask for galaxies that don't need marginalization
            
        Returns:
            tuple: (log_likelihoods, indices) or None if computation fails
        """
        try:
            log_pdf_no_marg = logpdf_normal_xp(
                self.xp,
                self.dL_gw_samples[:, self.xp.newaxis],
                loc=model_distances[self.xp.newaxis, no_marg_mask],
                scale=distance_uncertainties[self.xp.newaxis, no_marg_mask]
            )
            log_pdf_no_marg = self.xp.where(self.xp.isnan(log_pdf_no_marg), -self.xp.inf, log_pdf_no_marg)
            log_sum_no_marg = logsumexp_xp(self.xp, log_pdf_no_marg, axis=0) - \
                             self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
            
            no_marg_indices = self.xp.where(no_marg_mask)[0]
            return log_sum_no_marg, no_marg_indices
        except Exception:
            return None

    def _process_marginalized_galaxies_vectorized(self, model_distances, distance_uncertainties, H0, needs_marginalization):
        """Process galaxies that need redshift marginalization in vectorized mode.
        
        Args:
            model_distances: Luminosity distances for each galaxy
            distance_uncertainties: Distance uncertainties for each galaxy
            H0 (float): Hubble constant value
            needs_marginalization: Boolean mask for galaxies that need marginalization
            
        Returns:
            tuple: (log_likelihoods, indices) or None if computation fails
        """
        marg_indices = self.xp.where(needs_marginalization)[0]
        marg_likelihoods = []
        
        # Convert to Python list for iteration if JAX backend
        if self.backend_name == "jax":
            indices_list = [int(idx) for idx in marg_indices]
        else:
            indices_list = marg_indices
        
        for idx in indices_list:
            # Extract scalar values for JAX compatibility
            if self.backend_name == "jax":
                model_dist = float(model_distances[idx])
            else:
                model_dist = model_distances[idx]
                
            likelihood = self._marginalize_single_galaxy_redshift_vectorized(
                idx, model_dist, H0
            )
            marg_likelihoods.append(likelihood)
        
        return self.xp.array(marg_likelihoods), marg_indices

    def _marginalize_single_galaxy_redshift_vectorized(self, galaxy_idx, model_distance, H0):
        """Perform redshift marginalization for a single galaxy using vectorized quadrature.
        
        Args:
            galaxy_idx (int): Index of the galaxy to marginalize
            model_distance (float): Model luminosity distance for this galaxy
            H0 (float): Hubble constant value
            
        Returns:
            float: Marginalized log-likelihood for this galaxy
        """
        mu_z = self.z_values[galaxy_idx]
        sigma_z = self.z_err_values[galaxy_idx]
        
        # Compute integration bounds
        z_min = self.xp.maximum(mu_z - self.z_sigma_range * sigma_z, 1e-6)
        z_max = mu_z + self.z_sigma_range * sigma_z
        
        if z_max <= z_min:
            # Fallback to point estimate
            return self._compute_point_estimate_likelihood(model_distance)
        
        try:
            return self._perform_vectorized_quadrature_integration(
                mu_z, sigma_z, H0
            )
        except Exception:
            # Integration failed - return -inf
            return -self.xp.inf

    def _compute_point_estimate_likelihood(self, model_distance):
        """Compute likelihood using point estimate (no marginalization).
        
        Args:
            model_distance (float): Model luminosity distance
            
        Returns:
            float: Log-likelihood value
        """
        try:
            sigma_d = self.xp.maximum((model_distance / self.c_val) * self.sigma_v, 1e-9)
            log_pdf_point = logpdf_normal_xp(
                self.xp,
                self.dL_gw_samples,
                loc=model_distance,
                scale=sigma_d
            )
            if self.xp.any(~self.xp.isfinite(log_pdf_point)):
                return -self.xp.inf
            else:
                reduced_log_pdf = logsumexp_xp(self.xp, log_pdf_point)
                return reduced_log_pdf - self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
        except Exception:
            return -self.xp.inf

    def _perform_vectorized_quadrature_integration(self, mu_z, sigma_z, H0):
        """Perform Gaussian quadrature integration for redshift marginalization.
        
        Args:
            mu_z (float): Mean redshift
            sigma_z (float): Redshift uncertainty
            H0 (float): Hubble constant value
            
        Returns:
            float: Integrated log-likelihood
        """
        # Transform quadrature nodes to redshift domain
        z_quad = mu_z + self.xp.sqrt(2.0) * sigma_z * self._quad_nodes
        
        # Filter out negative redshifts
        valid_z_mask = z_quad > 0
        if not self.xp.any(valid_z_mask):
            return -self.xp.inf
        
        z_quad_valid = z_quad[valid_z_mask]
        quad_weights_valid = self._quad_weights[valid_z_mask]
        
        # Vectorized distance computation for all quadrature points
        try:
            model_d_quad = self._lum_dist_model(z_quad_valid, H0)
            # Ensure it's a backend array
            model_d_quad = self.xp.asarray(model_d_quad)
        except Exception:
            # Fallback to element-wise computation if vectorized fails
            model_d_list = []
            for z_val in z_quad_valid:
                try:
                    # Convert JAX array element to scalar for compatibility
                    z_scalar = float(z_val) if self.backend_name == "jax" else z_val
                    d_val = self._lum_dist_model(z_scalar, H0)
                    model_d_list.append(d_val)
                except Exception:
                    return -self.xp.inf
            
            if not model_d_list:
                return -self.xp.inf
            model_d_quad = self.xp.array(model_d_list)
        
        # Check for valid distances
        valid_d_mask = (self.xp.isfinite(model_d_quad)) & (model_d_quad > 0)
        if not self.xp.any(valid_d_mask):
            return -self.xp.inf
        
        model_d_quad_valid = model_d_quad[valid_d_mask]
        quad_weights_final = quad_weights_valid[valid_d_mask]
        
        # Compute sigma_d for each quadrature point
        sigma_d_quad = self.xp.maximum((model_d_quad_valid / self.c_val) * self.sigma_v, 1e-9)
        
        # Vectorized likelihood computation across quadrature points
        log_pdf_quad = logpdf_normal_xp(
            self.xp,
            self.dL_gw_samples[:, self.xp.newaxis],
            loc=model_d_quad_valid[self.xp.newaxis, :],
            scale=sigma_d_quad[self.xp.newaxis, :]
        )
        
        # Check for finite values
        if self.xp.any(~self.xp.isfinite(log_pdf_quad)):
            log_pdf_quad = self.xp.where(self.xp.isnan(log_pdf_quad), -self.xp.inf, log_pdf_quad)
        
        # Sum over GW samples for each quadrature point
        log_likelihood_quad = logsumexp_xp(self.xp, log_pdf_quad, axis=0) - \
                             self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
        
        # Apply quadrature weights and integrate
        log_weights_quad = self.xp.log(quad_weights_final / self.xp.sqrt(self.xp.pi))
        log_integrand_quad = log_weights_quad + log_likelihood_quad
        
        # Final integration using logsumexp
        return logsumexp_xp(self.xp, log_integrand_quad)

    def _compute_galaxy_likelihoods_looped(self, model_distances, distance_uncertainties, H0):
        """Compute galaxy likelihoods using memory-efficient looped approach.
        
        Args:
            model_distances: Luminosity distances for each galaxy
            distance_uncertainties: Distance uncertainties for each galaxy
            H0 (float): Hubble constant value
            
        Returns:
            array: Log-likelihood contribution for each galaxy, or None if computation fails
        """
        log_P_data_H0_zi_terms = self.xp.zeros(len(self.z_values))
        
        for i in range(len(self.z_values)):
            likelihood = self._compute_single_galaxy_likelihood_looped(
                i, model_distances[i], distance_uncertainties[i], H0
            )
            log_P_data_H0_zi_terms = self._update_array_element(
                log_P_data_H0_zi_terms, i, likelihood
            )
        
        return log_P_data_H0_zi_terms

    def _compute_single_galaxy_likelihood_looped(self, galaxy_idx, model_distance, distance_uncertainty, H0):
        """Compute likelihood for a single galaxy in looped mode.
        
        Args:
            galaxy_idx (int): Index of the galaxy
            model_distance (float): Model luminosity distance for this galaxy
            distance_uncertainty (float): Distance uncertainty for this galaxy
            H0 (float): Hubble constant value
            
        Returns:
            float: Log-likelihood contribution for this galaxy
        """
        mu_z = self.z_values[galaxy_idx]
        sigma_z = self.z_err_values[galaxy_idx]

        # Check if marginalization is needed
        if sigma_z < self.z_err_threshold:
            # Skip marginalization for essentially zero uncertainty
            return self._compute_simple_galaxy_likelihood(model_distance, distance_uncertainty)
        else:
            # Perform redshift marginalization
            return self._marginalize_single_galaxy_redshift_looped(
                mu_z, sigma_z, H0
            )

    def _compute_simple_galaxy_likelihood(self, model_distance, distance_uncertainty):
        """Compute likelihood for a single galaxy without redshift marginalization.
        
        Args:
            model_distance (float): Model luminosity distance
            distance_uncertainty (float): Distance uncertainty
            
        Returns:
            float: Log-likelihood value
        """
        try:
            log_pdf_for_one_galaxy = logpdf_normal_xp(
                self.xp,
                self.dL_gw_samples,
                loc=model_distance,
                scale=distance_uncertainty,
            )
        except Exception:
            return -self.xp.inf

        if self.xp.any(~self.xp.isfinite(log_pdf_for_one_galaxy)):
            return -self.xp.inf
        else:
            reduced_log_pdf = logsumexp_xp(self.xp, log_pdf_for_one_galaxy)
            return reduced_log_pdf - self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))

    def _marginalize_single_galaxy_redshift_looped(self, mu_z, sigma_z, H0):
        """Perform redshift marginalization for a single galaxy using looped quadrature.
        
        Args:
            mu_z (float): Mean redshift
            sigma_z (float): Redshift uncertainty  
            H0 (float): Hubble constant value
            
        Returns:
            float: Marginalized log-likelihood
        """
        # Compute integration bounds
        z_min = self.xp.maximum(mu_z - self.z_sigma_range * sigma_z, 1e-6)
        z_max = mu_z + self.z_sigma_range * sigma_z
        
        # Check if the integration range is meaningful
        if z_max <= z_min:
            # Fallback to point estimate
            model_distance = self._lum_dist_model(mu_z, H0)
            distance_uncertainty = self.xp.maximum((model_distance / self.c_val) * self.sigma_v, 1e-9)
            return self._compute_simple_galaxy_likelihood(model_distance, distance_uncertainty)

        # Gaussian quadrature integration
        return self._perform_looped_quadrature_integration(mu_z, sigma_z, H0)

    def _perform_looped_quadrature_integration(self, mu_z, sigma_z, H0):
        """Perform Gaussian quadrature integration using a loop over quadrature points.
        
        Args:
            mu_z (float): Mean redshift
            sigma_z (float): Redshift uncertainty
            H0 (float): Hubble constant value
            
        Returns:
            float: Integrated log-likelihood
        """
        log_integrand_values = []
        valid_evaluations = 0
        
        for node, weight in zip(self._quad_nodes, self._quad_weights):
            # Transform quadrature point to redshift domain
            z_j = mu_z + self.xp.sqrt(2.0) * sigma_z * node
            
            # Skip negative redshifts
            if z_j <= 0:
                continue
                
            try:
                likelihood_at_point = self._evaluate_likelihood_at_redshift_point(z_j, H0, weight)
                if likelihood_at_point is not None:
                    log_integrand_values.append(likelihood_at_point)
                    valid_evaluations += 1
            except Exception:
                # Skip this quadrature point if evaluation fails
                continue

        # Combine the integration results
        if valid_evaluations > 0 and len(log_integrand_values) > 0:
            log_integrand_array = self.xp.array(log_integrand_values)
            return logsumexp_xp(self.xp, log_integrand_array)
        else:
            # Integration failed - return -inf
            return -self.xp.inf

    def _evaluate_likelihood_at_redshift_point(self, z_j, H0, weight):
        """Evaluate likelihood at a specific redshift point for quadrature integration.
        
        Args:
            z_j (float): Redshift value
            H0 (float): Hubble constant value
            weight (float): Quadrature weight
            
        Returns:
            float: Log of weighted likelihood at this point, or None if evaluation fails
        """
        # Compute luminosity distance for this redshift
        model_d = self._lum_dist_model(z_j, H0)
        
        # Check if model distance is reasonable
        if not self.xp.isfinite(model_d) or model_d <= 0:
            return None
            
        sigma_d = self.xp.maximum((model_d / self.c_val) * self.sigma_v, 1e-9)
        
        # Compute likelihood for this redshift
        log_pdf = logpdf_normal_xp(
            self.xp,
            self.dL_gw_samples,
            loc=model_d,
            scale=sigma_d,
        )
        
        if self.xp.any(~self.xp.isfinite(log_pdf)):
            return None
        
        # Sum over GW samples
        current_logsumexp = logsumexp_xp(self.xp, log_pdf)
        log_likelihood_at_z = current_logsumexp - self.xp.log(self.xp.array(len(self.dL_gw_samples), dtype=self.xp.float64))
        
        # Store log(weight * likelihood) for numerical stability
        log_weight = self.xp.log(weight / self.xp.sqrt(self.xp.pi))
        return log_weight + log_likelihood_at_z

    def _compute_galaxy_weights(self, alpha):
        """Compute galaxy weights based on mass proxy and alpha parameter.
        
        Args:
            alpha (float): Mass bias parameter
            
        Returns:
            array: Normalized weights for each galaxy, or None if computation fails
        """
        if self.xp.isclose(alpha, 0.0):
            # Equal weights when alpha = 0
            return self.xp.full(len(self.mass_proxy_values), 1.0 / len(self.mass_proxy_values))
        else:
            # Mass-weighted based on alpha
            powered = self.mass_proxy_values ** alpha
            if self.xp.any(powered <= 0) or not self.xp.all(self.xp.isfinite(powered)):
                return None
            denom = powered.sum()
            if not self.xp.isfinite(denom) or denom <= 0:
                return None
            return powered / denom

    def _combine_weighted_likelihoods(self, galaxy_log_likelihoods, galaxy_weights):
        """Combine galaxy likelihoods with their weights to compute final likelihood.
        
        Args:
            galaxy_log_likelihoods: Log-likelihood contribution from each galaxy
            galaxy_weights: Weight for each galaxy based on mass proxy
            
        Returns:
            float: Final weighted log-likelihood
        """
        valid_mask = galaxy_weights > 0
        if not self.xp.any(valid_mask):
            return -self.xp.inf
        
        terms_to_sum = self.xp.log(galaxy_weights[valid_mask]) + galaxy_log_likelihoods[valid_mask]
        total_log_likelihood = logsumexp_xp(self.xp, terms_to_sum)

        if not self.xp.isfinite(total_log_likelihood):
            return -self.xp.inf

        return total_log_likelihood

    def _update_array_element(self, array, index, value):
        """Update array element in a backend-agnostic way.
        
        Args:
            array: Array to update
            index: Index to update
            value: New value
            
        Returns:
            Updated array
        """
        if self.backend_name == "jax":
            # JAX arrays are immutable, use at() method
            return array.at[index].set(value)
        else:
            # NumPy arrays are mutable
            array[index] = value
            return array
    
    def get_distance_cache_stats(self):
        """Get distance cache performance statistics.
        
        Returns:
            dict: Cache statistics including hit rate, build times, etc.
        """
        return self._distance_cache.get_statistics()
    
    def log_distance_cache_stats(self):
        """Log distance cache performance statistics."""
        self._distance_cache.log_statistics()
    
    def validate_distance_cache_accuracy(self, max_relative_error=1e-4):
        """Validate distance cache accuracy against direct computation.
        
        Args:
            max_relative_error: Maximum acceptable relative error
            
        Returns:
            bool: True if validation passes
        """
        # Create test points covering the typical range
        test_z_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
        test_h0_values = np.array([50.0, 65.0, 70.0, 75.0, 90.0, 120.0])
        
        return self._distance_cache.validate_accuracy(
            test_z_values, test_h0_values, max_relative_error
        )
    
    def clear_distance_cache(self):
        """Clear the distance cache and reset statistics."""
        self._distance_cache.clear_cache()

def get_log_likelihood_h0(
    requested_backend_str: str,
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
    z_err_threshold=DEFAULT_Z_ERR_THRESHOLD,
    n_quad_points=DEFAULT_QUAD_POINTS,
    z_sigma_range=DEFAULT_Z_MARGINALIZATION_SIGMA_RANGE,
    force_non_vectorized=False,
):
    """Create and configure an H0LogLikelihood instance with the specified backend.
    
    This is the main factory function for creating likelihood objects. It handles
    backend selection, memory optimization decisions, and proper configuration.
    
    Args:
        requested_backend_str (str): The backend to use ("auto", "numpy", "jax")
        dL_gw_samples: Gravitational wave luminosity distance samples
        host_galaxies_z: Redshifts of candidate host galaxies
        host_galaxies_mass_proxy: Mass proxy values for galaxies
        host_galaxies_z_err: Redshift uncertainties for galaxies
        ... (other parameters as before)
        
    Returns:
        H0LogLikelihood: Configured likelihood instance
    """
    # 1. Initialize backend
    backend_config = _initialize_backend(requested_backend_str)
    
    # 2. Analyze data dimensions
    data_dimensions = _analyze_data_dimensions(dL_gw_samples, host_galaxies_z)
    
    # 3. Determine vectorization strategy
    vectorization_config = _determine_vectorization_strategy(
        backend_config, data_dimensions, force_non_vectorized
    )
    
    # 4. Log configuration decisions
    _log_likelihood_configuration(backend_config, data_dimensions, vectorization_config)
    
    # 5. Create and return likelihood instance
    return H0LogLikelihood(
        backend_config.xp_module,
        backend_config.backend_name,
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
        use_vectorized_likelihood=vectorization_config.should_use_vectorized,
        z_err_threshold=z_err_threshold,
        n_quad_points=n_quad_points,
        z_sigma_range=z_sigma_range,
    )

# Supporting data classes for cleaner interfaces
@dataclass
class BackendConfig:
    """Configuration for computational backend."""
    xp_module: object
    backend_name: str
    device_name: str

@dataclass  
class DataDimensions:
    """Analysis of input data dimensions."""
    n_gw_samples: int
    n_hosts: int
    total_elements: int

@dataclass
class VectorizationConfig:
    """Configuration for vectorization strategy."""
    should_use_vectorized: bool
    memory_threshold_elements: float
    force_non_vectorized: bool

def _initialize_backend(requested_backend_str: str) -> BackendConfig:
    """Initialize and configure the computational backend.
    
    Args:
        requested_backend_str: Backend identifier ("auto", "numpy", "jax")
        
    Returns:
        BackendConfig: Backend configuration object
    """
    xp_module, backend_name, device_name = get_xp(requested_backend_str)
    return BackendConfig(
        xp_module=xp_module,
        backend_name=backend_name, 
        device_name=device_name
    )

def _analyze_data_dimensions(dL_gw_samples, host_galaxies_z) -> DataDimensions:
    """Analyze input data dimensions for memory and performance planning.
    
    Args:
        dL_gw_samples: GW distance samples
        host_galaxies_z: Host galaxy redshifts
        
    Returns:
        DataDimensions: Analysis of data sizes
    """
    # Ensure array-like inputs for length calculation
    n_gw_samples = len(np.asarray(dL_gw_samples))
    
    # Handle scalar vs array host redshifts
    host_z_array = np.asarray(host_galaxies_z)
    if host_z_array.ndim == 0:
        n_hosts = 1
    else:
        n_hosts = len(host_z_array)
    
    total_elements = n_gw_samples * n_hosts
    
    return DataDimensions(
        n_gw_samples=n_gw_samples,
        n_hosts=n_hosts,
        total_elements=total_elements
    )

def _determine_vectorization_strategy(
    backend_config: BackendConfig, 
    data_dimensions: DataDimensions,
    force_non_vectorized: bool
) -> VectorizationConfig:
    """Determine optimal vectorization strategy based on backend and data size.
    
    Args:
        backend_config: Backend configuration
        data_dimensions: Data dimension analysis
        force_non_vectorized: Override flag to force looped computation
        
    Returns:
        VectorizationConfig: Vectorization strategy configuration
    """
    memory_threshold_elements = MEMORY_THRESHOLD_BYTES / BYTES_PER_ELEMENT
    
    if backend_config.backend_name == "jax":
        # JAX benefits from vectorization unless explicitly overridden
        should_use_vectorized = not force_non_vectorized
    else:
        # NumPy: use memory-based decision unless overridden
        memory_allows_vectorization = data_dimensions.total_elements <= memory_threshold_elements
        should_use_vectorized = memory_allows_vectorization and not force_non_vectorized
    
    return VectorizationConfig(
        should_use_vectorized=should_use_vectorized,
        memory_threshold_elements=memory_threshold_elements,
        force_non_vectorized=force_non_vectorized
    )

def _log_likelihood_configuration(
    backend_config: BackendConfig,
    data_dimensions: DataDimensions, 
    vectorization_config: VectorizationConfig
):
    """Log the likelihood configuration decisions for debugging and monitoring.
    
    Args:
        backend_config: Backend configuration
        data_dimensions: Data dimensions
        vectorization_config: Vectorization configuration
    """
    if backend_config.backend_name == "jax":
        if vectorization_config.force_non_vectorized:
            logger.info(f"JAX backend FORCED to non-vectorized for testing/debugging purposes.")
        else:
            logger.info(f"Using JAX backend (on {backend_config.device_name}). Will use JAX-optimized likelihood path.")
    else:
        # NumPy backend logging
        memory_gb = MEMORY_THRESHOLD_BYTES / (1024**3)
        
        if vectorization_config.force_non_vectorized:
            logger.info(f"FORCED non-vectorized likelihood for testing/debugging purposes.")
        
        if vectorization_config.should_use_vectorized:
            logger.info(
                f"Using NumPy backend, VECTORIZED likelihood: N_samples ({data_dimensions.n_gw_samples}) * "
                f"N_hosts ({data_dimensions.n_hosts}) = {data_dimensions.total_elements} elements. "
                f"Threshold: {vectorization_config.memory_threshold_elements:.0f} elements ({memory_gb:.0f} GB)."
            )
        else:
            reason = "Forced by force_non_vectorized=True" if vectorization_config.force_non_vectorized else \
                    f"Exceeds threshold of {vectorization_config.memory_threshold_elements:.0f} elements ({memory_gb:.0f} GB)"
            logger.info(
                f"Using NumPy backend, LOOPED likelihood (memory efficient): N_samples ({data_dimensions.n_gw_samples}) * "
                f"N_hosts ({data_dimensions.n_hosts}) = {data_dimensions.total_elements} elements. {reason}."
            )

def run_mcmc_h0(
    log_likelihood_func,
    event_name,
    n_walkers=DEFAULT_MCMC_N_WALKERS,
    n_dim=DEFAULT_MCMC_N_DIM,
    initial_h0_mean=DEFAULT_MCMC_INITIAL_H0_MEAN,
    initial_h0_std=DEFAULT_MCMC_INITIAL_H0_STD,
    alpha_prior_min=DEFAULT_ALPHA_PRIOR_MIN,
    alpha_prior_max=DEFAULT_ALPHA_PRIOR_MAX,
    n_steps=DEFAULT_MCMC_N_STEPS,
    pool=None,
):
    """Run MCMC sampling for H0 and alpha parameters.
    
    This function orchestrates the entire MCMC process: initialization,
    execution, and error handling.
    
    Args:
        log_likelihood_func: The log likelihood function
        event_name: Name of the event for logging
        n_walkers: Number of MCMC walkers
        n_dim: Number of dimensions (2 for H0 and alpha)
        initial_h0_mean: Mean for H0 walker initialization
        initial_h0_std: Standard deviation for H0 initialization
        alpha_prior_min: Lower bound for alpha initialization
        alpha_prior_max: Upper bound for alpha initialization  
        n_steps: Number of MCMC steps
        pool: Optional pool object for parallelization
        
    Returns:
        emcee.EnsembleSampler: The sampler object after running, or None on failure
    """
    logger.info(f"Running MCMC for H0 and alpha on event {event_name} ({n_steps} steps, {n_walkers} walkers)...")
    
    # 1. Initialize walker positions
    initial_positions = _initialize_mcmc_walkers(
        n_walkers, initial_h0_mean, initial_h0_std, alpha_prior_min, alpha_prior_max
    )
    
    # 2. Create and configure sampler
    sampler = _create_mcmc_sampler(n_walkers, n_dim, log_likelihood_func, pool)
    
    # 3. Run MCMC with error handling
    return _execute_mcmc_run(sampler, initial_positions, n_steps, event_name)

def _initialize_mcmc_walkers(n_walkers, h0_mean, h0_std, alpha_min, alpha_max):
    """Initialize walker positions for MCMC sampling.
    
    Args:
        n_walkers: Number of walkers
        h0_mean: Mean H0 value for initialization
        h0_std: Standard deviation for H0 initialization
        alpha_min: Minimum alpha value
        alpha_max: Maximum alpha value
        
    Returns:
        np.array: Initial walker positions (n_walkers, 2)
    """
    # Initialize H0 positions around the mean
    pos_H0 = h0_mean + h0_std * np.random.randn(n_walkers, 1)
    
    # Initialize alpha positions uniformly within prior range
    pos_alpha = np.random.uniform(alpha_min, alpha_max, size=(n_walkers, 1))
    
    # Combine into full parameter space
    return np.hstack((pos_H0, pos_alpha))

def _create_mcmc_sampler(n_walkers, n_dim, log_likelihood_func, pool):
    """Create and configure the MCMC sampler.
    
    Args:
        n_walkers: Number of walkers
        n_dim: Number of dimensions  
        log_likelihood_func: Log likelihood function
        pool: Optional parallelization pool
        
    Returns:
        emcee.EnsembleSampler: Configured sampler
    """
    return emcee.EnsembleSampler(
        n_walkers, 
        n_dim, 
        log_likelihood_func, 
        moves=emcee.moves.StretchMove(),
        pool=pool
    )

def _execute_mcmc_run(sampler, initial_positions, n_steps, event_name):
    """Execute the MCMC run with comprehensive error handling.
    
    Args:
        sampler: Configured MCMC sampler
        initial_positions: Initial walker positions
        n_steps: Number of steps to run
        event_name: Event name for logging
        
    Returns:
        emcee.EnsembleSampler: Sampler after running, or None on failure
    """
    try:
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        logger.info(f"MCMC run completed successfully for {event_name}.")
        return sampler
        
    except ValueError as ve:
        logger.error(f"❌ ValueError during MCMC for {event_name}: {ve}")
        logger.error("  This can happen if the likelihood consistently returns -inf. Check priors or input data.")
        return None
        
    except Exception as e:
        logger.exception(f"❌ An unexpected error occurred during MCMC for {event_name}: {e}")
        return None

def process_mcmc_samples(sampler, event_name, burnin=DEFAULT_MCMC_BURNIN, thin_by=DEFAULT_MCMC_THIN_BY, n_dim=DEFAULT_MCMC_N_DIM):
    """Process MCMC samples by applying burn-in, thinning, and validation.
    
    Args:
        sampler: MCMC sampler object after running
        event_name: Name of the event for logging  
        burnin: Number of burn-in steps to discard
        thin_by: Factor to thin the samples by
        n_dim: Expected number of dimensions in the chain
        
    Returns:
        np.array: Processed samples, or None if processing fails
    """
    if sampler is None:
        logger.warning(f"⚠️ Sampler object is None for {event_name}. Cannot process MCMC samples.")
        return None

    logger.info(f"Processing MCMC samples for {event_name} (burn-in: {burnin}, thin: {thin_by})...")
    
    try:
        # 1. Extract and validate raw samples
        raw_samples = _extract_raw_samples(sampler, burnin, thin_by)
        if raw_samples is None:
            return None
            
        # 2. Validate sample dimensions and content
        if not _validate_sample_dimensions(raw_samples, event_name, sampler):
            return None
            
        # 3. Format samples according to expected dimensions
        return _format_processed_samples(raw_samples, n_dim, event_name)
        
    except Exception as e:
        logger.exception(f"❌ Error processing MCMC chain for {event_name}: {e}")
        return None

def _extract_raw_samples(sampler, burnin, thin_by):
    """Extract raw samples from sampler with burn-in and thinning.
    
    Args:
        sampler: MCMC sampler object
        burnin: Number of burn-in steps
        thin_by: Thinning factor
        
    Returns:
        np.array: Raw flattened samples, or None if extraction fails
    """
    # Extract samples: shape (n_steps, n_walkers, n_dim) -> (n_samples_total, n_dim)
    flat_samples = sampler.get_chain(discard=burnin, thin=thin_by, flat=True)
    return flat_samples

def _validate_sample_dimensions(samples, event_name, sampler):
    """Validate that extracted samples have reasonable dimensions.
    
    Args:
        samples: Extracted samples array
        event_name: Event name for logging
        sampler: Original sampler for diagnostics
        
    Returns:
        bool: True if samples are valid, False otherwise
    """
    if samples.shape[0] == 0:
        original_length = sampler.get_chain().shape[0]
        logger.warning(
            f"⚠️ MCMC for {event_name} resulted in NO valid samples after burn-in and thinning."
        )
        logger.warning(f"  Original chain length before discard: {original_length}")
        return False
    return True

def _format_processed_samples(samples, expected_n_dim, event_name):
    """Format processed samples according to expected dimensions.
    
    Args:
        samples: Raw processed samples
        expected_n_dim: Expected number of dimensions
        event_name: Event name for logging
        
    Returns:
        np.array: Properly formatted samples, or None if formatting fails
    """
    # Handle different dimensional cases
    if expected_n_dim == 1 and samples.ndim == 2 and samples.shape[1] == 1:
        # Single parameter case - extract the column
        processed_samples = samples[:, 0]
    elif expected_n_dim > 1 and samples.ndim == 2 and samples.shape[1] == expected_n_dim:
        # Multi-dimensional case - keep as is
        logger.info(f"  Note: MCMC had {expected_n_dim} dimensions. Returning all dimensions after processing.")
        processed_samples = samples
    elif samples.ndim == 1 and expected_n_dim == 1:
        # Already 1D for single parameter
        processed_samples = samples
    else:
        # Unexpected shape
        logger.warning(
            f"⚠️ Unexpected shape for flat_samples: {samples.shape}. Expected ({'*', expected_n_dim}). "
            f"Cannot safely extract parameters."
        )
        return None

    logger.info(f"  Successfully processed MCMC samples for {event_name}. Number of samples: {len(processed_samples)}.")
    return processed_samples

def profile_likelihood_call(log_like_func_to_profile, h0_values_to_test, n_calls=10):
    """Helper function to profile the likelihood call multiple times."""
    logger.info(f"Profiling {log_like_func_to_profile.__class__.__name__} ({'vectorized' if getattr(log_like_func_to_profile, 'use_vectorized_likelihood', False) else 'looped'})...")
    for h0_val in h0_values_to_test:
        for _ in range(n_calls):
            _ = log_like_func_to_profile([h0_val]) # Call the likelihood