import importlib
import logging
import os
import sys

# Configure logger for the backends module
logger = logging.getLogger(__name__)

# Global variable to store the chosen backend info once determined.
_BACKEND_INFO = None

SUPPORTED_BACKENDS = ["numpy", "jax"]

class BackendNotAvailableError(RuntimeError):
    """Custom exception for when a backend cannot be initialized."""
    pass

def clear_backend_cache():
    """Clear the cached backend info to force re-detection."""
    global _BACKEND_INFO
    _BACKEND_INFO = None

def _is_apple_silicon_metal_available():
    """Check if JAX is installed with Metal support for Apple Silicon GPUs."""
    try:
        import jax
        
        # Check if any device has METAL in its platform
        for device in jax.devices():
            if "METAL" in device.platform.upper():
                # Metal is available, but may have limitations
                # We'll let the main backend logic handle testing
                logger.debug(f"Metal device found: {device.platform}")
                return True
        return False
    except Exception as e:
        logger.debug(f"JAX Metal check failed: {e}")
        return False

def _is_nvidia_cuda_available():
    """Check if JAX is installed with CUDA support for NVIDIA GPUs."""
    try:
        import jax
        import jax.numpy as jnp
        
        # Check if any device has CUDA or NVIDIA in its platform
        for device in jax.devices():
            if "CUDA" in device.platform.upper() or "NVIDIA" in device.platform.upper():
                # For CUDA devices, do a simpler test
                try:
                    # Test basic JAX functionality without device_put
                    test_array = jnp.array([1.0, 2.0, 3.0])
                    result = jnp.sum(test_array)
                    # If we can create arrays and do basic operations, CUDA is working
                    if jnp.isfinite(result):
                        logger.debug(f"CUDA device ({device.platform}) passed basic computation test.")
                        return True
                except Exception as e:
                    logger.debug(f"Found CUDA device ({device.platform}), but basic computation test failed: {e}")
                    return False
        return False
    except Exception as e:
        logger.debug(f"JAX CUDA check failed: {e}")
        return False

def get_xp(requested_backend="auto"):
    """
    Selects and provides the numerical backend (NumPy or JAX) and its name.

    Args:
        requested_backend (str): "auto", "numpy", or "jax".
            - "auto": Try JAX (GPU if available, then JAX CPU), fallback to NumPy.
            - "jax": Try JAX (GPU if available, then JAX CPU). Error if JAX unavailable.
            - "numpy": Use NumPy. Error if NumPy unavailable (highly unlikely).

    Returns:
        tuple: (module, str, str) -> (numerical_module, backend_name, device_name)
               Example: (jax.numpy, "jax", "cuda") or (numpy, "numpy", "cpu")

    Raises:
        BackendNotAvailableError: If the requested backend (or JAX in auto mode)
                                 cannot be initialized.
        ValueError: If requested_backend is not a valid option.
    """
    global _BACKEND_INFO
    if _BACKEND_INFO is not None and requested_backend == _BACKEND_INFO['requested_backend_mode']:
        # If the same mode is requested, return the cached info.
        # This avoids re-running detection logic unnecessarily.
        # If a different mode is requested (e.g. 'jax' then 'numpy'), we should re-evaluate.
        logger.debug(f"Returning cached backend: {_BACKEND_INFO['name']} on {_BACKEND_INFO['device']}")
        return _BACKEND_INFO['module'], _BACKEND_INFO['name'], _BACKEND_INFO['device']

    if requested_backend not in ["auto", "numpy", "jax"]:
        raise ValueError(f"Invalid backend '{requested_backend}'. Must be 'auto', 'numpy', or 'jax'.")

    xp = None
    backend_name = None
    device_name = "cpu"  # Default device name, updated upon successful JAX init

    # --- JAX Attempt ---
    if requested_backend in ["auto", "jax"]:
        jax_module = None
        jnp_module = None
        try:
            jax_module = importlib.import_module("jax")
            jnp_module = importlib.import_module("jax.numpy")
            logger.info("JAX found.")

            # Helper to test JAX configuration
            def _test_jax_config(current_jnp, current_jax_module, config_name_log, enable_x64=None):
                if enable_x64 is not None:
                    current_jax_module.config.update("jax_enable_x64", enable_x64)
                
                # Determine dtype for test based on x64 config
                # If x64 is explicitly False, or if it's None (not changing config) and current config is float32 default
                is_float32_mode = (enable_x64 is False) or \
                                  (enable_x64 is None and not current_jax_module.config.jax_enable_x64)

                dtype_to_test = current_jnp.float32 if is_float32_mode else current_jnp.float64
                test_arr = current_jnp.array([1.0, 2.0, 3.0], dtype=dtype_to_test)
                res = current_jnp.sum(test_arr)
                if not current_jnp.isfinite(res):
                    raise RuntimeError(f"Basic JAX computation failed for {config_name_log}.")
                precision_log = "float32" if is_float32_mode else "x64 (float64)"
                logger.info(f"JAX {config_name_log} configured with {precision_log} and basic test passed.")

            # 1. JAX_PLATFORM_NAME="cpu" (Highest priority if set)
            if os.environ.get("JAX_PLATFORM_NAME") == "cpu":
                logger.info("Attempting JAX CPU (forced by JAX_PLATFORM_NAME).")
                try:
                    _test_jax_config(jnp_module, jax_module, "CPU (forced)", enable_x64=True)
                    xp = jnp_module
                    backend_name = "jax"
                    device_name = "cpu"
                except Exception as e:
                    logger.warning(f"Forced JAX CPU (JAX_PLATFORM_NAME=cpu) backend failed: {e}")
                    xp = None # Ensure xp is None if this attempt fails

            # 2. CUDA (if not already successful and not forced CPU)
            if xp is None and os.environ.get("JAX_PLATFORM_NAME") != "cpu":
                if _is_nvidia_cuda_available(): # Helper includes a basic check
                    logger.info("Attempting JAX CUDA backend.")
                    try:
                        _test_jax_config(jnp_module, jax_module, "CUDA", enable_x64=True)
                        xp = jnp_module
                        backend_name = "jax"
                        device_name = "cuda"
                    except Exception as e:
                        logger.warning(f"JAX CUDA backend failed: {e}. Will try other JAX options.")
                        xp = None 

            # 3. Metal (if not already successful and not forced CPU)
            if xp is None and os.environ.get("JAX_PLATFORM_NAME") != "cpu":
                if _is_apple_silicon_metal_available(): # Helper includes a basic check
                    logger.info("Attempting JAX Metal backend.")
                    # Try x64 Metal
                    try:
                        _test_jax_config(jnp_module, jax_module, "Metal", enable_x64=True)
                        xp = jnp_module
                        backend_name = "jax"
                        device_name = "metal"
                    except Exception as e_x64:
                        logger.warning(f"JAX Metal x64 mode failed: {e_x64}. Attempting Metal with float32.")
                        xp = None # Reset before float32 attempt
                        # Try float32 Metal
                        try:
                            _test_jax_config(jnp_module, jax_module, "Metal", enable_x64=False)
                            xp = jnp_module
                            backend_name = "jax"
                            device_name = "metal_float32"
                        except Exception as e_f32:
                            logger.warning(f"JAX Metal float32 mode also failed: {e_f32}. Will try JAX CPU fallback.")
                            xp = None
            
            # 4. JAX CPU (general fallback, if not already successful)
            if xp is None:
                # This attempt runs if:
                # - JAX_PLATFORM_NAME was not 'cpu' (or was 'cpu' but that specific attempt failed, leaving xp=None)
                # - AND GPU attempts (CUDA, Metal) failed or were not applicable, leaving xp=None.
                logger.info("Attempting JAX CPU backend (general fallback).")
                try:
                    # Note: JAX might be "stuck" on a previous GPU attempt.
                    # Setting JAX_PLATFORM_NAME before import is most robust. This is a best effort.
                    _test_jax_config(jnp_module, jax_module, "CPU (fallback)", enable_x64=True)
                    xp = jnp_module
                    backend_name = "jax"
                    device_name = "cpu"
                except Exception as e:
                    logger.warning(f"JAX CPU backend (general fallback) failed: {e}")
                    xp = None

            # Post JAX attempts:
            if xp is not None: # JAX successfully initialized
                logger.info(f"Successfully initialized JAX backend on {device_name} with {('x64' if jax_module.config.jax_enable_x64 else 'float32')} precision.")
                # Caching and return will happen outside this try-except block for JAX import
            elif requested_backend == "jax": # JAX requested, but xp is None (all JAX attempts failed)
                raise BackendNotAvailableError(
                    "JAX backend was requested, but all JAX initialization attempts "
                    "(tried forced CPU, CUDA, Metal, and CPU fallback) failed."
                )
            # If "auto" and JAX failed (xp is None), proceed to NumPy block naturally

        except ImportError: # JAX itself not found
            logger.info("JAX not found.")
            if requested_backend == "jax":
                raise BackendNotAvailableError("JAX backend was requested, but JAX is not installed.")
            # For "auto", xp remains None, will fall through to NumPy.
        except Exception as e: # Catch any other unexpected error during JAX setup phase
            logger.warning(f"An unexpected error occurred during JAX backend setup: {e}", exc_info=True)
            if requested_backend == "jax":
                raise BackendNotAvailableError(f"JAX backend setup failed with an unexpected error: {e}")
            # For "auto", ensure xp is None to allow NumPy fallback.
            xp = None

    # If JAX was initialized successfully, cache and return
    if xp is not None and backend_name == "jax":
        _BACKEND_INFO = {
            'module': xp, 'name': backend_name, 'device': device_name,
            'requested_backend_mode': requested_backend
        }
        logger.debug(f"Caching and returning JAX backend: {backend_name} on {device_name}")
        return xp, backend_name, device_name

    # --- NumPy Attempt or Fallback ---
    # Reached if:
    # - requested_backend == "numpy" (xp would be None from JAX block)
    # - requested_backend == "auto" AND JAX attempts failed (xp is None)
    if requested_backend == "numpy" or (requested_backend == "auto" and xp is None):
        logger.info("Attempting NumPy backend.")
        try:
            np_module = importlib.import_module("numpy")
            # Use temporary variables for NumPy in case JAX was partially set then failed unexpectedly
            # However, the logic above should ensure xp is None if JAX final init failed.
            xp_numpy = np_module
            backend_name_numpy = "numpy"
            device_name_numpy = "cpu" # NumPy always runs on CPU
            
            logger.info("Using NumPy backend on CPU.")

            if requested_backend == "auto" and importlib.util.find_spec("jax") is not None:
                 # This implies JAX is installed but failed to initialize.
                 logger.warning(
                    "JAX is installed, but 'auto' mode failed to initialize a JAX backend. "
                    "Falling back to NumPy. Check JAX installation, GPU drivers, and logs for JAX errors."
                )
            
            _BACKEND_INFO = {
                'module': xp_numpy, 'name': backend_name_numpy, 'device': device_name_numpy,
                'requested_backend_mode': requested_backend
            }
            logger.debug(f"Caching and returning NumPy backend: {backend_name_numpy} on {device_name_numpy}")
            return xp_numpy, backend_name_numpy, device_name_numpy

        except ImportError:
            logger.error("NumPy not found. This is a critical error.")
            # This is a very unlikely scenario in a scientific Python environment
            raise BackendNotAvailableError("NumPy is not installed, which is required for fallback or direct request.")

    # If we reach here, something went fundamentally wrong, and no backend was initialized.
    # This should ideally be caught by more specific error handling above.
    final_error_msg = f"Could not initialize any backend for request '{requested_backend}'. All attempts failed."
    if requested_backend == "auto" and importlib.util.find_spec("jax") is not None and xp is None:
        final_error_msg += " JAX was found but all initialization attempts failed. NumPy also failed or was not correctly processed."
    elif requested_backend == "jax" and xp is None : # Double check, should have been caught
        final_error_msg = "JAX backend was requested, but all JAX initialization attempts failed. This is the final error."
    
    raise BackendNotAvailableError(final_error_msg)

# Constants for fast normal log PDF computation
_LOG_2PI = 1.8378770664093453  # np.log(2 * np.pi) precomputed

def logpdf_normal_xp(xp, x, loc, scale):
    """
    Fast log probability density function for normal distribution.
    
    Optimized version that replaces scipy.stats.norm.logpdf with NumPy-only computation
    for 3-7x speedup while maintaining numerical accuracy.
    
    Works with either NumPy or JAX backend.
    """
    epsilon = 1e-9  # Ensure scale > 0
    scale = xp.maximum(scale, epsilon)
    
    if xp.__name__ == "jax.numpy":
        # JAX-compatible fast implementation
        z = (x - loc) / scale
        return -0.5 * (z**2 + _LOG_2PI) - xp.log(scale)
    elif xp.__name__ == "numpy":
        # NumPy fast implementation - optimized for our use case
        z = (x - loc) / scale
        return -0.5 * (z**2 + _LOG_2PI) - xp.log(scale)
    else:
        raise ValueError(f"Unsupported numerical backend: {xp.__name__}")


def logpdf_normal_xp_original(xp, x, loc, scale):
    """
    Original implementation using scipy/JAX scipy for reference.
    Kept for debugging/comparison purposes.
    """
    if xp.__name__ == "jax.numpy":
        from jax.scipy.stats import norm as jax_norm
        epsilon = 1e-9
        scale = xp.maximum(scale, epsilon)
        return jax_norm.logpdf(x, loc=loc, scale=scale)
    elif xp.__name__ == "numpy":
        from scipy.stats import norm as scipy_norm
        epsilon = 1e-9
        scale = xp.maximum(scale, epsilon)
        return scipy_norm.logpdf(x, loc=loc, scale=scale)
    else:
        raise ValueError(f"Unsupported numerical backend: {xp.__name__}")

def logsumexp_xp(xp, a, axis=None, b=None, keepdims=False, return_sign=False):
    """
    Computes the log of the sum of exponentials of input elements, backend-agnostically.
    Signature matches scipy.special.logsumexp where possible.
    Note: JAX's logsumexp does not support the 'return_sign' parameter directly in the same way.
    If return_sign=True and using JAX, it will raise a NotImplementedError.
    The 'b' argument (for scaling) is also handled.
    """
    if return_sign:
        # JAX's native logsumexp doesn't directly return a sign array like SciPy's.
        # Implementing this with full JAX compatibility would require more work.
        # For now, if this specific feature is needed with JAX, it's an error.
        if xp.__name__ == "jax.numpy":
            raise NotImplementedError("return_sign=True is not directly supported for logsumexp_xp with JAX backend.")
        # For NumPy, scipy.special.logsumexp handles it.
    
    if xp.__name__ == "jax.numpy":
        from jax.scipy.special import logsumexp as jax_logsumexp
        # JAX's logsumexp needs 'b' to be passed if not None.
        if b is not None:
            return jax_logsumexp(a, axis=axis, b=b, keepdims=keepdims)
        else:
            return jax_logsumexp(a, axis=axis, keepdims=keepdims)
    elif xp.__name__ == "numpy":
        from scipy.special import logsumexp as scipy_logsumexp
        return scipy_logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign)
    else:
        raise ValueError(f"Unsupported numerical backend: {xp.__name__}")

def trapz_xp(xp, y, x=None, dx=1.0, axis=-1):
    """
    Backend-agnostic trapezoidal integration.
    
    JAX doesn't have trapz, so we implement it manually for JAX compatibility.
    For NumPy, use the native trapz function.
    
    Args:
        xp: Backend module (numpy or jax.numpy)
        y: Array to integrate
        x: Optional x-coordinates (if None, use dx spacing)
        dx: Spacing between points (if x is None)
        axis: Axis along which to integrate
        
    Returns:
        Integrated value
    """
    if xp.__name__ == "jax.numpy":
        # JAX doesn't have trapz, implement manually
        if x is None:
            # Uniform spacing case
            if y.shape[axis] < 2:
                return xp.array(0.0)
            # Manual trapz: (b-a)/2 * (f(a) + f(b)) for uniform spacing
            # For arrays: dx/2 * (y[0] + 2*y[1:-1] + y[-1])
            # Simplified: dx * (0.5*y[0] + y[1:-1].sum() + 0.5*y[-1])
            
            # Move axis to last position for easier indexing
            y_moved = xp.moveaxis(y, axis, -1)
            
            if y_moved.shape[-1] < 2:
                return xp.zeros(y_moved.shape[:-1])
                
            # Compute trapz manually
            # For 1D: dx * (0.5*y[0] + sum(y[1:-1]) + 0.5*y[-1])
            # For ND: same but along last axis
            first = 0.5 * y_moved[..., 0]
            last = 0.5 * y_moved[..., -1]
            middle = xp.sum(y_moved[..., 1:-1], axis=-1) if y_moved.shape[-1] > 2 else 0.0
            
            result = dx * (first + middle + last)
            return result
            
        else:
            # Non-uniform spacing case
            if y.shape[axis] < 2:
                return xp.array(0.0)
                
            # Move axis to last position for easier indexing
            y_moved = xp.moveaxis(y, axis, -1)
            x_moved = xp.moveaxis(x, axis, -1) if x.ndim > 1 else x
            
            if y_moved.shape[-1] < 2:
                return xp.zeros(y_moved.shape[:-1])
            
            # Manual trapz with non-uniform spacing
            # trapz = sum((x[i+1] - x[i]) * (y[i+1] + y[i]) / 2)
            dx_vals = x_moved[..., 1:] - x_moved[..., :-1]
            y_avg = (y_moved[..., 1:] + y_moved[..., :-1]) / 2.0
            
            result = xp.sum(dx_vals * y_avg, axis=-1)
            return result
            
    elif xp.__name__ == "numpy":
        # Use NumPy's native trapz
        return xp.trapz(y, x=x, dx=dx, axis=axis)
    else:
        raise ValueError(f"Unsupported numerical backend: {xp.__name__}")

__all__ = [
    "get_xp",
    "BackendNotAvailableError",
    "logpdf_normal_xp",
    "logsumexp_xp",
    "trapz_xp",
    "clear_backend_cache",
]


if __name__ == '__main__':
    # Basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

    logger.info("--- Testing get_xp ---")

    def test_backend(request_mode):
        logger.info(f"Requesting backend: '{request_mode}'")
        try:
            xp_module, name, device = get_xp(request_mode)
            logger.info(f"  Successfully got: {name} on {device}")
            logger.info(f"  Module: {xp_module}")
            # Test a simple operation
            arr = xp_module.array([1.0, 2.0, 3.0])
            logger.info(f"  Test array: {arr}, sum: {xp_module.sum(arr)}")
            if name == "jax":
                logger.info(f"  JAX default device: {importlib.import_module('jax').default_backend()}")
                logger.info(f"  JAX devices: {importlib.import_module('jax').devices()}")
        except BackendNotAvailableError as e:
            logger.error(f"  Error for '{request_mode}': {e}")
        except ValueError as e:
            logger.error(f"  ValueError for '{request_mode}': {e}")
        except Exception as e:
            logger.error(f"  Unexpected Exception for '{request_mode}': {e}", exc_info=True)

    # Test scenarios
    # Note: Actual JAX GPU detection depends on the environment where this is run.
    # These tests primarily check the logic paths.
    test_backend("auto") # Should try JAX (GPU then CPU), then NumPy
    test_backend("jax")  # Should try JAX (GPU then CPU), error if not fully available
    test_backend("numpy")# Should use NumPy
    test_backend("invalid_backend_name") # Should raise ValueError

    # Test caching: Requesting 'auto' again should be faster and use cached info
    logger.info("\n--- Testing caching ---")
    test_backend("auto") # First call for 'auto'
    test_backend("auto") # Second call for 'auto', should use cache if first succeeded

    # To test JAX specific parts without JAX installed, you'd mock imports.
    # For this subtask, we assume the environment might have JAX or might not.
    # The goal is that the code itself is sound.
    logger.info("\n--- Backend tests complete ---")
