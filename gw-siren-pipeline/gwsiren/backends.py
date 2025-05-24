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
        # Attempt to use the Metal backend. This usually involves checking devices.
        # jax.devices('metal') would raise an error if Metal is not available/configured.
        # However, a more robust check is to see if 'METAL' is in the platform string of any device.
        # JAX might also default to Metal on M-series Macs if available.
        for device in jax.devices():
            if "METAL" in device.platform.upper() or "GPU" in device.device_kind.upper(): # A bit more general for Apple GPUs
                 # Check if we can actually perform a computation
                try:
                    # Test with float32 for device availability check
                    _ = jax.device_put(jax.numpy.array([1.0], dtype=jax.numpy.float32), device=device).block_until_ready()
                    return True
                except Exception:
                    logger.debug(f"Found Metal device ({device.platform}), but computation test failed.")
                    return False # Found device but it's not usable
        return False
    except Exception as e:
        logger.debug(f"JAX Metal check failed: {e}")
        return False

def _is_nvidia_cuda_available():
    """Check if JAX is installed with CUDA support for NVIDIA GPUs."""
    try:
        import jax
        # jax.devices('cuda') or jax.devices('gpu') would error out if not available.
        # Checking for 'CUDA' or 'NVIDIA' in platform string is safer.
        for device in jax.devices():
            if "CUDA" in device.platform.upper() or "NVIDIA" in device.platform.upper():
                # Check if we can actually perform a computation
                try:
                    # Test with float32 for device availability check
                    _ = jax.device_put(jax.numpy.array([1.0], dtype=jax.numpy.float32), device=device).block_until_ready()
                    return True
                except Exception:
                    logger.debug(f"Found CUDA device ({device.platform}), but computation test failed.")
                    return False # Found device but it's not usable
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
    device_name = "cpu"  # Default device

    # --- JAX Attempt ---
    if requested_backend in ["auto", "jax"]:
        try:
            jax_module = importlib.import_module("jax")
            jnp_module = importlib.import_module("jax.numpy")
            logger.info("JAX found.")

            # Configure JAX for double precision (float64)
            jax_module.config.update("jax_enable_x64", True)
            logger.info("JAX double precision (x64) enabled.")

            # Check if JAX_PLATFORM_NAME is set to cpu first
            if os.environ.get("JAX_PLATFORM_NAME") == "cpu":
                # Force CPU usage, skip GPU detection
                xp = jnp_module
                backend_name = "jax"
                device_name = "cpu"
                logger.info("JAX using CPU (forced by JAX_PLATFORM_NAME=cpu).")
            elif _is_apple_silicon_metal_available():
                device_name = "metal"
                xp = jnp_module
                backend_name = "jax"
                logger.info("JAX using Apple Silicon (Metal) GPU.")
            elif _is_nvidia_cuda_available():
                device_name = "cuda"
                xp = jnp_module
                backend_name = "jax"
                logger.info("JAX using NVIDIA (CUDA) GPU.")
            else:
                # JAX on CPU (fallback)
                # Verify JAX CPU backend is functional
                try:
                    cpu_devices = jax_module.devices("cpu")
                    if not cpu_devices:
                         raise BackendNotAvailableError("JAX found but no CPU devices reported.")
                    # Test a simple computation on JAX CPU
                    # Test with float32 for device availability check
                    _ = jax_module.device_put(jnp_module.array([1.0], dtype=jnp_module.float32), device=cpu_devices[0]).block_until_ready()
                    xp = jnp_module
                    backend_name = "jax"
                    device_name = "cpu" # Explicitly CPU
                    logger.info("JAX using CPU.")
                except Exception as e:
                    logger.warning(f"JAX CPU backend test failed: {e}")
                    if requested_backend == "jax":
                        raise BackendNotAvailableError(f"JAX CPU backend initialization failed: {e}")
                    # Fallback to NumPy will be handled below if "auto"

        except ImportError:
            logger.info("JAX not found.")
            if requested_backend == "jax":
                raise BackendNotAvailableError("JAX backend was requested, but JAX is not installed.")
        except BackendNotAvailableError as e: # Catch specific JAX init errors
             if requested_backend == "jax":
                raise e # Re-raise if JAX was explicitly requested
             logger.warning(f"JAX initialization failed: {e}. Will try NumPy for 'auto' mode.")
        except Exception as e: # Catch any other JAX related errors
            logger.warning(f"An unexpected error occurred during JAX setup: {e}")
            if requested_backend == "jax":
                raise BackendNotAvailableError(f"JAX initialization failed unexpectedly: {e}")
            # Fallback to NumPy for "auto"

    # --- NumPy Attempt or Fallback ---
    if xp is None: # If JAX wasn't successfully initialized or "numpy" was requested
        if requested_backend == "jax" and backend_name is None:
            # This case should ideally be caught by specific errors above,
            # but as a safeguard: if JAX was requested and not set up, error out.
            raise BackendNotAvailableError("JAX backend was requested, but failed to initialize and no fallback was specified.")

        try:
            np_module = importlib.import_module("numpy")
            xp = np_module
            backend_name = "numpy"
            device_name = "cpu" # NumPy always runs on CPU
            logger.info("Using NumPy backend on CPU.")
            if requested_backend == "auto" and importlib.util.find_spec("jax") is not None:
                 logger.warning(
                    "JAX is installed, but 'auto' mode failed to initialize a JAX backend. "
                    "Falling back to NumPy. Check JAX installation and GPU drivers if JAX/GPU was expected."
                )

        except ImportError:
            logger.error("NumPy not found. This is a critical error.")
            # This is a very unlikely scenario in a scientific Python environment
            raise BackendNotAvailableError("NumPy is not installed, which is required.")

    if xp is None or backend_name is None:
        # Should be caught by earlier specific errors.
        raise BackendNotAvailableError(f"Could not initialize any backend for request '{requested_backend}'.")

    _BACKEND_INFO = {
        'module': xp,
        'name': backend_name,
        'device': device_name,
        'requested_backend_mode': requested_backend # Store the mode that led to this config
    }
    return xp, backend_name, device_name

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
