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

            # Hardware detection
            if _is_apple_silicon_metal_available():
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
                # JAX on CPU
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


def logpdf_normal_xp(xp, x, loc, scale):
    """
    Computes the log of the probability density function for a normal distribution.
    Works with either NumPy or JAX backend.
    Ensures scale is positive to avoid issues.
    """
    if xp.__name__ == "jax.numpy":
        from jax.scipy.stats import norm as jax_norm
        # JAX's norm.logpdf can handle scale=0 if x==loc, but to be safe and
        # consistent with typical stats usage, we ensure scale > 0 or return -inf.
        # However, JAX's behavior with non-finite scale or loc might differ from scipy.
        # Smallest positive float64: np.finfo(np.float64).tiny
        # Using a slightly larger epsilon for safety.
        epsilon = 1e-9 # Was 1e-15, but sigma_d_val_for_hosts used 1e-9
        scale = xp.maximum(scale, epsilon)
        return jax_norm.logpdf(x, loc=loc, scale=scale)
    elif xp.__name__ == "numpy":
        from scipy.stats import norm as scipy_norm
        # Scipy's norm.logpdf returns -inf if scale is <=0, which is desired.
        # It also handles NaNs in inputs gracefully (returns NaN).
        # We ensure scale is at least a very small positive number to avoid potential issues
        # if the input scale could be zero or negative due to calculations.
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

__all__ = [
    "get_xp",
    "BackendNotAvailableError",
    "logpdf_normal_xp",
    "logsumexp_xp", # Add this
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
