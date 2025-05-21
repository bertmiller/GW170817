import logging
import numpy

logger = logging.getLogger(__name__)

def get_xp(preferred_backend: str = "auto"):
    """
    Selects the numerical backend (NumPy or JAX) based on user preference and hardware.

    Args:
        preferred_backend (str): "auto", "numpy", or "jax".
                                 "auto" prioritizes JAX on GPU, then NumPy.
                                 "numpy" forces NumPy.
                                 "jax" forces JAX (CPU or GPU).

    Returns:
        tuple: (module, str) - The numerical library (NumPy or JAX.numpy) and its name.
    """
    if preferred_backend == "numpy":
        logger.info("Using NumPy backend as explicitly requested.")
        return numpy, "numpy"

    if preferred_backend in ("jax", "auto"):
        try:
            import jax
            import jax.numpy as jnp
            logger.info("Successfully imported JAX.")

            # Check for GPU presence
            try:
                # Updated check for JAX versions 0.4.0 and above
                gpu_devices = [d for d in jax.devices() if d.device_kind.lower().startswith('gpu') or d.platform.lower() == 'gpu']
                # For older JAX or other accelerators like Metal on macOS
                if not gpu_devices:
                    gpu_devices = [d for d in jax.devices() if d.platform.lower() in ['gpu', 'metal'] or 'gpu' in d.device_kind.lower()]
                is_gpu_present = len(gpu_devices) > 0
                if is_gpu_present:
                    logger.info(f"JAX detected GPU devices: {gpu_devices}")
                else:
                    logger.info("JAX: No GPU devices detected.")
            except Exception as e:
                logger.warning(f"Could not reliably determine JAX device kind: {e}. Assuming CPU.")
                is_gpu_present = False


            if preferred_backend == "jax":
                if is_gpu_present:
                    logger.info("Using JAX backend on GPU as explicitly requested.")
                else:
                    logger.info("Using JAX backend on CPU as explicitly requested (GPU not found or not usable).")
                return jnp, "jax"
            
            # preferred_backend == "auto"
            if is_gpu_present:
                logger.info("Auto backend selection: Using JAX on GPU.")
                return jnp, "jax"
            else:
                logger.info("Auto backend selection: JAX is available, but no GPU detected. Falling back to NumPy.")
                return numpy, "numpy"

        except ImportError:
            if preferred_backend == "jax":
                logger.warning("JAX backend was requested, but JAX installation not found. Falling back to NumPy.")
            else: # auto
                logger.info("Auto backend selection: JAX installation not found. Falling back to NumPy.")
            return numpy, "numpy"
        except Exception as e: # Catch other JAX related errors
            logger.error(f"An unexpected error occurred while initializing JAX: {e}. Falling back to NumPy.")
            return numpy, "numpy"

    else:
        logger.warning(f"Invalid preferred_backend '{preferred_backend}'. Falling back to NumPy.")
        return numpy, "numpy"

if __name__ == '__main__':
    # Basic test and demonstration
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing with preferred_backend='auto'")
    xp_module, backend_name = get_xp("auto")
    logger.info(f"Selected backend: {backend_name}, module: {xp_module}")
    arr = xp_module.array([1,2,3])
    logger.info(f"Test array created with {backend_name}: {arr}")

    logger.info("\nTesting with preferred_backend='numpy'")
    xp_module, backend_name = get_xp("numpy")
    logger.info(f"Selected backend: {backend_name}, module: {xp_module}")
    arr = xp_module.array([1,2,3])
    logger.info(f"Test array created with {backend_name}: {arr}")
    
    logger.info("\nTesting with preferred_backend='jax'")
    xp_module, backend_name = get_xp("jax")
    logger.info(f"Selected backend: {backend_name}, module: {xp_module}")
    # JAX arrays might print differently or have different types
    arr = xp_module.array([1,2,3])
    logger.info(f"Test array created with {backend_name}: {arr}, type: {type(arr)}")

    # Test invalid backend
    logger.info("\nTesting with preferred_backend='invalid_backend_name'")
    xp_module, backend_name = get_xp("invalid_backend_name")
    logger.info(f"Selected backend: {backend_name}, module: {xp_module}")
    arr = xp_module.array([1,2,3])
    logger.info(f"Test array created with {backend_name}: {arr}")

    # Test JAX import error (manual simulation for testing this path)
    # To truly test this, you'd need to run in an env without JAX.
    # We can simulate by temporarily making jax un-importable if we could manipulate sys.modules
    # For now, we trust the try-except block.
    logger.info("\nNote: JAX import error path is harder to simulate directly here without altering environment.")
    logger.info("Assuming JAX is importable for this demonstration run if installed.")

    # Example of how to use it:
    # xp, _ = get_xp(preferred_backend="auto")
    # a = xp.arange(10)
    # print(a)

def log_gaussian(xp: object, x, mu, sigma):
    """
    Calculates the log of the Gaussian probability density function.

    log P(x | mu, sigma) = -0.5 * log(2 * pi) - log(sigma) - 0.5 * ((x - mu) / sigma)^2

    Args:
        xp: The numerical library (NumPy or JAX.numpy).
        x: Value or array of values.
        mu: Mean of the Gaussian.
        sigma: Standard deviation of the Gaussian.

    Returns:
        The log of the Gaussian PDF.
    """
    return -0.5 * xp.log(2.0 * xp.pi) - xp.log(sigma) - 0.5 * xp.square((x - mu) / sigma)
