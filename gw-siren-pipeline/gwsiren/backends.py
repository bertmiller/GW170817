"""Backend-agnostic numerical utilities for gwsiren."""

from __future__ import annotations

import logging
from typing import Any
import numpy

logger = logging.getLogger(__name__)


def log_gaussian(
    xp: Any,
    x: Any,
    mu: Any,
    sigma: Any,
) -> Any:
    """Compute the log of a Gaussian probability density function.

    This implementation is backend agnostic and works with ``numpy`` or
    ``jax.numpy`` by passing the appropriate module as ``xp``.

    Args:
        xp: Numerical backend module (``numpy`` or ``jax.numpy``).
        x: Point or array at which to evaluate the log PDF.
        mu: Mean of the distribution.
        sigma: Standard deviation of the distribution.

    Returns:
        The log probability density evaluated at ``x``.
    """
    term1 = -0.5 * xp.log(2.0 * xp.pi)
    term2 = -xp.log(sigma)
    term3 = -0.5 * xp.square((x - mu) / sigma)
    return term1 + term2 + term3

def get_xp(preferred_backend: str = "auto") -> tuple[object, str]:
    """Determine which numerical module to use.

    This inspects the requested backend, the availability of JAX, and whether a
    GPU is visible. It returns the numerical module to use and a string
    identifier.

    Args:
        preferred_backend: Desired backend. ``"auto"`` prefers JAX with GPU if
            available, otherwise NumPy. ``"numpy"`` forces NumPy. ``"jax"`` forces
            JAX even if only the CPU is available.

    Returns:
        tuple[object, str]: ``(module, name)`` where ``name`` is ``"numpy"`` or
        ``"jax"``.
    """

    backend = preferred_backend.lower()

    if backend == "numpy":
        logger.info("User selected NumPy backend.")
        return numpy, "numpy"

    # Attempt to import JAX
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except ImportError:
        if backend == "jax":
            logger.warning(
                "JAX backend was requested, but JAX is not installed. Falling back to NumPy."
            )
        else:
            logger.info("JAX not found. Defaulting to NumPy backend.")
        return numpy, "numpy"

    # JAX successfully imported
    try:
        devices = jax.devices()  # type: ignore[attr-defined]
        has_gpu = any(
            d.platform.lower() in ["gpu", "metal"]
            or "gpu" in getattr(d, "device_kind", "").lower()
            for d in devices
        )
    except Exception as exc:  # pragma: no cover - unlikely
        logger.warning(
            "Could not reliably determine JAX device types: %s. Assuming no specialized hardware (GPU/TPU) for JAX.",
            exc,
        )
        has_gpu = False

    if backend == "jax":
        if has_gpu:
            logger.info("JAX backend selected by user. Using JAX on GPU.")
        else:
            logger.info(
                "JAX backend selected by user. Using JAX on CPU (No GPU detected/available to JAX)."
            )
        return jnp, "jax"  # type: ignore[return-value]

    if backend == "auto":
        if has_gpu:
            logger.info("Auto backend selection: JAX available with GPU. Using JAX on GPU.")
            return jnp, "jax"  # type: ignore[return-value]
        logger.info(
            "Auto backend selection: JAX is available, but no GPU detected. Falling back to NumPy as per current policy for 'auto' mode."
        )
        return numpy, "numpy"

    logger.warning("Unexpected condition in backend selection. Defaulting to NumPy.")
    return numpy, "numpy"
