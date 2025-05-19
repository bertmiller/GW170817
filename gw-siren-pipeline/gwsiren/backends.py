"""Backend-agnostic numerical utilities for gwsiren."""

from __future__ import annotations

import logging
from typing import Any

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

