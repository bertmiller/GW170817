"""Helper functions used in tests."""

from __future__ import annotations

import numpy as np


def rhat(chains: np.ndarray) -> float:
    """Compute the Gelman-Rubin statistic for convergence diagnostics.

    Args:
        chains: Array with shape ``(n_chains, n_samples)``.

    Returns:
        Estimated R-hat value.
    """
    m, n = chains.shape
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    b = n * np.var(chain_means, ddof=1)
    w = np.mean(np.var(chains, axis=1, ddof=1))
    var_hat = ((n - 1) / n) * w + b / n
    return np.sqrt(var_hat / w)


def effective_sample_size(chain: np.ndarray) -> float:
    """Estimate the effective sample size of a 1-D chain."""
    chain = np.asarray(chain)
    n = len(chain)
    mean = np.mean(chain)
    var = np.var(chain, ddof=0)
    if var == 0:
        return float(n)
    acf = np.correlate(chain - mean, chain - mean, mode="full")[-n:]
    acf /= var * np.arange(n, 0, -1)
    # sum positive lags until negative
    positive_acf = acf[1:]
    idx = np.where(positive_acf < 0)[0]
    if idx.size > 0:
        t = idx[0]
    else:
        t = len(positive_acf)
    ess = n / (1 + 2 * positive_acf[:t].sum())
    return float(ess) 