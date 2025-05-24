"""Tests for MCMC convergence utilities."""

from __future__ import annotations

import numpy as np
import pytest

from .utils.test_helpers import rhat, effective_sample_size


def test_rhat_unity_for_identical_chains():
    """R-hat for identical chains should equal sqrt((n-1)/n)."""
    chains = np.tile(np.linspace(0, 1, 10), (4, 1))
    expected = np.sqrt((10 - 1) / 10)
    assert pytest.approx(rhat(chains), rel=1e-6) == expected


def test_effective_sample_size_less_than_n():
    """ESS should be less than chain length for correlated data."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=100)
    chain[1:] = chain[:-1] * 0.9 + chain[1:] * 0.1
    ess = effective_sample_size(chain)
    assert ess < len(chain) 