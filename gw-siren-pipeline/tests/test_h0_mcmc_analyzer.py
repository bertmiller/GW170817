import numpy as np
import pytest

from gwsiren.h0_mcmc_analyzer import (
    get_log_likelihood_h0,
    DEFAULT_H0_PRIOR_MIN,
    DEFAULT_H0_PRIOR_MAX
)


def test_log_likelihood_basic_and_bounds():
    rng = np.random.default_rng(0)
    dL = rng.normal(50.0, 5.0, size=20)
    host_z = np.array([0.1, 0.12, 0.08])

    ll = get_log_likelihood_h0(dL, host_z)
    val = ll([70.0])
    assert np.isfinite(val)
    assert ll([DEFAULT_H0_PRIOR_MIN - 1]) == -np.inf
    assert ll([DEFAULT_H0_PRIOR_MAX + 1]) == -np.inf

