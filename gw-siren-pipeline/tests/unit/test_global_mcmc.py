import numpy as np
import emcee
import pytest

from gwsiren.global_mcmc import run_global_mcmc, process_global_mcmc_samples


@pytest.fixture(autouse=True)
def _stub_main_module(monkeypatch):
    import types, sys

    dummy = types.ModuleType("main")
    dummy.CONFIG = None
    monkeypatch.setitem(sys.modules, "main", dummy)
    yield
    monkeypatch.delitem(sys.modules, "main", raising=False)


class SimpleGaussianLL:
    """Simple 2D Gaussian log-likelihood for testing."""

    def __init__(self):
        self.h0_min = 20.0
        self.h0_max = 150.0
        self.alpha_min = -2.0
        self.alpha_max = 2.0

    def __call__(self, theta):
        h0, alpha = theta
        if not (self.h0_min <= h0 <= self.h0_max):
            return -np.inf
        if not (self.alpha_min <= alpha <= self.alpha_max):
            return -np.inf
        return -0.5 * (((h0 - 70.0) / 5.0) ** 2 + ((alpha - 0.0) / 1.0) ** 2)


def test_run_global_mcmc_returns_sampler(mock_config):
    ll = SimpleGaussianLL()
    sampler = run_global_mcmc(
        ll,
        n_walkers=8,
        n_steps=10,
        initial_pos_config={"H0": {"mean": 70.0, "std": 2.0}, "alpha": {"mean": 0.0, "std": 0.2}},
    )
    assert isinstance(sampler, emcee.EnsembleSampler)
    assert sampler.get_chain().shape == (10, 8, 2)


def test_process_global_mcmc_samples_returns_array(mock_config):
    ll = SimpleGaussianLL()
    sampler = run_global_mcmc(
        ll,
        n_walkers=6,
        n_steps=15,
        initial_pos_config={"H0": {"mean": 70.0, "std": 3.0}, "alpha": {"mean": 0.0, "std": 0.3}},
    )
    samples = process_global_mcmc_samples(sampler, burnin=5, thin_by=1, n_dim=2)
    assert samples is not None
    assert samples.shape[1] == 2
    assert samples.shape[0] > 0
