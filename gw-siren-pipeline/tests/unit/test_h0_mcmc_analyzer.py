import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from h0_mcmc_analyzer import (
    H0LogLikelihood,
    get_log_likelihood_h0,
    DEFAULT_SIGMA_V_PEC,
    DEFAULT_C_LIGHT,
    DEFAULT_OMEGA_M,
    DEFAULT_H0_PRIOR_MIN,
    DEFAULT_H0_PRIOR_MAX,
)


@pytest.fixture(autouse=True)
def _stub_main_module(monkeypatch):
    """Prevent importing the real ``main`` module during ``mock_config`` setup."""
    import types, sys

    dummy = types.ModuleType("main")
    dummy.CONFIG = None
    monkeypatch.setitem(sys.modules, "main", dummy)
    yield
    monkeypatch.delitem(sys.modules, "main", raising=False)


class FakeArray:
    """Simple container reporting a custom length while storing a small array."""

    def __init__(self, values, fake_len):
        self._values = np.asarray(values)
        self._fake_len = fake_len

    def __len__(self):
        return self._fake_len

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)

    def __getitem__(self, item):
        return self._values[item]

    def __iter__(self):
        return iter(self._values)


def test_H0LogLikelihood_init_success(mock_config):
    dL_gw_samples = np.array([100.0, 150.0])
    host_z = np.array([0.02, 0.03])

    ll = H0LogLikelihood(dL_gw_samples, host_z)

    assert isinstance(ll, H0LogLikelihood)
    assert np.allclose(ll.dL_gw_samples, dL_gw_samples)
    assert np.allclose(ll.z_values, host_z)
    assert ll.sigma_v == DEFAULT_SIGMA_V_PEC
    assert ll.c_val == DEFAULT_C_LIGHT
    assert ll.omega_m_val == DEFAULT_OMEGA_M
    assert ll.h0_min == DEFAULT_H0_PRIOR_MIN
    assert ll.h0_max == DEFAULT_H0_PRIOR_MAX
    assert ll.use_vectorized_likelihood is False

    # Single-value sequence should also be stored as an array
    ll_scalar = H0LogLikelihood(dL_gw_samples, [0.02])
    assert np.allclose(ll_scalar.z_values, np.array([0.02]))


@pytest.mark.parametrize("invalid_dl", [None, []])
def test_H0LogLikelihood_init_invalid_dl_samples(invalid_dl, mock_config):
    with pytest.raises(ValueError, match="dL_gw_samples cannot be None or empty."):
        H0LogLikelihood(invalid_dl, np.array([0.02]))


@pytest.mark.parametrize("invalid_z", [None, []])
def test_H0LogLikelihood_init_invalid_host_z(invalid_z, mock_config):
    with pytest.raises(ValueError, match="host_galaxies_z cannot be None or empty."):
        H0LogLikelihood(np.array([100.0]), invalid_z)


def test_H0LogLikelihood_lum_dist_model_single_z(mock_config):
    ll = H0LogLikelihood(np.array([10.0]), np.array([0.01]))
    test_z = 0.05
    test_H0 = 70.0
    cosmo = FlatLambdaCDM(H0=test_H0 * u.km / u.s / u.Mpc, Om0=DEFAULT_OMEGA_M)
    expected = cosmo.luminosity_distance(test_z).value

    result = ll._lum_dist_model(test_z, test_H0)
    assert np.isclose(result, expected)


def test_H0LogLikelihood_lum_dist_model_array_z(mock_config):
    ll = H0LogLikelihood(np.array([10.0]), np.array([0.01]))
    test_z = np.array([0.05, 0.1])
    test_H0 = 70.0
    cosmo = FlatLambdaCDM(H0=test_H0 * u.km / u.s / u.Mpc, Om0=DEFAULT_OMEGA_M)
    expected = cosmo.luminosity_distance(test_z).value

    result = ll._lum_dist_model(test_z, test_H0)
    assert np.allclose(result, expected)


def test_H0LogLikelihood_lum_dist_model_z_zero(mock_config):
    ll = H0LogLikelihood(np.array([10.0]), np.array([0.01]))
    result = ll._lum_dist_model(0.0, 70.0)
    assert np.isclose(result, 0.0)


def test_call_vectorized_vs_looped_equivalence(mock_config):
    dL_gw_samples = np.array([50.0, 60.0, 70.0, 80.0])
    host_z = np.array([0.01, 0.015])
    likelihood_vectorized = H0LogLikelihood(
        dL_gw_samples,
        host_z,
        use_vectorized_likelihood=True,
    )
    likelihood_looped = H0LogLikelihood(
        dL_gw_samples,
        host_z,
        use_vectorized_likelihood=False,
    )

    H0_value = 70.0
    log_L_vectorized = likelihood_vectorized([H0_value])
    log_L_looped = likelihood_looped([H0_value])

    assert np.isfinite(log_L_vectorized)
    assert np.isfinite(log_L_looped)
    assert np.isclose(log_L_vectorized, log_L_looped, rtol=1e-5, atol=1e-8)


def test_get_log_likelihood_returns_H0LogLikelihood_instance(mock_config):
    dL_gw_samples = np.array([100.0, 110.0])
    host_z = np.array([0.02, 0.03])

    instance = get_log_likelihood_h0(dL_gw_samples, host_z)
    assert isinstance(instance, H0LogLikelihood)


def test_get_log_likelihood_chooses_vectorized_for_small_data(mock_config):
    dL_gw_samples = np.array([1.0] * 10)
    host_z = np.array([0.1] * 5)

    instance = get_log_likelihood_h0(dL_gw_samples, host_z)
    assert instance.use_vectorized_likelihood is True


def test_get_log_likelihood_chooses_looped_due_to_large_elements(mock_config):
    base_samples = np.array([1.0, 2.0])
    base_hosts = np.array([0.1, 0.2])

    # Fake lengths so that len(samples) * len(hosts) exceeds threshold
    fake_samples = FakeArray(base_samples, 600_000_000)
    fake_hosts = FakeArray(base_hosts, 2)

    instance = get_log_likelihood_h0(fake_samples, fake_hosts)
    assert instance.use_vectorized_likelihood is False


def test_get_log_likelihood_passes_other_args_correctly(mock_config):
    dL_gw_samples = np.array([1.0, 2.0])
    host_z = np.array([0.01])

    instance = get_log_likelihood_h0(
        dL_gw_samples,
        host_z,
        sigma_v=500,
        h0_min=20,
        h0_max=180,
        omega_m_val=0.25,
    )

    assert instance.sigma_v == 500
    assert instance.h0_min == 20
    assert instance.h0_max == 180
    assert instance.omega_m_val == 0.25
