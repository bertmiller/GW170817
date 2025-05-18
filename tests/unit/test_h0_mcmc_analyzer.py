import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from h0_mcmc_analyzer import (
    H0LogLikelihood,
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
