import pytest
import numpy as np
import importlib # For checking if JAX is installed
# from astropy.cosmology import FlatLambdaCDM # No longer used in H0LogLikelihood
# from astropy import units as u # No longer used in H0LogLikelihood

from gwsiren.h0_mcmc_analyzer import (
    H0LogLikelihood,
    get_log_likelihood_h0,
    # DEFAULT_SIGMA_V_PEC, # These are used by H0LogLikelihood constructor defaults
    # DEFAULT_C_LIGHT,
    # DEFAULT_OMEGA_M,
    # DEFAULT_H0_PRIOR_MIN,
    # DEFAULT_H0_PRIOR_MAX,
)
from gwsiren.backends import get_xp # To get xp_np for direct instantiation
# from gwsiren import CONFIG # Not strictly needed if passing backend_str directly


# Example mock data (adjust sizes as needed for thorough testing)
MOCK_DL_GW_SAMPLES = np.array(np.random.normal(loc=700, scale=70, size=100), dtype=np.float64)
MOCK_HOST_ZS = np.array([0.1, 0.12, 0.09, 0.15, 0.11] * 5, dtype=np.float64) # N_hosts = 25
MOCK_HOST_MASS_PROXY = np.array(np.random.uniform(low=10, high=100, size=len(MOCK_HOST_ZS)), dtype=np.float64)
MOCK_HOST_Z_ERR = np.array(np.random.uniform(low=0.001, high=0.01, size=len(MOCK_HOST_ZS)), dtype=np.float64)
MOCK_THETA = np.array([70.0, 0.1], dtype=np.float64) # H0, alpha


@pytest.fixture(autouse=True)
def _stub_main_module(monkeypatch):
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


# Test cases for verifying weight application in H0LogLikelihood
WEIGHT_TEST_CASES_FOR_LIKELIHOOD = [
    (
        "flat_prior_alpha_0",
        np.array([10.0, 20.0, 70.0]),
        0.0,
        0.0,
    ),
    (
        "mass_proportional_alpha_1",
        np.array([10.0, 30.0, 60.0]),
        1.0,
        0.0,
    ),
    (
        "inverse_mass_proportional_alpha_neg1",
        np.array([10.0, 20.0, 100.0]),
        -1.0,
        0.0,
    ),
    (
        "fractional_alpha_dominant_mass",
        np.array([1.0, 1.0, 100.0]),
        0.5,
        0.0,
    ),
    (
        "all_equal_masses_any_alpha",
        np.array([50.0, 50.0, 50.0, 50.0]),
        0.8,
        0.0,
    ),
]


def test_H0LogLikelihood_init_success(mock_config):
    dL_gw_samples = np.array([100.0, 150.0])
    host_z = np.array([0.02, 0.03])

    mass_proxy = np.array([1.0, 2.0])
    z_err = np.array([0.0, 0.0])
    ll = H0LogLikelihood(dL_gw_samples, host_z, mass_proxy, z_err)

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
    ll_scalar = H0LogLikelihood(dL_gw_samples, [0.02], [1.0], [0.0])
    assert np.allclose(ll_scalar.z_values, np.array([0.02]))


@pytest.mark.parametrize("invalid_dl", [None, []])
def test_H0LogLikelihood_init_invalid_dl_samples(invalid_dl, mock_config):
    with pytest.raises(ValueError, match="dL_gw_samples cannot be None or empty."):
        H0LogLikelihood(invalid_dl, np.array([0.02]), np.array([1.0]), np.array([0.0]))


@pytest.mark.parametrize("invalid_z", [None, []])
def test_H0LogLikelihood_init_invalid_host_z(invalid_z, mock_config):
    with pytest.raises(ValueError, match="host_galaxies_z cannot be None or empty."):
        H0LogLikelihood(np.array([100.0]), invalid_z, np.array([1.0]), np.array([0.0]))


def test_H0LogLikelihood_lum_dist_model_single_z(mock_config):
    ll = H0LogLikelihood(np.array([10.0]), np.array([0.01]), np.array([1.0]), np.array([0.0]))
    test_z = 0.05
    test_H0 = 70.0
    cosmo = FlatLambdaCDM(H0=test_H0 * u.km / u.s / u.Mpc, Om0=DEFAULT_OMEGA_M)
    expected = cosmo.luminosity_distance(test_z).value

    result = ll._lum_dist_model(test_z, test_H0)
    assert np.isclose(result, expected)


def test_H0LogLikelihood_lum_dist_model_array_z(mock_config):
    ll = H0LogLikelihood(np.array([10.0]), np.array([0.01]), np.array([1.0]), np.array([0.0]))
    test_z = np.array([0.05, 0.1])
    test_H0 = 70.0
    cosmo = FlatLambdaCDM(H0=test_H0 * u.km / u.s / u.Mpc, Om0=DEFAULT_OMEGA_M)
    expected = cosmo.luminosity_distance(test_z).value

    result = ll._lum_dist_model(test_z, test_H0)
    assert np.allclose(result, expected)


def test_H0LogLikelihood_lum_dist_model_z_zero(mock_config):
    ll = H0LogLikelihood(np.array([10.0]), np.array([0.01]), np.array([1.0]), np.array([0.0]))
    result = ll._lum_dist_model(0.0, 70.0)
    assert np.isclose(result, 0.0)


def test_lum_dist_model_uses_single_cosmology_instance(mock_config, mocker):
    call_counter = {'count': 0}

    class DummyCosmo:
        def __init__(self, *args, **kwargs):
            call_counter['count'] += 1

        def luminosity_distance(self, z):
            return np.asarray(z) * u.Mpc

    mocker.patch('gwsiren.h0_mcmc_analyzer.FlatLambdaCDM', DummyCosmo)

    ll = H0LogLikelihood(np.array([10.0]), np.array([0.01]), np.array([1.0]), np.array([0.0]))

    ll._lum_dist_model(0.1, 70.0)
    ll._lum_dist_model(0.2, 70.0)

    assert call_counter['count'] == 1


def test_call_vectorized_vs_looped_equivalence(mock_config):
    dL_gw_samples = np.array([50.0, 60.0, 70.0, 80.0])
    host_z = np.array([0.01, 0.015])
    mass_proxy = np.array([1.0, 2.0])
    z_err = np.array([0.0, 0.0])
    likelihood_vectorized = H0LogLikelihood(
        dL_gw_samples,
        host_z,
        mass_proxy,
        z_err,
        use_vectorized_likelihood=True,
    )
    likelihood_looped = H0LogLikelihood(
        dL_gw_samples,
        host_z,
        mass_proxy,
        z_err,
        use_vectorized_likelihood=False,
    )

    H0_value = 70.0
    log_L_vectorized = likelihood_vectorized([H0_value, 0.0])
    log_L_looped = likelihood_looped([H0_value, 0.0])

    assert np.isfinite(log_L_vectorized)
    assert np.isfinite(log_L_looped)
    assert np.isclose(log_L_vectorized, log_L_looped, rtol=1e-5, atol=1e-8)


def test_get_log_likelihood_returns_H0LogLikelihood_instance(mock_config):
    dL_gw_samples = np.array([100.0, 110.0])
    host_z = np.array([0.02, 0.03])

    instance = get_log_likelihood_h0(dL_gw_samples, host_z, np.ones_like(host_z), np.zeros_like(host_z))
    assert isinstance(instance, H0LogLikelihood)


def test_get_log_likelihood_chooses_vectorized_for_small_data(mock_config):
    dL_gw_samples = np.array([1.0] * 10)
    host_z = np.array([0.1] * 5)

    instance = get_log_likelihood_h0(dL_gw_samples, host_z, np.ones_like(host_z), np.zeros_like(host_z))
    assert instance.use_vectorized_likelihood is True


def test_get_log_likelihood_chooses_looped_due_to_large_elements(mock_config):
    base_samples = np.array([1.0, 2.0])
    base_hosts = np.array([0.1, 0.2])

    # Fake lengths so that len(samples) * len(hosts) exceeds threshold
    fake_samples = FakeArray(base_samples, 600_000_000)
    fake_hosts = FakeArray(base_hosts, 2)

    instance = get_log_likelihood_h0(fake_samples, fake_hosts, np.ones_like(base_hosts), np.zeros_like(base_hosts))
    assert instance.use_vectorized_likelihood is False


def test_get_log_likelihood_passes_other_args_correctly(mock_config):
    dL_gw_samples = np.array([1.0, 2.0])
    host_z = np.array([0.01])

    instance = get_log_likelihood_h0(
        dL_gw_samples,
        host_z,
        np.array([1.0]),
        np.array([0.0]),
        sigma_v=500,
        h0_min=20,
        h0_max=180,
        omega_m_val=0.25,
    )

    assert instance.sigma_v == 500
    assert instance.h0_min == 20
    assert instance.h0_max == 180
    assert instance.omega_m_val == 0.25


@pytest.mark.parametrize(
    "test_name, mass_proxy_values, alpha_input, expected_log_L",
    WEIGHT_TEST_CASES_FOR_LIKELIHOOD,
)
def test_h0_loglikelihood_weight_application(
    test_name,
    mass_proxy_values,
    alpha_input,
    expected_log_L,
    mock_config,
    mocker,
):
    """Verify that galaxy weights are correctly applied in the log-likelihood."""

    dL_gw_samples = np.array([100.0])
    host_z = np.full(len(mass_proxy_values), 0.1)

    likelihood = H0LogLikelihood(dL_gw_samples, host_z, mass_proxy_values, np.zeros_like(host_z))

    # Force per-galaxy log likelihood terms to zero so only weights matter
    mocker.patch(
        "gwsiren.backends.logpdf_normal_xp", # Patching the backend version now
        # This mock needs to be careful about the xp argument if it's called by different backends.
        # For simplicity, assuming it's called with numpy arrays for this weight test.
        # A more robust mock might inspect `xp` or be specific to the H0LogLikelihood instance.
        side_effect=lambda xp, x, loc=None, scale=None: xp.zeros_like(x) if hasattr(xp, 'zeros_like') else np.zeros_like(x),
    )

    actual_log_L = likelihood([70.0, alpha_input])
    assert np.isclose(actual_log_L, expected_log_L)


@pytest.mark.skipif(not importlib.util.find_spec("jax"), reason="JAX not installed, skipping JAX comparison tests")
def test_numerical_equivalence_numpy_vectorized_vs_jax():
    # For NumPy vectorized path, get_log_likelihood_h0 will choose it for small data.
    ll_numpy_vec_obj = get_log_likelihood_h0(
        requested_backend_str="numpy",
        dL_gw_samples=MOCK_DL_GW_SAMPLES.copy(),
        host_galaxies_z=MOCK_HOST_ZS.copy(),
        host_galaxies_mass_proxy=MOCK_HOST_MASS_PROXY.copy(),
        host_galaxies_z_err=MOCK_HOST_Z_ERR.copy()
    )
    # JAX will use its single optimized path
    ll_jax_obj = get_log_likelihood_h0(
        requested_backend_str="jax",
        dL_gw_samples=MOCK_DL_GW_SAMPLES.copy(),
        host_galaxies_z=MOCK_HOST_ZS.copy(),
        host_galaxies_mass_proxy=MOCK_HOST_MASS_PROXY.copy(),
        host_galaxies_z_err=MOCK_HOST_Z_ERR.copy()
    )

    val_numpy_vec = ll_numpy_vec_obj(MOCK_THETA)
    val_jax = ll_jax_obj(MOCK_THETA)

    assert np.allclose(val_numpy_vec, val_jax, rtol=1e-8, atol=1e-8), \
        f"NumPy vectorized ({val_numpy_vec}) and JAX ({val_jax}) outputs differ."

@pytest.mark.skipif(not importlib.util.find_spec("jax"), reason="JAX not installed, skipping JAX comparison tests")
def test_numerical_equivalence_numpy_looped_vs_jax():
    # To get NumPy looped path, we directly instantiate H0LogLikelihood
    xp_np, backend_name_np, _ = get_xp("numpy") # Get the numpy module via get_xp

    ll_numpy_loop_obj = H0LogLikelihood(
        xp=xp_np, backend_name=backend_name_np,
        dL_gw_samples=xp_np.asarray(MOCK_DL_GW_SAMPLES),
        host_galaxies_z=xp_np.asarray(MOCK_HOST_ZS),
        host_galaxies_mass_proxy=xp_np.asarray(MOCK_HOST_MASS_PROXY),
        host_galaxies_z_err=xp_np.asarray(MOCK_HOST_Z_ERR),
        # Default priors and cosmology params from h0_mcmc_analyzer will be used by H0LogLikelihood constructor
        use_vectorized_likelihood=False # Force looped path for NumPy
    )
    
    ll_jax_obj = get_log_likelihood_h0(
        requested_backend_str="jax",
        dL_gw_samples=MOCK_DL_GW_SAMPLES.copy(),
        host_galaxies_z=MOCK_HOST_ZS.copy(),
        host_galaxies_mass_proxy=MOCK_HOST_MASS_PROXY.copy(),
        host_galaxies_z_err=MOCK_HOST_Z_ERR.copy()
    )

    val_numpy_loop = ll_numpy_loop_obj(MOCK_THETA)
    val_jax = ll_jax_obj(MOCK_THETA)
    
    assert np.allclose(val_numpy_loop, val_jax, rtol=1e-8, atol=1e-8), \
        f"NumPy looped ({val_numpy_loop}) and JAX ({val_jax}) outputs differ."
