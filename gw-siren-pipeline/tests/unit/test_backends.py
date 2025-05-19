import numpy as np
from scipy.stats import norm

from gwsiren.backends import log_gaussian


def test_log_gaussian_scalar():
    x, mu, sigma = 1.0, 0.0, 1.0
    expected = norm.logpdf(x, loc=mu, scale=sigma)
    actual = log_gaussian(np, x, mu, sigma)
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_log_gaussian_array():
    x = np.array([1.0, 2.0, 0.5])
    mu = np.array([0.0, 2.5, 0.5])
    sigma = np.array([1.0, 0.5, 2.0])
    expected = norm.logpdf(x, loc=mu, scale=sigma)
    actual = log_gaussian(np, x, mu, sigma)
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_log_gaussian_broadcast():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    mu = np.array([0.0, 1.0])
    sigma = 1.5
    expected = norm.logpdf(x, loc=mu, scale=sigma)
    actual = log_gaussian(np, x, mu, sigma)
    np.testing.assert_allclose(actual, expected, rtol=1e-9)
