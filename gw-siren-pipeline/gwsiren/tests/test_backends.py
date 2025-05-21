import numpy
import scipy.stats
from numpy.testing import assert_allclose
import pytest # For caplog
import logging
import sys
from unittest.mock import patch, MagicMock

# Assuming 'gw_siren_pipeline.gwsiren.backends' is the correct module path
# If the file is directly in 'gwsiren', it might be 'gwsiren.backends'
# Based on previous tasks, it seems 'gw_siren_pipeline.gwsiren.backends' is what's expected.
from gw_siren_pipeline.gwsiren.backends import log_gaussian, get_xp 

# Attempt to import jax and jax.numpy for type checking and real JAX tests if available
try:
    import jax
    import jax.numpy as jnp
    JAX_IS_AVAILABLE_FOR_REAL = True
except ImportError:
    jax = None # type: ignore
    jnp = None # type: ignore
    JAX_IS_AVAILABLE_FOR_REAL = False


# Use numpy backend for log_gaussian tests as we are comparing with scipy.stats
numpy_xp, _ = get_xp("numpy")

class TestLogGaussian:
    # ... (existing TestLogGaussian methods remain unchanged) ...
    def test_scalar_input(self):
        """Tests log_gaussian with scalar inputs."""
        x, mu, sigma = 0.5, 0.0, 1.0
        expected = scipy.stats.norm.logpdf(x, loc=mu, scale=sigma)
        actual = log_gaussian(numpy_xp, x, mu, sigma)
        assert_allclose(actual, expected, rtol=1e-9,
                        err_msg="Scalar input test failed for log_gaussian.")

    def test_array_input(self):
        """Tests log_gaussian with NumPy array inputs."""
        x = numpy_xp.array([-1.0, 0.0, 1.0, 2.5])
        mu = 0.5
        sigma = 1.5
        
        expected = scipy.stats.norm.logpdf(x, loc=mu, scale=sigma)
        actual = log_gaussian(numpy_xp, x, mu, sigma)
        assert_allclose(actual, expected, rtol=1e-9,
                        err_msg="Array input test failed for log_gaussian.")

    def test_different_parameters(self):
        """Tests log_gaussian with different mu and sigma values."""
        x = numpy_xp.array([10.0, 12.0])
        mu = 11.0
        sigma = 2.0
        
        expected = scipy.stats.norm.logpdf(x, loc=mu, scale=sigma)
        actual = log_gaussian(numpy_xp, x, mu, sigma)
        assert_allclose(actual, expected, rtol=1e-9,
                        err_msg="Different parameters test failed for log_gaussian.")

    def test_zero_sigma(self):
        """Tests log_gaussian behavior with sigma = 0."""
        x, mu, sigma = 1.0, 0.0, 0.0
        expected_scipy = scipy.stats.norm.logpdf(x, loc=mu, scale=sigma) 
        actual = log_gaussian(numpy_xp, x, mu, sigma)
        assert actual == expected_scipy

        x_eq_mu, mu_eq_x, sigma_zero = 0.0, 0.0, 0.0
        expected_scipy_nan = scipy.stats.norm.logpdf(x_eq_mu, loc=mu_eq_x, scale=sigma_zero)
        actual_nan = log_gaussian(numpy_xp, x_eq_mu, mu_eq_x, sigma_zero)
        assert numpy_xp.isnan(actual_nan) and numpy_xp.isnan(expected_scipy_nan)


    def test_negative_sigma(self):
        """Tests log_gaussian behavior with sigma < 0 (should be nan)."""
        x, mu, sigma = 0.5, 0.0, -1.0
        expected_scipy = scipy.stats.norm.logpdf(x, loc=mu, scale=sigma) 
        actual = log_gaussian(numpy_xp, x, mu, sigma) 
        assert numpy_xp.isnan(actual) and numpy_xp.isnan(expected_scipy)


class TestGetXpFallbacks:
    """Tests for get_xp backend selection, focusing on fallback mechanisms."""

    @patch.dict(sys.modules, {'jax': None, 'jax.numpy': None})
    def test_get_xp_jax_preferred_jax_unavailable(self, caplog):
        """Test get_xp with preferred_backend='jax' when JAX is not importable."""
        caplog.set_level(logging.WARNING)
        xp_module, backend_name = get_xp(preferred_backend="jax")
        
        assert backend_name == "numpy"
        assert xp_module is numpy
        assert any("JAX backend was requested, but JAX installation not found." in rec.message for rec in caplog.records)
        assert any(rec.levelname == "WARNING" for rec in caplog.records)

    @patch.dict(sys.modules, {'jax': None, 'jax.numpy': None})
    def test_get_xp_auto_preferred_jax_unavailable(self, caplog):
        """Test get_xp with preferred_backend='auto' when JAX is not importable."""
        caplog.set_level(logging.INFO) # Auto mode logs fallback as INFO
        xp_module, backend_name = get_xp(preferred_backend="auto")
        
        assert backend_name == "numpy"
        assert xp_module is numpy
        assert any("Auto backend selection: JAX installation not found. Falling back to NumPy." in rec.message for rec in caplog.records)

    # To mock jax.devices(), jax itself must be importable, but jax.devices() needs to be controlled.
    # We also need to ensure that our 'jax' and 'jnp' in the test file are the mocked ones.
    @patch('gw_siren_pipeline.gwsiren.backends.jax', MagicMock()) # Mock the jax module used by backends.py
    def test_get_xp_auto_preferred_jax_cpu_only(self, caplog):
        """Test get_xp with preferred_backend='auto', JAX available, but only CPU devices."""
        # Configure the mocked jax.devices() to return only CPU-like devices
        # The .lower() and startswith('gpu') checks mean 'cpu' won't match.
        # The 'metal' check is also important for macOS.
        # We need to mock the device object structure expected by get_xp.
        
        mock_cpu_device = MagicMock()
        mock_cpu_device.device_kind = "CPU" # For older JAX versions or if platform is not standard
        mock_cpu_device.platform = "cpu"    # For newer JAX versions

        # Ensure the 'jax' seen by backends.py is our MagicMock
        sys.modules['jax'] = sys.modules['gw_siren_pipeline.gwsiren.backends.jax']
        sys.modules['jax.numpy'] = MagicMock() # jnp also needs to be mockable
        
        # Set the return value for jax.devices()
        sys.modules['gw_siren_pipeline.gwsiren.backends.jax'].devices.return_value = [mock_cpu_device]
        # Ensure jnp is set to something non-None for the try block in get_xp
        sys.modules['gw_siren_pipeline.gwsiren.backends.jnp'] = sys.modules['jax.numpy']


        caplog.set_level(logging.INFO)
        xp_module, backend_name = get_xp(preferred_backend="auto")
        
        assert backend_name == "numpy" # Should fall back to numpy if auto and no GPU
        assert xp_module is numpy
        assert any("JAX is available, but no GPU detected. Falling back to NumPy." in rec.message for rec in caplog.records)
        
        # Cleanup sys.modules if changed, or rely on test isolation if pytest handles it well.
        # For safety, if we explicitly put things in sys.modules, we might need to remove them.
        # However, @patch should handle the scope.

    @patch('gw_siren_pipeline.gwsiren.backends.jax', MagicMock())
    def test_get_xp_jax_preferred_jax_cpu_only(self, caplog):
        """Test get_xp with preferred_backend='jax', JAX available, but only CPU devices."""
        mock_cpu_device = MagicMock()
        mock_cpu_device.device_kind = "CPU"
        mock_cpu_device.platform = "cpu"

        # Ensure the 'jax' seen by backends.py is our MagicMock
        sys.modules['jax'] = sys.modules['gw_siren_pipeline.gwsiren.backends.jax']
        # Mock jax.numpy as jnp for the import inside get_xp
        mock_jnp = MagicMock()
        sys.modules['jax.numpy'] = mock_jnp
        sys.modules['gw_siren_pipeline.gwsiren.backends.jnp'] = mock_jnp


        sys.modules['gw_siren_pipeline.gwsiren.backends.jax'].devices.return_value = [mock_cpu_device]

        caplog.set_level(logging.INFO)
        xp_module, backend_name = get_xp(preferred_backend="jax")
        
        assert backend_name == "jax" 
        assert xp_module is mock_jnp # Should be the mocked jax.numpy
        assert any("Using JAX backend on CPU as explicitly requested" in rec.message for rec in caplog.records)

    @pytest.mark.skipif(not JAX_IS_AVAILABLE_FOR_REAL, reason="Real JAX installation not found, cannot test real GPU detection logic.")
    def test_get_xp_auto_real_jax_gpu_if_present(self, caplog):
        """Test get_xp with 'auto' and a real JAX, expecting JAX if GPU is present."""
        caplog.set_level(logging.INFO)
        # This test relies on the actual hardware environment.
        # We can't force GPU presence here, but we can check consistency.
        
        real_xp_module, real_backend_name = get_xp(preferred_backend="auto")
        
        try:
            # Check if JAX reports any GPU devices
            # This check is similar to the one in get_xp
            gpu_devices = [d for d in jax.devices() if d.device_kind.lower().startswith('gpu') or d.platform.lower() == 'gpu']
            if not gpu_devices: # Fallback for Metal etc.
                 gpu_devices = [d for d in jax.devices() if d.platform.lower() in ['gpu', 'metal'] or 'gpu' in d.device_kind.lower()]
            is_real_gpu_present = len(gpu_devices) > 0
        except Exception:
            is_real_gpu_present = False # If jax.devices() fails for some reason

        if is_real_gpu_present:
            assert real_backend_name == "jax"
            assert real_xp_module is jnp
            assert any("Auto backend selection: Using JAX on GPU." in rec.message for rec in caplog.records)
        else:
            assert real_backend_name == "numpy"
            assert real_xp_module is numpy
            assert any("JAX is available, but no GPU detected. Falling back to NumPy." in rec.message for rec in caplog.records)

    def test_get_xp_numpy_preferred(self, caplog):
        """Test get_xp with preferred_backend='numpy'."""
        caplog.set_level(logging.INFO)
        xp_module, backend_name = get_xp(preferred_backend="numpy")
        
        assert backend_name == "numpy"
        assert xp_module is numpy
        assert any("Using NumPy backend as explicitly requested." in rec.message for rec in caplog.records)

    def test_get_xp_invalid_preference(self, caplog):
        """Test get_xp with an invalid preferred_backend string."""
        caplog.set_level(logging.WARNING)
        xp_module, backend_name = get_xp(preferred_backend="invalid_backend_name")
        
        assert backend_name == "numpy" # Should default to numpy
        assert xp_module is numpy
        assert any("Invalid preferred_backend 'invalid_backend_name'. Falling back to NumPy." in rec.message for rec in caplog.records)
        assert any(rec.levelname == "WARNING" for rec in caplog.records)
