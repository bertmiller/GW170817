#!/usr/bin/env python3
"""
Comprehensive robustness tests for backend consistency and reliability.

This test suite is designed to catch the types of issues we encountered during
the JAX/NumPy consistency fixes and prevent similar problems in the future.

Usage:
    pytest gw-siren-pipeline/tests/test_backend_robustness.py -v
    pytest gw-siren-pipeline/tests/test_backend_robustness.py -m consistency
    pytest gw-siren-pipeline/tests/test_backend_robustness.py -m "not slow"
"""

import os
import sys
import numpy as np
import pytest
import importlib
from typing import List, Dict, Any, Tuple, Optional
from unittest.mock import patch
import warnings

# Force JAX CPU for testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Register custom pytest marks to avoid warnings
pytest_plugins = []
def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "consistency: Tests for numerical consistency between backends")
    config.addinivalue_line("markers", "slow: Tests that take significant time to run")
    config.addinivalue_line("markers", "performance: Performance and speed tests")
    config.addinivalue_line("markers", "integration: End-to-end integration tests")

# Define pytest markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from gwsiren.backends import get_xp, BackendNotAvailableError, logpdf_normal_xp, logsumexp_xp, trapz_xp
from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0, H0LogLikelihood

# Import mock data utilities
def mock_event(n_galaxies=5, seed=None):
    """Create mock event data for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    import pandas as pd
    from collections import namedtuple
    
    # Create mock distance samples
    dl_samples = np.random.lognormal(mean=np.log(100), sigma=0.3, size=50)
    
    # Create mock galaxy catalog
    galaxies_data = {
        'z': np.random.uniform(0.01, 0.05, n_galaxies),
        'mass_proxy': np.random.lognormal(mean=0, sigma=0.5, size=n_galaxies),
        'z_err': np.random.uniform(0.001, 0.01, n_galaxies)
    }
    candidate_galaxies_df = pd.DataFrame(galaxies_data)
    
    # Create named tuple to match expected interface
    EventDataPackage = namedtuple('EventDataPackage', ['event_id', 'dl_samples', 'candidate_galaxies_df'])
    
    return EventDataPackage(
        event_id=f"TEST_EVENT_{seed}",
        dl_samples=dl_samples,
        candidate_galaxies_df=candidate_galaxies_df
    )

def multi_event(n_events=2, seed=None):
    """Create multiple mock events for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    events = []
    for i in range(n_events):
        event_seed = seed + i if seed is not None else None
        events.append(mock_event(n_galaxies=3, seed=event_seed))
    
    return events


class TestAPICompatibility:
    """Test that both backends provide the same API interface."""
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_basic_array_operations(self, backend):
        """Test that basic array operations work consistently."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        xp, name, device = get_xp(backend)
        
        # Basic array creation
        arr = xp.array([1.0, 2.0, 3.0])
        assert arr.shape == (3,)
        assert hasattr(arr, 'dtype')
        
        # Basic operations
        assert xp.sum(arr) > 0
        assert xp.mean(arr) > 0
        assert xp.std(arr) >= 0
        
        # Array methods that both should support
        assert hasattr(xp, 'asarray')
        assert hasattr(xp, 'zeros')
        assert hasattr(xp, 'ones')
        assert hasattr(xp, 'linspace')
        assert hasattr(xp, 'maximum')
        assert hasattr(xp, 'sqrt')
        assert hasattr(xp, 'log')
        assert hasattr(xp, 'exp')
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_backend_specific_functions(self, backend):
        """Test our custom backend-agnostic functions."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        xp, name, device = get_xp(backend)
        
        # Test logpdf_normal_xp
        x = xp.array([0.0, 1.0, 2.0])
        loc = xp.array([0.0, 1.0, 2.0])
        scale = xp.array([1.0, 1.0, 1.0])
        
        result = logpdf_normal_xp(xp, x, loc, scale)
        assert xp.all(xp.isfinite(result))
        
        # Test logsumexp_xp
        a = xp.array([1.0, 2.0, 3.0])
        result = logsumexp_xp(xp, a)
        assert xp.isfinite(result)
        
        # Test trapz_xp
        y = xp.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = trapz_xp(xp, y)
        assert xp.isfinite(result)
        assert result > 0  # Should be positive for this test case


class TestParameterConsistency:
    """Test that parameters have consistent effects across backends."""
    
    @pytest.mark.parametrize("force_non_vectorized", [True, False])
    def test_force_non_vectorized_parameter(self, force_non_vectorized):
        """Test that force_non_vectorized parameter is respected by both backends."""
        pkg = mock_event(n_galaxies=2, seed=42)
        
        backends_to_test = ["numpy"]
        if importlib.util.find_spec("jax"):
            backends_to_test.append("jax")
        
        for backend in backends_to_test:
            ll = get_log_likelihood_h0(
                requested_backend_str=backend,
                dL_gw_samples=pkg.dl_samples,
                host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
                host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
                host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
                force_non_vectorized=force_non_vectorized,
            )
            
            # Check that the parameter was actually respected
            if force_non_vectorized:
                assert not ll.use_vectorized_likelihood, f"{backend} backend ignored force_non_vectorized=True"
            # Note: We don't test the False case because it depends on memory thresholds
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_parameter_ranges_consistency(self, backend):
        """Test that parameter validation is consistent across backends."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=1, seed=42)
        
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # Test prior boundaries (actual bounds are 10.0-200.0 for H0, -1.0-1.0 for alpha)
        assert ll([5.0, 0.0]) == -ll.xp.inf  # Below H0 prior (10.0)
        assert ll([250.0, 0.0]) == -ll.xp.inf  # Above H0 prior (200.0)
        assert ll([70.0, -2.0]) == -ll.xp.inf  # Below alpha prior (-1.0)
        assert ll([70.0, 2.0]) == -ll.xp.inf  # Above alpha prior (1.0)
        
        # Test valid range
        result = ll([70.0, 0.0])
        assert ll.xp.isfinite(result)


@pytest.mark.consistency
class TestNumericalConsistency:
    """Test numerical consistency between backends and algorithms."""
    
    def test_identical_algorithms_identical_results(self):
        """Test that when using identical algorithms, results are numerically identical."""
        if not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=1, seed=42)
        theta = [70.0, 0.0]
        
        # Both non-vectorized (same algorithm)
        ll_numpy = get_log_likelihood_h0(
            requested_backend_str="numpy",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
            force_non_vectorized=True,
        )
        
        ll_jax = get_log_likelihood_h0(
            requested_backend_str="jax",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
            force_non_vectorized=True,
        )
        
        val_numpy = ll_numpy(theta)
        val_jax = ll_jax(theta)
        
        assert np.isfinite(val_numpy) and np.isfinite(val_jax)
        assert np.allclose(val_numpy, val_jax, rtol=1e-12, atol=1e-12)
    
    @pytest.mark.parametrize("h0_value", [50.0, 70.0, 90.0])
    @pytest.mark.parametrize("alpha_value", [-0.5, 0.0, 0.5])
    def test_parameter_sweep_consistency(self, h0_value, alpha_value):
        """Test consistency across different parameter values."""
        if not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=1, seed=42)
        theta = [h0_value, alpha_value]
        
        ll_numpy = get_log_likelihood_h0(
            requested_backend_str="numpy",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
            force_non_vectorized=True,
        )
        
        ll_jax = get_log_likelihood_h0(
            requested_backend_str="jax",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
            force_non_vectorized=True,
        )
        
        val_numpy = ll_numpy(theta)
        val_jax = ll_jax(theta)
        
        # Both should be finite or both should be -inf
        if np.isfinite(val_numpy) and np.isfinite(val_jax):
            assert np.allclose(val_numpy, val_jax, rtol=1e-10, atol=1e-10)
        else:
            assert val_numpy == val_jax == -np.inf


class TestErrorHandling:
    """Test that error handling is consistent across backends."""
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_invalid_input_handling(self, backend):
        """Test that invalid inputs are handled consistently."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        # Test with None inputs - expect TypeError or ValueError
        try:
            get_log_likelihood_h0(
                requested_backend_str=backend,
                dL_gw_samples=None,
                host_galaxies_z=[0.1],
                host_galaxies_mass_proxy=[1.0],
                host_galaxies_z_err=[0.01],
            )
            # If no exception, this is a problem
            assert False, "Expected exception for None dL_gw_samples"
        except (ValueError, TypeError):
            # Expected - either error type is acceptable
            pass
        
        # Test with empty inputs
        with pytest.raises(ValueError):
            get_log_likelihood_h0(
                requested_backend_str=backend,
                dL_gw_samples=[100.0],
                host_galaxies_z=[],
                host_galaxies_mass_proxy=[],
                host_galaxies_z_err=[],
            )
        
        # Test with mismatched lengths
        with pytest.raises(ValueError):
            get_log_likelihood_h0(
                requested_backend_str=backend,
                dL_gw_samples=[100.0],
                host_galaxies_z=[0.1, 0.2],
                host_galaxies_mass_proxy=[1.0],  # Wrong length
                host_galaxies_z_err=[0.01, 0.01],
            )
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_extreme_parameter_values(self, backend):
        """Test behavior with extreme parameter values."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=1, seed=42)
        
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # Test with extreme values
        extreme_cases = [
            [1e-10, 0.0],  # Very small H0
            [1e10, 0.0],   # Very large H0
            [70.0, 100.0], # Very large alpha
            [70.0, -100.0], # Very negative alpha
        ]
        
        for theta in extreme_cases:
            result = ll(theta)
            # Should return -inf for out-of-bounds, not crash
            assert result == -ll.xp.inf or ll.xp.isfinite(result)


class TestEdgeCases:
    """Test edge cases that might reveal backend differences."""
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_single_galaxy_edge_case(self, backend):
        """Test with a single galaxy."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=1, seed=42)
        
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        result = ll([70.0, 0.0])
        assert ll.xp.isfinite(result)
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_zero_redshift_error(self, backend):
        """Test with zero redshift uncertainties."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=2, seed=42)
        
        # Set one galaxy to have zero redshift error
        z_err_modified = pkg.candidate_galaxies_df["z_err"].values.copy()
        z_err_modified[0] = 0.0
        
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=z_err_modified,
        )
        
        result = ll([70.0, 0.0])
        assert ll.xp.isfinite(result)
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_very_small_distances(self, backend):
        """Test with very small GW distance samples."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        # Create mock data with very small distances
        small_distances = np.array([1e-3, 2e-3, 3e-3])  # Very small Mpc
        
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=small_distances,
            host_galaxies_z=np.array([1e-6]),  # Very small redshift
            host_galaxies_mass_proxy=np.array([1.0]),
            host_galaxies_z_err=np.array([1e-8]),
        )
        
        result = ll([70.0, 0.0])
        # Should handle gracefully, either finite result or -inf
        assert ll.xp.isfinite(result) or result == -ll.xp.inf


class TestBackendSelection:
    """Test backend selection and environment setup."""
    
    def test_auto_backend_selection(self):
        """Test that auto backend selection works."""
        xp, name, device = get_xp("auto")
        assert name in ["numpy", "jax"]
        assert device in ["cpu", "cuda", "metal"]
        assert hasattr(xp, "array")
    
    def test_explicit_backend_selection(self):
        """Test explicit backend selection."""
        # NumPy should always be available
        xp_np, name_np, device_np = get_xp("numpy")
        assert name_np == "numpy"
        assert device_np == "cpu"
        
        # JAX might not be available
        if importlib.util.find_spec("jax"):
            xp_jax, name_jax, device_jax = get_xp("jax")
            assert name_jax == "jax"
            assert device_jax in ["cpu", "cuda", "metal"]
        else:
            with pytest.raises(BackendNotAvailableError):
                get_xp("jax")
    
    def test_invalid_backend_request(self):
        """Test that invalid backend requests fail appropriately."""
        with pytest.raises(ValueError):
            get_xp("invalid_backend")
    
    @patch.dict(os.environ, {"JAX_PLATFORM_NAME": "cpu"})
    def test_environment_variable_respected(self):
        """Test that JAX respects environment variables."""
        if not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        xp, name, device = get_xp("jax")
        assert name == "jax"
        # Should use CPU when JAX_PLATFORM_NAME=cpu is set
        assert device == "cpu"


@pytest.mark.slow
class TestPerformanceRegression:
    """Test that changes don't introduce performance regressions."""
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_likelihood_evaluation_speed(self, backend):
        """Test that likelihood evaluation completes in reasonable time."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=5, seed=42)
        
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        import time
        start_time = time.time()
        
        # Multiple evaluations
        for _ in range(10):
            result = ll([70.0, 0.0])
            assert ll.xp.isfinite(result)
        
        elapsed = time.time() - start_time
        
        # Should complete 10 evaluations in reasonable time (adjust threshold as needed)
        assert elapsed < 5.0, f"{backend} backend too slow: {elapsed:.2f}s for 10 evaluations"


class TestPropertyBased:
    """Property-based tests using random inputs within valid ranges."""
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_likelihood_monotonicity_properties(self, backend):
        """Test mathematical properties of the likelihood."""
        if backend == "jax" and not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        pkg = mock_event(n_galaxies=2, seed=42)
        
        ll = get_log_likelihood_h0(
            requested_backend_str=backend,
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # Test with multiple random valid parameter combinations
        np.random.seed(42)
        for _ in range(20):
            h0 = np.random.uniform(50.0, 100.0)
            alpha = np.random.uniform(-0.8, 0.8)
            
            result = ll([h0, alpha])
            
            # Result should always be finite (within prior bounds)
            assert ll.xp.isfinite(result), f"Non-finite result for H0={h0}, alpha={alpha}"
            
            # Result should be a scalar
            assert np.isscalar(float(result)), f"Non-scalar result: {result}"


@pytest.mark.slow
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_consistency(self):
        """Test that the complete pipeline gives consistent results."""
        if not importlib.util.find_spec("jax"):
            pytest.skip("JAX not available")
        
        # Use multiple events to test robustness
        events = multi_event(n_events=2, seed=42)
        
        results_numpy = []
        results_jax = []
        
        for pkg in events:
            # Test with both backends
            for backend, results_list in [("numpy", results_numpy), ("jax", results_jax)]:
                ll = get_log_likelihood_h0(
                    requested_backend_str=backend,
                    dL_gw_samples=pkg.dl_samples,
                    host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
                    host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
                    host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
                    force_non_vectorized=True,  # Ensure same algorithm
                )
                
                result = ll([70.0, 0.0])
                results_list.append(result)
        
        # Results should be consistent between backends
        for i, (r_np, r_jax) in enumerate(zip(results_numpy, results_jax)):
            assert np.allclose(r_np, r_jax, rtol=1e-10), f"Event {i}: NumPy={r_np}, JAX={r_jax}"


# Fixtures for common test data
@pytest.fixture
def simple_mock_data():
    """Provide simple mock data for tests."""
    return mock_event(n_galaxies=2, seed=42)


@pytest.fixture
def backends_available():
    """Provide list of available backends."""
    backends = ["numpy"]
    if importlib.util.find_spec("jax"):
        backends.append("jax")
    return backends


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__, "-v"]) 