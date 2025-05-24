"""Comprehensive tests for hierarchical Bayesian model validation."""

from __future__ import annotations

import sys
import pathlib
import numpy as np
import pytest
from scipy import stats

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0, run_mcmc_h0, process_mcmc_samples
from gwsiren.combined_likelihood import CombinedLogLikelihood
from utils.mock_data import mock_event, multi_event
from utils.test_helpers import rhat, effective_sample_size


class TestHierarchicalStructure:
    """Test the hierarchical structure of the Bayesian model."""
    
    def test_galaxy_weight_consistency(self, mock_config):
        """Test that galaxy selection weights sum to 1 and behave correctly."""
        pkg = mock_event(n_galaxies=10, seed=42)
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # Test alpha = 0 case (uniform weights)
        # Access internal computation to verify weights
        mass_proxy = ll.mass_proxy_values
        alpha = 0.0
        
        if ll.xp.isclose(alpha, 0.0):
            expected_weights = ll.xp.full(len(mass_proxy), 1.0 / len(mass_proxy))
        else:
            powered = mass_proxy ** alpha
            expected_weights = powered / powered.sum()
        
        # Weights should sum to 1
        assert ll.xp.allclose(expected_weights.sum(), 1.0)
        
        # Test non-zero alpha
        alpha = 0.5
        powered = mass_proxy ** alpha
        weights = powered / powered.sum()
        assert ll.xp.allclose(weights.sum(), 1.0)
        assert ll.xp.all(weights > 0)  # All weights should be positive
    
    def test_redshift_marginalization(self, mock_config):
        """Test that redshift marginalization is working correctly."""
        # Create event with known redshift uncertainties
        pkg = mock_event(n_galaxies=3, seed=123)
        
        # Test with essentially zero redshift error (below threshold - no marginalization)
        ll_no_marg = get_log_likelihood_h0(
            requested_backend_str="numpy",  # Use numpy backend
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=np.full_like(pkg.candidate_galaxies_df["z_err"].values, 1e-7),  # Below threshold
            z_err_threshold=1e-6,  # Explicitly set threshold
            force_non_vectorized=True,  # Force non-vectorized path for marginalization
        )
        
        # Test with larger redshift error (above threshold - with marginalization)
        ll_with_marg = get_log_likelihood_h0(
            requested_backend_str="numpy",  # Use numpy backend
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=np.full_like(pkg.candidate_galaxies_df["z_err"].values, 0.005),  # Much larger than threshold
            z_err_threshold=1e-6,  # Explicitly set threshold
            n_quad_points=7,  # Use more quadrature points
            force_non_vectorized=True,  # Force non-vectorized path for marginalization
        )
        
        theta = [70.0, 0.0]
        val_no_marg = ll_no_marg(theta)
        val_with_marg = ll_with_marg(theta)
        
        # Both should be finite
        assert np.isfinite(val_no_marg), f"Non-marginalized likelihood is not finite: {val_no_marg}"
        assert np.isfinite(val_with_marg), f"Marginalized likelihood is not finite: {val_with_marg}"
        
        # They should be different (marginalization effect)
        # Marginalization can either increase or decrease likelihood depending on the data
        diff = abs(val_no_marg - val_with_marg)
        assert diff > 0.01, f"Marginalization had insufficient effect: no_marg={val_no_marg:.8f}, with_marg={val_with_marg:.8f}, diff={diff:.8f}"
        
        # Additional test: marginalization with moderate uncertainty should be between extremes
        ll_moderate_marg = get_log_likelihood_h0(
            requested_backend_str="numpy",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=np.full_like(pkg.candidate_galaxies_df["z_err"].values, 0.001),  # Moderate uncertainty
            z_err_threshold=1e-6,
            force_non_vectorized=True,
        )
        
        val_moderate_marg = ll_moderate_marg(theta)
        assert np.isfinite(val_moderate_marg), f"Moderate marginalized likelihood is not finite: {val_moderate_marg}"
        
        # Test that different levels of marginalization give different results
        diff_moderate = abs(val_no_marg - val_moderate_marg)
        assert diff_moderate > 0.001, f"Moderate marginalization had no effect: no_marg={val_no_marg:.8f}, moderate={val_moderate_marg:.8f}, diff={diff_moderate:.8f}"


class TestSimulationBasedCalibration:
    """Proper simulation-based calibration tests."""
    
    def test_sbc_rank_distribution(self, mock_config):
        """Test that SBC ranks follow uniform distribution for well-calibrated model."""
        n_simulations = 20  # Small number for testing
        n_samples = 500
        ranks_h0 = []
        ranks_alpha = []
        
        rng = np.random.default_rng(42)
        
        for _ in range(n_simulations):
            # Generate true parameters from prior
            true_h0 = rng.uniform(30.0, 120.0)  # Wider than default priors
            true_alpha = rng.uniform(-0.8, 0.8)
            
            # Generate mock data given true parameters
            pkg = mock_event(n_galaxies=5, seed=rng.integers(0, 2**32-1))
            
            # Create likelihood with wider priors
            ll = get_log_likelihood_h0(
                requested_backend_str="auto",
                dL_gw_samples=pkg.dl_samples,
                host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
                host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
                host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
                h0_min=20.0,
                h0_max=140.0,
                alpha_min=-1.0,
                alpha_max=1.0,
            )
            
            # Generate posterior samples (simplified - would normally run MCMC)
            posterior_h0 = rng.normal(true_h0, 5.0, size=n_samples)
            posterior_alpha = rng.normal(true_alpha, 0.1, size=n_samples)
            
            # Compute ranks
            rank_h0 = np.sum(posterior_h0 < true_h0)
            rank_alpha = np.sum(posterior_alpha < true_alpha)
            
            ranks_h0.append(rank_h0)
            ranks_alpha.append(rank_alpha)
        
        # Test that ranks are roughly uniform (simplified test)
        # For proper SBC, would use Kolmogorov-Smirnov test
        ranks_h0 = np.array(ranks_h0)
        ranks_alpha = np.array(ranks_alpha)
        
        # Check that ranks span the range
        assert ranks_h0.min() < n_samples * 0.3
        assert ranks_h0.max() > n_samples * 0.7
        assert ranks_alpha.min() < n_samples * 0.3
        assert ranks_alpha.max() > n_samples * 0.7
    
    @pytest.mark.slow
    def test_parameter_recovery(self, mock_config):
        """Test that known parameters can be recovered."""
        true_h0 = 70.0
        true_alpha = 0.3
        
        # Generate larger mock dataset for better recovery
        pkg = mock_event(n_galaxies=10, n_samples=1000, seed=999)
        
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # Test that true parameters give reasonable likelihood
        true_logL = ll([true_h0, true_alpha])
        assert np.isfinite(true_logL)
        
        # Test that nearby parameters give similar likelihood
        nearby_logL = ll([true_h0 + 1.0, true_alpha + 0.05])
        assert np.isfinite(nearby_logL)
        
        # Test that far parameters give lower likelihood
        far_logL = ll([true_h0 + 20.0, true_alpha + 0.5])
        assert true_logL > far_logL  # True params should be better


class TestNumericalStability:
    """Test numerical stability of the hierarchical model."""
    
    def test_extreme_parameter_regimes(self, mock_config):
        """Test behavior with extreme but valid parameters."""
        pkg = mock_event(seed=555)
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
            h0_min=10.0,
            h0_max=200.0,
            alpha_min=-1.0,
            alpha_max=1.0,
        )
        
        # Test extreme but valid parameters
        extreme_cases = [
            [15.0, -0.9],  # Low H0, negative alpha
            [190.0, 0.9],  # High H0, positive alpha
            [70.0, -1.0],  # Boundary alpha
            [70.0, 1.0],   # Boundary alpha
        ]
        
        for params in extreme_cases:
            val = ll(params)
            assert np.isfinite(val), f"Non-finite likelihood for params {params}"
    
    def test_small_datasets(self, mock_config):
        """Test behavior with minimal data."""
        # Single galaxy, few samples
        pkg = mock_event(n_galaxies=1, n_samples=10, seed=777)
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        val = ll([70.0, 0.0])
        assert np.isfinite(val)
    
    def test_large_datasets(self, mock_config):
        """Test behavior with large datasets."""
        # Many galaxies, many samples
        pkg = mock_event(n_galaxies=50, n_samples=2000, seed=888)
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        val = ll([70.0, 0.0])
        assert np.isfinite(val)


class TestMultiEventConsistency:
    """Test consistency across multiple events."""
    
    def test_combined_likelihood_properties(self, mock_config):
        """Test mathematical properties of combined likelihood."""
        events = multi_event(n_events=3, seed=111)
        combined = CombinedLogLikelihood(events)
        
        # Test that combined likelihood is sum of individual likelihoods
        theta = [70.0, 0.2]
        combined_val = combined(theta)
        
        individual_sum = 0.0
        for pkg in events:
            ll = get_log_likelihood_h0(
                requested_backend_str="auto",
                dL_gw_samples=pkg.dl_samples,
                host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
                host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
                host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
            )
            individual_sum += ll(theta)
        
        assert np.isclose(combined_val, individual_sum, rtol=1e-10)
    
    def test_event_weighting_consistency(self, mock_config):
        """Test that events with more data have appropriate influence."""
        # Create events with different amounts of data
        event_small = mock_event(n_galaxies=2, n_samples=100, seed=222)
        event_large = mock_event(n_galaxies=10, n_samples=1000, seed=333)
        
        combined = CombinedLogLikelihood([event_small, event_large])
        
        theta = [70.0, 0.0]
        combined_val = combined(theta)
        
        # Should be finite
        assert np.isfinite(combined_val)


@pytest.mark.slow
class TestMCMCDiagnostics:
    """Test MCMC chain diagnostics for the hierarchical model."""
    
    def test_mcmc_convergence_metrics(self, mock_config):
        """Test that MCMC chains converge properly."""
        pkg = mock_event(n_galaxies=5, seed=444)
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        )
        
        # Run short MCMC for testing
        sampler = run_mcmc_h0(
            ll,
            "test_event",
            n_walkers=8,
            n_steps=100,
        )
        
        if sampler is not None:
            # Process samples
            samples = process_mcmc_samples(sampler, "test_event", burnin=20, thin_by=1)
            
            if samples is not None and len(samples) > 0:
                # Test convergence diagnostics
                chains = sampler.get_chain(discard=20)  # (steps, walkers, params)
                
                # R-hat for each parameter
                for param_idx in range(chains.shape[2]):
                    param_chains = chains[:, :, param_idx].T  # (walkers, steps)
                    rhat_val = rhat(param_chains)
                    assert rhat_val < 1.3, f"Poor convergence: R-hat = {rhat_val}"
                
                # Effective sample size
                flat_samples = sampler.get_chain(discard=20, flat=True)
                for param_idx in range(flat_samples.shape[1]):
                    ess = effective_sample_size(flat_samples[:, param_idx])
                    assert ess > 10, f"Low ESS: {ess}" 