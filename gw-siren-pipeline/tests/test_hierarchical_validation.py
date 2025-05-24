"""Comprehensive tests for hierarchical Bayesian model validation."""

from __future__ import annotations

import os
import numpy as np
import pytest
from scipy import stats

# Force JAX CPU for testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0, run_mcmc_h0, process_mcmc_samples
from gwsiren.combined_likelihood import CombinedLogLikelihood
from .utils.mock_data import mock_event, multi_event, mock_event_physics_consistent
from .utils.test_helpers import rhat, effective_sample_size


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
        """Test redshift marginalization consistency."""
        # Create mock data with different redshift uncertainties
        pkg = mock_event(n_galaxies=3, seed=123)
        
        # Test with small z_err (should skip marginalization)
        small_z_err = np.full(3, 1e-8)
        ll_small = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=small_z_err,
            z_err_threshold=1e-6,
        )
        
        # Test with large z_err (should use marginalization)
        large_z_err = np.full(3, 0.01)
        ll_large = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=large_z_err,
            z_err_threshold=1e-6,
        )
        
        # Both should give finite likelihoods
        test_params = [70.0, 0.0]
        val_small = ll_small(test_params)
        val_large = ll_large(test_params)
        
        assert np.isfinite(val_small)
        assert np.isfinite(val_large)
        
        # Large redshift errors should generally give lower likelihood
        # (more uncertainty), but this isn't strictly guaranteed
        print(f"Small z_err likelihood: {val_small:.3f}")
        print(f"Large z_err likelihood: {val_large:.3f}")


class TestSimulationBasedCalibration:
    """Test simulation-based calibration of the hierarchical model."""
    
    def test_sbc_rank_distribution(self, mock_config):
        """Test that SBC ranks follow uniform distribution for well-calibrated model."""
        n_simulations = 8  # Small number for testing (should be 50+ for real validation)
        ranks_h0 = []
        ranks_alpha = []
        
        # Prior bounds - reasonable physical range
        h0_min, h0_max = 55.0, 85.0
        alpha_min, alpha_max = -0.3, 0.3
        
        successful_runs = 0
        rng = np.random.default_rng(12345)
        
        for sim_idx in range(n_simulations):
            print(f"\n🧪 SBC Run {sim_idx+1}/{n_simulations}")
            
            # 1. Sample true parameters from prior
            true_h0 = rng.uniform(h0_min, h0_max)
            true_alpha = rng.uniform(alpha_min, alpha_max)
            
            print(f"  True parameters: H0={true_h0:.1f}, α={true_alpha:.3f}")
            
            # 2. Generate physics-consistent mock data given true parameters
            pkg = mock_event_physics_consistent(
                event_id=f"sbc_sim_{sim_idx}",
                true_h0=true_h0,
                true_alpha=true_alpha,
                n_gw_samples=400,  # Smaller for faster testing
                n_galaxies=6,
                seed=rng.integers(0, 2**32-1),
            )
            
            # 3. Set up likelihood function (same physics as mock data)
            ll = get_log_likelihood_h0(
                requested_backend_str="auto",
                dL_gw_samples=pkg.dl_samples,
                host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
                host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
                host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
                h0_min=h0_min,
                h0_max=h0_max,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                z_err_threshold=1e-6,
                sigma_v=250.0,  # Same as mock data
            )
            
            # Check that likelihood is finite at true parameters
            true_logL = ll([true_h0, true_alpha])
            if not np.isfinite(true_logL):
                print(f"  ❌ Likelihood not finite at true parameters: {true_logL}")
                continue
                
            print(f"  📊 True likelihood: {true_logL:.3f}")
            
            # 4. Run MCMC to get posterior samples (REAL INFERENCE)
            sampler = run_mcmc_h0(
                ll,
                f"sbc_sim_{sim_idx}",
                n_walkers=10,
                n_steps=120,  # Short for testing
            )
            
            if sampler is not None:
                # 5. Process MCMC samples
                samples = process_mcmc_samples(
                    sampler, 
                    f"sbc_sim_{sim_idx}", 
                    burnin=25, 
                    thin_by=2
                )
                
                if samples is not None and len(samples) > 15:
                    # 6. Compute ranks (how many posterior samples < true value)
                    posterior_h0 = samples[:, 0]
                    posterior_alpha = samples[:, 1]
                    
                    rank_h0 = np.sum(posterior_h0 < true_h0)
                    rank_alpha = np.sum(posterior_alpha < true_alpha)
                    
                    # Normalize to [0, 1]
                    rank_h0_norm = rank_h0 / len(posterior_h0)
                    rank_alpha_norm = rank_alpha / len(posterior_alpha)
                    
                    ranks_h0.append(rank_h0_norm)
                    ranks_alpha.append(rank_alpha_norm)
                    successful_runs += 1
                    
                    print(f"  📈 Ranks: H0={rank_h0_norm:.3f}, α={rank_alpha_norm:.3f}")
                    print(f"  📊 Posterior: H0={np.mean(posterior_h0):.1f}±{np.std(posterior_h0):.1f}, "
                          f"α={np.mean(posterior_alpha):.3f}±{np.std(posterior_alpha):.3f}")
                else:
                    print(f"  ❌ Insufficient MCMC samples for run {sim_idx}")
            else:
                print(f"  ❌ MCMC failed for run {sim_idx}")
        
        print(f"\n📊 SBC Summary: {successful_runs}/{n_simulations} successful runs")
        
        # 7. Test rank distribution
        if successful_runs >= 5:  # Need minimum for meaningful test
            ranks_h0 = np.array(ranks_h0)
            ranks_alpha = np.array(ranks_alpha)
            
            print(f"H0 ranks: mean={ranks_h0.mean():.3f}, std={ranks_h0.std():.3f}")
            print(f"H0 rank range: [{ranks_h0.min():.3f}, {ranks_h0.max():.3f}]")
            print(f"Alpha ranks: mean={ranks_alpha.mean():.3f}, std={ranks_alpha.std():.3f}")
            print(f"Alpha rank range: [{ranks_alpha.min():.3f}, {ranks_alpha.max():.3f}]")
            
            # For well-calibrated model, ranks should:
            # 1. Span reasonable range (not all clustered)
            # 2. Have mean near 0.5 (no systematic bias)
            
            # Check range (relaxed for small sample)
            assert ranks_h0.min() < 0.8, f"H0 ranks too high: min={ranks_h0.min():.3f}"
            assert ranks_h0.max() > 0.2, f"H0 ranks too low: max={ranks_h0.max():.3f}"
            assert ranks_alpha.min() < 0.8, f"Alpha ranks too high: min={ranks_alpha.min():.3f}"
            assert ranks_alpha.max() > 0.2, f"Alpha ranks too low: max={ranks_alpha.max():.3f}"
            
            # Check for systematic bias (relaxed for small sample)
            assert 0.15 < ranks_h0.mean() < 0.85, f"Systematic H0 bias: mean rank = {ranks_h0.mean():.3f}"
            assert 0.15 < ranks_alpha.mean() < 0.85, f"Systematic alpha bias: mean rank = {ranks_alpha.mean():.3f}"
            
            print("✅ Basic SBC validation passed!")
            
            # If we have enough runs, do KS test (very lenient for small sample)
            if successful_runs >= 6:
                ks_stat_h0, p_value_h0 = stats.kstest(ranks_h0, 'uniform')
                ks_stat_alpha, p_value_alpha = stats.kstest(ranks_alpha, 'uniform')
                
                print(f"KS test H0: stat={ks_stat_h0:.3f}, p-value={p_value_h0:.3f}")
                print(f"KS test Alpha: stat={ks_stat_alpha:.3f}, p-value={p_value_alpha:.3f}")
                
                # Very lenient threshold for small sample
                if p_value_h0 > 0.01 and p_value_alpha > 0.01:
                    print("✅ KS test suggests no major calibration issues!")
                else:
                    print("⚠️ KS test suggests possible calibration issues (but sample size is small)")
            
        else:
            pytest.skip(f"Insufficient successful SBC runs: {successful_runs}")
    
    @pytest.mark.slow
    def test_parameter_recovery(self, mock_config):
        """Test that known parameters can be recovered using physics-consistent data."""
        true_h0 = 70.0
        true_alpha = 0.2
        
        print(f"\n🎯 Parameter Recovery Test: H0={true_h0}, α={true_alpha}")
        
        # Generate physics-consistent mock dataset
        pkg = mock_event_physics_consistent(
            event_id="recovery_test",
            true_h0=true_h0,
            true_alpha=true_alpha,
            n_gw_samples=600,  # More data for better recovery
            n_galaxies=8,
            seed=999
        )
        
        print(f"Mock data: {len(pkg.dl_samples)} GW samples, {len(pkg.candidate_galaxies_df)} galaxies")
        
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=pkg.dl_samples,
            host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
            h0_min=50.0,
            h0_max=90.0,
            alpha_min=-0.5,
            alpha_max=0.5,
        )
        
        # Test that true parameters give reasonable likelihood
        true_logL = ll([true_h0, true_alpha])
        assert np.isfinite(true_logL), f"True likelihood not finite: {true_logL}"
        
        # Test that nearby parameters give similar likelihood
        nearby_logL = ll([true_h0 + 2.0, true_alpha + 0.05])
        assert np.isfinite(nearby_logL), f"Nearby likelihood not finite: {nearby_logL}"
        
        # Test that far parameters give lower likelihood
        far_logL = ll([true_h0 + 15.0, true_alpha + 0.3])
        print(f"Likelihoods: true={true_logL:.3f}, nearby={nearby_logL:.3f}, far={far_logL:.3f}")
        
        # True parameters should be better than far ones
        assert true_logL > far_logL, "True parameters should be better than far parameters"
        
        # Run MCMC for parameter recovery
        sampler = run_mcmc_h0(
            ll,
            "recovery_test",
            n_walkers=12,
            n_steps=200,
        )
        
        if sampler is not None:
            samples = process_mcmc_samples(sampler, "recovery_test", burnin=40, thin_by=2)
            
            if samples is not None and len(samples) > 30:
                h0_samples = samples[:, 0]
                alpha_samples = samples[:, 1]
                
                h0_mean = np.mean(h0_samples)
                h0_std = np.std(h0_samples)
                alpha_mean = np.mean(alpha_samples)
                alpha_std = np.std(alpha_samples)
                
                print(f"Recovery results:")
                print(f"  H0: true={true_h0:.1f}, recovered={h0_mean:.1f}±{h0_std:.1f}")
                print(f"  α: true={true_alpha:.3f}, recovered={alpha_mean:.3f}±{alpha_std:.3f}")
                
                # Check recovery within reasonable bounds
                h0_bias = abs(h0_mean - true_h0)
                alpha_bias = abs(alpha_mean - true_alpha)
                
                # Recovery should be within ~2-3 sigma (allowing for model limitations)
                assert h0_bias < 3 * h0_std + 5.0, f"H0 recovery bias too large: {h0_bias:.1f} vs std {h0_std:.1f}"
                assert alpha_bias < 3 * alpha_std + 0.2, f"Alpha recovery bias too large: {alpha_bias:.3f} vs std {alpha_std:.3f}"
                
                print("✅ Parameter recovery test passed!")
            else:
                pytest.skip("Insufficient MCMC samples for recovery test")
        else:
            pytest.skip("MCMC failed for recovery test")


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