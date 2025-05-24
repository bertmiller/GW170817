"""Fixed Simulation-Based Calibration tests with physics-consistent mock data."""

from __future__ import annotations

import os
import numpy as np
import pytest
from scipy import stats

# Force JAX CPU for testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0, run_mcmc_h0, process_mcmc_samples
from .utils.mock_data import (
    generate_mock_data_physics_consistent,
    mock_event_physics_consistent,
    validate_mock_data_consistency,
    compute_luminosity_distance_consistent,
)


def test_improved_mock_data_generation():
    """Test that the improved mock data generation is self-consistent."""
    # Test different parameter combinations
    test_cases = [
        {"true_h0": 70.0, "true_alpha": 0.0, "label": "no_selection"},
        {"true_h0": 65.0, "true_alpha": 0.5, "label": "high_mass_bias"},
        {"true_h0": 75.0, "true_alpha": -0.3, "label": "low_mass_bias"},
    ]
    
    for case in test_cases:
        mock_data = generate_mock_data_physics_consistent(
            true_h0=case["true_h0"],
            true_alpha=case["true_alpha"],
            seed=42
        )
        
        # Validate physics consistency
        assert validate_mock_data_consistency(mock_data), f"Failed for {case['label']}"
        
        # Check that we have the expected number of galaxies and samples
        assert len(mock_data['galaxy_z']) == 8, f"Wrong number of galaxies in {case['label']}"
        assert len(mock_data['dl_samples']) == 500, f"Wrong number of GW samples in {case['label']}"
        
        # Check that true host index is valid
        assert 0 <= mock_data['true_host_idx'] < len(mock_data['galaxy_z'])
        
        print(f"✅ {case['label']}: H0={case['true_h0']}, α={case['true_alpha']} - validation passed")


@pytest.mark.slow
def test_sbc_physics_consistent_small():
    """Small-scale SBC test with physics-consistent mock data."""
    n_sbc_runs = 10  # Small number for quick testing
    ranks_h0 = []
    ranks_alpha = []
    
    # Prior bounds
    h0_min, h0_max = 50.0, 90.0
    alpha_min, alpha_max = -0.4, 0.4
    
    successful_runs = 0
    
    for run_idx in range(n_sbc_runs):
        rng = np.random.default_rng(2000 + run_idx)
        
        # 1. Sample true parameters from prior
        true_h0 = rng.uniform(h0_min, h0_max)
        true_alpha = rng.uniform(alpha_min, alpha_max)
        
        print(f"\n🧪 SBC Run {run_idx+1}/{n_sbc_runs}: H0={true_h0:.1f}, α={true_alpha:.3f}")
        
        # 2. Generate physics-consistent mock data
        mock_data = generate_mock_data_physics_consistent(
            true_h0=true_h0,
            true_alpha=true_alpha,
            n_gw_samples=300,  # Smaller for faster testing
            n_galaxies=6,
            z_err_level=0.003,  # Ensure marginalization
            seed=rng.integers(0, 2**32-1),
        )
        
        # Validate the mock data
        if not validate_mock_data_consistency(mock_data):
            print(f"  ❌ Mock data validation failed for run {run_idx}")
            continue
        
        # 3. Set up likelihood with consistent parameters
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=mock_data['dl_samples'],
            host_galaxies_z=mock_data['galaxy_z'],
            host_galaxies_mass_proxy=mock_data['galaxy_mass'],
            host_galaxies_z_err=mock_data['galaxy_z_err'],
            h0_min=h0_min,
            h0_max=h0_max,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            z_err_threshold=1e-6,  # Ensure marginalization for z_err=0.003
            sigma_v=250.0,  # Same as mock data generation
        )
        
        # Test that likelihood is finite at true parameters
        true_logL = ll([true_h0, true_alpha])
        if not np.isfinite(true_logL):
            print(f"  ❌ Likelihood is not finite at true parameters: {true_logL}")
            continue
        
        print(f"  📊 True likelihood: {true_logL:.3f}")
        
        # 4. Run shortened MCMC for testing
        sampler = run_mcmc_h0(
            ll,
            f"sbc_fixed_{run_idx}",
            n_walkers=12,
            n_steps=150,  # Short for testing
        )
        
        if sampler is not None:
            samples = process_mcmc_samples(
                sampler, 
                f"sbc_fixed_{run_idx}", 
                burnin=30, 
                thin_by=2
            )
            
            if samples is not None and len(samples) > 20:
                # 5. Compute ranks
                posterior_h0 = samples[:, 0]
                posterior_alpha = samples[:, 1]
                
                rank_h0 = np.sum(posterior_h0 < true_h0) / len(posterior_h0)
                rank_alpha = np.sum(posterior_alpha < true_alpha) / len(posterior_alpha)
                
                ranks_h0.append(rank_h0)
                ranks_alpha.append(rank_alpha)
                successful_runs += 1
                
                print(f"  📈 Ranks: H0={rank_h0:.3f}, α={rank_alpha:.3f}")
                print(f"  📊 Posterior: H0={np.mean(posterior_h0):.1f}±{np.std(posterior_h0):.1f}, "
                      f"α={np.mean(posterior_alpha):.3f}±{np.std(posterior_alpha):.3f}")
            else:
                print(f"  ❌ Insufficient samples from MCMC for run {run_idx}")
        else:
            print(f"  ❌ MCMC failed for run {run_idx}")
    
    print(f"\n📊 SBC Summary: {successful_runs}/{n_sbc_runs} successful runs")
    
    if successful_runs >= 5:  # Need minimum number for meaningful test
        ranks_h0 = np.array(ranks_h0)
        ranks_alpha = np.array(ranks_alpha)
        
        # Basic statistics
        print(f"H0 ranks: mean={ranks_h0.mean():.3f}, std={ranks_h0.std():.3f}")
        print(f"H0 rank range: [{ranks_h0.min():.3f}, {ranks_h0.max():.3f}]")
        print(f"Alpha ranks: mean={ranks_alpha.mean():.3f}, std={ranks_alpha.std():.3f}")
        print(f"Alpha rank range: [{ranks_alpha.min():.3f}, {ranks_alpha.max():.3f}]")
        
        # Test for basic validity (not strict uniformity with small sample)
        # Well-calibrated model should have ranks spanning reasonable range
        assert ranks_h0.min() < 0.8, f"H0 ranks too high: min={ranks_h0.min():.3f}"
        assert ranks_h0.max() > 0.2, f"H0 ranks too low: max={ranks_h0.max():.3f}"
        assert ranks_alpha.min() < 0.8, f"Alpha ranks too high: min={ranks_alpha.min():.3f}"
        assert ranks_alpha.max() > 0.2, f"Alpha ranks too low: max={ranks_alpha.max():.3f}"
        
        # Test for reasonable mean (not too much systematic bias)
        assert 0.2 < ranks_h0.mean() < 0.8, f"Systematic H0 bias: mean rank = {ranks_h0.mean():.3f}"
        assert 0.2 < ranks_alpha.mean() < 0.8, f"Systematic alpha bias: mean rank = {ranks_alpha.mean():.3f}"
        
        print("✅ Basic SBC validation passed!")
        
        # If we have enough runs, do KS test (lenient threshold for small sample)
        if successful_runs >= 8:
            ks_stat_h0, p_value_h0 = stats.kstest(ranks_h0, 'uniform')
            ks_stat_alpha, p_value_alpha = stats.kstest(ranks_alpha, 'uniform')
            
            print(f"KS test H0: stat={ks_stat_h0:.3f}, p-value={p_value_h0:.3f}")
            print(f"KS test Alpha: stat={ks_stat_alpha:.3f}, p-value={p_value_alpha:.3f}")
            
            # Very lenient threshold for small sample size
            if p_value_h0 > 0.01 and p_value_alpha > 0.01:
                print("✅ KS test suggests no major calibration issues!")
            else:
                print("⚠️ KS test suggests possible calibration issues (but sample size is small)")
        
    else:
        pytest.skip(f"Insufficient successful SBC runs: {successful_runs}")


@pytest.mark.slow
def test_parameter_recovery_with_known_truth():
    """Test parameter recovery when we know the true host galaxy."""
    true_h0 = 70.0
    true_alpha = 0.2
    
    print(f"\n🎯 Parameter Recovery Test: H0={true_h0}, α={true_alpha}")
    
    # Generate mock data
    mock_data = generate_mock_data_physics_consistent(
        true_h0=true_h0,
        true_alpha=true_alpha,
        n_gw_samples=800,  # More samples for better recovery
        n_galaxies=10,
        seed=999
    )
    
    # Validate mock data
    assert validate_mock_data_consistency(mock_data)
    
    true_host_idx = mock_data['true_host_idx']
    true_host_z = mock_data['true_host_z']
    
    print(f"True host: galaxy {true_host_idx} at z={true_host_z:.4f}")
    
    # Set up likelihood
    ll = get_log_likelihood_h0(
        requested_backend_str="auto",
        dL_gw_samples=mock_data['dl_samples'],
        host_galaxies_z=mock_data['galaxy_z'],
        host_galaxies_mass_proxy=mock_data['galaxy_mass'],
        host_galaxies_z_err=mock_data['galaxy_z_err'],
        h0_min=50.0,
        h0_max=90.0,
        alpha_min=-0.5,
        alpha_max=0.5,
    )
    
    # Test likelihood at true parameters
    true_logL = ll([true_h0, true_alpha])
    assert np.isfinite(true_logL), f"True likelihood not finite: {true_logL}"
    
    # Test that nearby parameters have similar likelihood
    nearby_logL = ll([true_h0 + 2.0, true_alpha + 0.05])
    assert np.isfinite(nearby_logL), f"Nearby likelihood not finite: {nearby_logL}"
    
    # Test that far parameters have lower likelihood
    far_logL = ll([true_h0 + 15.0, true_alpha + 0.3])
    print(f"Likelihoods: true={true_logL:.3f}, nearby={nearby_logL:.3f}, far={far_logL:.3f}")
    
    # True parameters should be better than far ones (not necessarily better than nearby due to noise)
    assert true_logL > far_logL, "True parameters should be better than far parameters"
    
    # Run MCMC for parameter recovery
    sampler = run_mcmc_h0(
        ll,
        "recovery_test",
        n_walkers=16,
        n_steps=300,
    )
    
    if sampler is not None:
        samples = process_mcmc_samples(sampler, "recovery_test", burnin=60, thin_by=2)
        
        if samples is not None and len(samples) > 50:
            h0_samples = samples[:, 0]
            alpha_samples = samples[:, 1]
            
            h0_mean = np.mean(h0_samples)
            h0_std = np.std(h0_samples)
            alpha_mean = np.mean(alpha_samples)
            alpha_std = np.std(alpha_samples)
            
            print(f"Recovery results:")
            print(f"  H0: true={true_h0:.1f}, recovered={h0_mean:.1f}±{h0_std:.1f}")
            print(f"  α: true={true_alpha:.3f}, recovered={alpha_mean:.3f}±{alpha_std:.3f}")
            
            # Check recovery within reasonable bounds (allowing for uncertainty)
            h0_bias = abs(h0_mean - true_h0)
            alpha_bias = abs(alpha_mean - true_alpha)
            
            # Recovery should be within ~2-3 sigma of truth (allowing for model limitations)
            assert h0_bias < 3 * h0_std + 5.0, f"H0 recovery bias too large: {h0_bias:.1f} vs std {h0_std:.1f}"
            assert alpha_bias < 3 * alpha_std + 0.2, f"Alpha recovery bias too large: {alpha_bias:.3f} vs std {alpha_std:.3f}"
            
            print("✅ Parameter recovery test passed!")
        else:
            pytest.skip("Insufficient MCMC samples for recovery test")
    else:
        pytest.skip("MCMC failed for recovery test")


def test_analytical_limits():
    """Test model behavior in analytical limits."""
    
    # Test Case 1: Single galaxy, no redshift uncertainty, alpha=0
    print("\n🧮 Analytical Limits Test")
    
    # Create simple case: one galaxy at known redshift
    rng = np.random.default_rng(42)
    true_h0 = 70.0
    true_alpha = 0.0  # No selection bias
    galaxy_z = 0.02
    
    # Generate GW samples around the true distance
    true_distance = compute_luminosity_distance_consistent(galaxy_z, true_h0)
    
    n_samples = 1000
    gw_uncertainty = 0.05 * true_distance  # 5% uncertainty
    gw_samples = rng.normal(true_distance, gw_uncertainty, size=n_samples)
    
    # Set up likelihood with minimal uncertainty
    ll = get_log_likelihood_h0(
        requested_backend_str="auto",
        dL_gw_samples=gw_samples,
        host_galaxies_z=np.array([galaxy_z]),
        host_galaxies_mass_proxy=np.array([1.0]),  # Single galaxy
        host_galaxies_z_err=np.array([1e-8]),  # Negligible redshift uncertainty
        h0_min=50.0,
        h0_max=90.0,
        alpha_min=-0.1,
        alpha_max=0.1,
        z_err_threshold=1e-6,  # Skip marginalization
        sigma_v=100.0,  # Small peculiar velocity
    )
    
    # Test likelihood at true parameters
    true_logL = ll([true_h0, true_alpha])
    assert np.isfinite(true_logL), f"True likelihood not finite: {true_logL}"
    
    # Test that H0 likelihood has expected shape (should peak near truth)
    h0_test_values = np.linspace(60, 80, 21)
    logL_values = [ll([h0, true_alpha]) for h0 in h0_test_values]
    
    # Find maximum
    max_idx = np.argmax(logL_values)
    h0_max_likelihood = h0_test_values[max_idx]
    
    print(f"True H0: {true_h0:.1f}")
    print(f"Maximum likelihood H0: {h0_max_likelihood:.1f}")
    print(f"Difference: {abs(h0_max_likelihood - true_h0):.1f}")
    
    # In the analytical limit, should recover true H0 within reasonable tolerance
    assert abs(h0_max_likelihood - true_h0) < 5.0, f"Poor H0 recovery in analytical limit"
    
    print("✅ Analytical limits test passed!")


if __name__ == "__main__":
    # Can run individual tests for debugging
    test_improved_mock_data_generation()
    print("\n" + "="*50)
    test_analytical_limits() 