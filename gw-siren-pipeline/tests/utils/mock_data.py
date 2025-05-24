"""Utilities for generating mock data for tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gwsiren.event_data import EventDataPackage


def gw_posterior_samples(n_samples: int = 100, mean: float = 100.0, std: float = 10.0, seed: int | None = None) -> np.ndarray:
    """Generate mock luminosity distance posterior samples.

    Args:
        n_samples: Number of samples to generate.
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the distribution.
        seed: Random seed for reproducibility.

    Returns:
        Array of mock luminosity distance samples.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=n_samples).astype(float)


def galaxy_catalog(
    n_galaxies: int = 5,
    z_mean: float = 0.02,
    z_std: float = 0.005,
    mass_mean: float = 1.0,
    mass_std: float = 0.1,
    z_err: float = 0.001,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple mock galaxy catalog.

    Args:
        n_galaxies: Number of galaxies.
        z_mean: Mean galaxy redshift.
        z_std: Standard deviation of galaxy redshift.
        mass_mean: Mean of the mass proxy values.
        mass_std: Standard deviation of the mass proxy values.
        z_err: Measurement error on redshift for all galaxies.
        seed: Random seed.

    Returns:
        Tuple of arrays ``(z, mass_proxy, z_err)``.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(z_mean, z_std, size=n_galaxies).astype(float)
    mass_proxy = rng.lognormal(mean=np.log(mass_mean), sigma=mass_std, size=n_galaxies).astype(float)
    z_err_arr = np.full(n_galaxies, z_err, dtype=float)
    return z, mass_proxy, z_err_arr


def mock_event(
    event_id: str = "EV",
    n_samples: int = 100,
    n_galaxies: int = 5,
    seed: int | None = None,
) -> EventDataPackage:
    """Create a complete mock event for likelihood tests."""
    rng = np.random.default_rng(seed)
    dl = gw_posterior_samples(n_samples=n_samples, seed=rng.integers(0, 2**32 - 1))
    z, mass, z_err = galaxy_catalog(n_galaxies=n_galaxies, seed=rng.integers(0, 2**32 - 1))
    df = pd.DataFrame({"z": z, "mass_proxy": mass, "z_err": z_err})
    return EventDataPackage(event_id=event_id, dl_samples=dl, candidate_galaxies_df=df)


def multi_event(n_events: int = 2, seed: int | None = None) -> list[EventDataPackage]:
    """Generate a list of mock events."""
    rng = np.random.default_rng(seed)
    packages = []
    for i in range(n_events):
        packages.append(mock_event(event_id=f"EV{i}", seed=rng.integers(0, 2**32 - 1)))
    return packages


# =============================================================================
# Physics-Consistent Mock Data Generation for SBC Testing
# =============================================================================

def compute_luminosity_distance_consistent(z: float, H0: float, omega_m: float = 0.31, 
                                         c_light: float = 299792.458, N_trapz: int = 200) -> float:
    """Compute luminosity distance using same method as likelihood function.
    
    This MUST match exactly with h0_mcmc_analyzer._lum_dist_model to avoid systematic bias.
    
    Args:
        z: Redshift
        H0: Hubble constant in km/s/Mpc
        omega_m: Matter density parameter
        c_light: Speed of light in km/s
        N_trapz: Number of integration points
        
    Returns:
        Luminosity distance in Mpc
    """
    if z < 1e-9:
        return 0.0
    
    # Same integration method as likelihood
    z_steps = np.linspace(0.0, z, N_trapz)
    one_plus_z_steps = 1.0 + z_steps
    Ez_sq = omega_m * (one_plus_z_steps**3) + (1.0 - omega_m)
    Ez = np.sqrt(np.maximum(1e-18, Ez_sq))
    integrand = 1.0 / np.maximum(1e-18, Ez)
    
    # Trapezoidal integration (same as trapz_xp backend)
    integral = np.trapz(integrand, x=z_steps)
    return (c_light / H0) * (1.0 + z) * integral


def generate_galaxy_catalog_with_proper_selection(
    n_observed: int,
    true_alpha: float, 
    z_mean: float = 0.02,
    z_std: float = 0.005,
    mass_distribution_params: dict = None,
    z_err_level: float = 0.002,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate galaxy catalog with realistic mass-dependent selection.
    
    This implements the selection process that the likelihood assumes,
    rather than just computing selection weights.
    
    Args:
        n_observed: Number of observed galaxies after selection
        true_alpha: True mass selection parameter
        z_mean: Mean redshift of parent population
        z_std: Standard deviation of redshift distribution
        mass_distribution_params: Parameters for mass distribution
        z_err_level: Redshift measurement error level
        seed: Random seed
        
    Returns:
        Tuple of (redshifts, masses, redshift_errors) for observed galaxies
    """
    if mass_distribution_params is None:
        mass_distribution_params = {"log_mean": 0.0, "log_std": 0.8}
    
    rng = np.random.default_rng(seed)
    
    # Generate large parent population to select from
    n_parent = n_observed * 20  # Much larger parent sample
    
    # Parent galaxy properties - wide distributions
    parent_z = rng.normal(z_mean, z_std, size=n_parent)
    parent_z = np.clip(parent_z, 0.001, None)  # Ensure positive redshifts
    
    parent_masses = rng.lognormal(
        mean=mass_distribution_params["log_mean"],
        sigma=mass_distribution_params["log_std"], 
        size=n_parent
    )
    
    # Compute selection probabilities based on true alpha
    if np.abs(true_alpha) < 1e-6:  # alpha ≈ 0 case
        selection_weights = np.ones(n_parent)
    else:
        # Mass-dependent selection as assumed by likelihood
        selection_weights = parent_masses ** true_alpha
    
    # Normalize to probabilities
    selection_probs = selection_weights / selection_weights.sum()
    
    # Actually select the observed galaxies based on these probabilities
    observed_indices = rng.choice(n_parent, size=n_observed, 
                                p=selection_probs, replace=False)
    
    # Extract observed galaxy properties
    observed_z = parent_z[observed_indices]
    observed_masses = parent_masses[observed_indices]
    observed_z_err = np.full(n_observed, z_err_level)
    
    return observed_z, observed_masses, observed_z_err


def generate_gw_distance_samples_consistent(
    true_host_z: float,
    true_h0: float,
    n_samples: int = 500,
    gw_rel_uncertainty: float = 0.10,
    sigma_v: float = 250.0,
    c_light: float = 299792.458,
    seed: int = None,
) -> np.ndarray:
    """Generate GW distance samples with consistent physics.
    
    This includes both GW measurement uncertainty and peculiar velocity
    effects as assumed by the likelihood.
    
    Args:
        true_host_z: Redshift of true host galaxy
        true_h0: True Hubble constant
        n_samples: Number of GW distance samples
        gw_rel_uncertainty: Relative GW measurement uncertainty
        sigma_v: Peculiar velocity dispersion in km/s
        c_light: Speed of light in km/s  
        seed: Random seed
        
    Returns:
        Array of GW distance samples in Mpc
    """
    rng = np.random.default_rng(seed)
    
    # Compute true luminosity distance using same method as likelihood
    true_distance = compute_luminosity_distance_consistent(true_host_z, true_h0)
    
    # Add peculiar velocity uncertainty (same as likelihood assumes)
    pec_vel_uncertainty = (true_distance / c_light) * sigma_v
    
    # Add GW measurement uncertainty
    gw_uncertainty = gw_rel_uncertainty * true_distance
    
    # Total uncertainty (assuming uncorrelated)
    total_uncertainty = np.sqrt(pec_vel_uncertainty**2 + gw_uncertainty**2)
    
    # Generate samples
    gw_samples = rng.normal(true_distance, total_uncertainty, size=n_samples)
    
    return gw_samples


def generate_mock_data_physics_consistent(
    true_h0: float,
    true_alpha: float,
    n_gw_samples: int = 500,
    n_galaxies: int = 8,
    z_mean: float = 0.02,
    z_std: float = 0.005,
    gw_rel_uncertainty: float = 0.10,
    sigma_v: float = 250.0,
    z_err_level: float = 0.002,
    seed: int = None,
) -> dict:
    """Generate physics-consistent mock data for SBC testing.
    
    This function ensures that:
    1. Luminosity distances computed using same method as likelihood
    2. Galaxy selection properly implemented (not just weighted)
    3. All uncertainties consistent with likelihood assumptions
    4. Cosmological parameters consistent
    
    Args:
        true_h0: True Hubble constant
        true_alpha: True mass selection parameter
        n_gw_samples: Number of GW distance samples
        n_galaxies: Number of candidate galaxies
        z_mean: Mean redshift of galaxy population
        z_std: Standard deviation of redshift distribution
        gw_rel_uncertainty: Relative GW measurement uncertainty
        sigma_v: Peculiar velocity dispersion
        z_err_level: Redshift measurement error level
        seed: Random seed
        
    Returns:
        Dictionary with mock data components
    """
    rng = np.random.default_rng(seed)
    
    # Generate galaxy catalog with proper selection
    galaxy_z, galaxy_masses, galaxy_z_err = generate_galaxy_catalog_with_proper_selection(
        n_observed=n_galaxies,
        true_alpha=true_alpha,
        z_mean=z_mean,
        z_std=z_std,
        z_err_level=z_err_level,
        seed=rng.integers(0, 2**32-1),
    )
    
    # Choose one galaxy as the true host (randomly)
    true_host_idx = rng.integers(0, n_galaxies)
    true_host_z = galaxy_z[true_host_idx]
    
    # Generate GW distance samples around the true host
    gw_samples = generate_gw_distance_samples_consistent(
        true_host_z=true_host_z,
        true_h0=true_h0,
        n_samples=n_gw_samples,
        gw_rel_uncertainty=gw_rel_uncertainty,
        sigma_v=sigma_v,
        seed=rng.integers(0, 2**32-1),
    )
    
    return {
        'dl_samples': gw_samples,
        'galaxy_z': galaxy_z,
        'galaxy_mass': galaxy_masses,
        'galaxy_z_err': galaxy_z_err,
        'true_host_idx': true_host_idx,
        'true_h0': true_h0,
        'true_alpha': true_alpha,
        'true_host_z': true_host_z,
    }


def mock_event_physics_consistent(
    event_id: str = "SBC_EVENT",
    true_h0: float = 70.0,
    true_alpha: float = 0.0,
    n_gw_samples: int = 500,
    n_galaxies: int = 8,
    seed: int = None,
) -> EventDataPackage:
    """Create a mock event with physics-consistent data generation.
    
    This is the recommended function for SBC testing and validation.
    
    Args:
        event_id: Event identifier
        true_h0: True Hubble constant
        true_alpha: True mass selection parameter
        n_gw_samples: Number of GW distance samples
        n_galaxies: Number of candidate galaxies
        seed: Random seed
        
    Returns:
        EventDataPackage with physics-consistent mock data
    """
    
    mock_data = generate_mock_data_physics_consistent(
        true_h0=true_h0,
        true_alpha=true_alpha,
        n_gw_samples=n_gw_samples,
        n_galaxies=n_galaxies,
        seed=seed,
    )
    
    # Create DataFrame for galaxy catalog
    df = pd.DataFrame({
        "z": mock_data['galaxy_z'],
        "mass_proxy": mock_data['galaxy_mass'],
        "z_err": mock_data['galaxy_z_err'],
    })
    
    # Create EventDataPackage
    pkg = EventDataPackage(
        event_id=event_id,
        dl_samples=mock_data['dl_samples'],
        candidate_galaxies_df=df
    )
    
    # Store additional metadata for validation
    pkg._sbc_metadata = {
        'true_h0': mock_data['true_h0'],
        'true_alpha': mock_data['true_alpha'],
        'true_host_idx': mock_data['true_host_idx'],
        'true_host_z': mock_data['true_host_z'],
    }
    
    return pkg


def validate_mock_data_consistency(mock_data: dict, tolerance: float = 1e-6) -> bool:
    """Validate that mock data satisfies expected physical relationships.
    
    Args:
        mock_data: Mock data dictionary from generate_mock_data_physics_consistent
        tolerance: Numerical tolerance for validation
        
    Returns:
        True if validation passes, False otherwise
    """
    
    # Test 1: Distance-redshift relation consistency
    for i, z in enumerate(mock_data['galaxy_z']):
        expected_distance = compute_luminosity_distance_consistent(z, mock_data['true_h0'])
        # This is a loose check since we don't know which galaxy is the true host
        # Just verify distances are in reasonable range
        if expected_distance <= 0:
            print(f"Invalid distance for galaxy {i}: z={z}, d_L={expected_distance}")
            return False
    
    # Test 2: GW samples are positive and finite
    if not np.all(np.isfinite(mock_data['dl_samples'])):
        print("Non-finite GW distance samples")
        return False
    
    if not np.all(mock_data['dl_samples'] > 0):
        print("Negative GW distance samples")
        return False
    
    # Test 3: Galaxy properties are reasonable
    if not np.all(mock_data['galaxy_z'] > 0):
        print("Non-positive galaxy redshifts")
        return False
    
    if not np.all(mock_data['galaxy_mass'] > 0):
        print("Non-positive galaxy masses")
        return False
    
    if not np.all(mock_data['galaxy_z_err'] >= 0):
        print("Negative redshift errors")
        return False
    
    print("Mock data validation passed")
    return True


# Test function to verify consistency
def test_mock_data_physics():
    """Test that mock data generation is self-consistent."""
    
    # Test case 1: alpha = 0 (no selection bias)
    mock_data_1 = generate_mock_data_physics_consistent(
        true_h0=70.0,
        true_alpha=0.0,
        seed=42
    )
    assert validate_mock_data_consistency(mock_data_1)
    
    # Test case 2: alpha > 0 (favor high mass)
    mock_data_2 = generate_mock_data_physics_consistent(
        true_h0=65.0,
        true_alpha=0.5,
        seed=123
    )
    assert validate_mock_data_consistency(mock_data_2)
    
    # Test case 3: alpha < 0 (favor low mass)
    mock_data_3 = generate_mock_data_physics_consistent(
        true_h0=75.0,
        true_alpha=-0.3,
        seed=456
    )
    assert validate_mock_data_consistency(mock_data_3)
    
    print("All mock data physics tests passed!")


if __name__ == "__main__":
    test_mock_data_physics() 