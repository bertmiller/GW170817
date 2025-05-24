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

