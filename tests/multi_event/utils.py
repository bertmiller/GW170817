"""Utilities for multi-event test data."""
import numpy as np
import pandas as pd
from pathlib import Path


def create_mock_event_files(
    base_dir: Path, event_id: str, mu_dl: float, sigma_dl: float, z_host: float
) -> None:
    """Generate mock GW posterior and galaxy CSV files for an event.

    Args:
        base_dir: Temporary directory to write files into.
        event_id: Identifier used for filenames.
        mu_dl: Mean of the luminosity distance samples (Mpc).
        sigma_dl: Standard deviation of the distance samples.
        z_host: Redshift of the single mock host galaxy.
    """
    gw_dir = base_dir / "gw"
    gal_dir = base_dir / "gal"
    gw_dir.mkdir(parents=True, exist_ok=True)
    gal_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    dls = rng.normal(mu_dl, sigma_dl, size=50)
    ra = rng.uniform(0, 360, size=50)
    dec = rng.uniform(-30, 30, size=50)
    np.savez(gw_dir / f"{event_id}_gw_posteriors.npz", dl=dls, ra=ra, dec=dec)

    df = pd.DataFrame(
        {
            "PGC": [1],
            "ra": [ra.mean()],
            "dec": [dec.mean()],
            "z": [z_host],
            "mass_proxy": [1.0],
            "z_err": [0.001],
        }
    )
    df.to_csv(gal_dir / f"{event_id}_cat_glade+_n128_cdf0.9_zfb0.05.csv", index=False)


def build_test_config(tmp_path: Path, event_ids: list[str]) -> Path:
    """Create a minimal config YAML for multi-event tests."""
    cfg = {
        "catalog": {
            "glade_plus_url": "http://example.com/gal.txt",
            "glade24_url": "http://example.com/gal24.txt",
            "data_dir": str(tmp_path / "cat"),
        },
        "skymap": {"default_nside": 16, "credible_level": 0.9},
        "mcmc": {
            "walkers": 4,
            "steps": 20,
            "burnin": 5,
            "thin_by": 1,
            "prior_h0_min": 20.0,
            "prior_h0_max": 140.0,
        },
        "cosmology": {"sigma_v_pec": 200.0, "c_light": 299792.458, "omega_m": 0.3},
        "fetcher": {"cache_dir_name": "cache", "timeout": 1, "max_retries": 1},
        "multi_event_analysis": {
            "run_settings": {
                "run_label": "test_run",
                "base_output_directory": str(tmp_path / "out"),
                "candidate_galaxy_cache_dir": str(tmp_path / "gal"),
                "gw_posteriors_cache_dir": str(tmp_path / "gw"),
            },
            "events_to_combine": [{"event_id": eid} for eid in event_ids],
            "priors": {"H0": {"min": 20.0, "max": 140.0}, "alpha": {"min": -1.0, "max": 1.0}},
            "mcmc": {"n_walkers": 4, "n_steps": 20, "burnin": 5, "thin_by": 1},
            "cosmology": {"sigma_v_pec": 200.0},
        },
    }
    path = tmp_path / "test_config.yaml"
    import yaml

    with open(path, "w") as fh:
        yaml.dump(cfg, fh)
    return path
