import importlib
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


def test_e2e_gw170817_pipeline_with_derived_data(mocker, mock_config, tmp_path):
    """Run the H0 pipeline using pre-derived GW170817 mock data."""
    event_name = "GW170817"
    
    # Import pipeline after config fixture patched CONFIG
    pipeline = importlib.import_module("scripts.h0_e2e_pipeline")
    pipeline_mod = importlib.import_module("gwsiren.pipeline")
    h0_mcmc_analyzer_mod = importlib.import_module("gwsiren.h0_mcmc_analyzer")

    # Ensure pipeline uses temporary output directory
    mocker.patch.object(pipeline, "OUTPUT_DIR", str(tmp_path))
    mocker.patch.object(pipeline_mod, "OUTPUT_DIR", str(tmp_path))

    # Mock cache configuration
    mocker.patch.object(
        pipeline_mod,
        "configure_astropy_cache",
        return_value=str(tmp_path / "cache"),
    )
    # Derived GW samples
    derived_ra_deg = np.full(75, 197.45042353)
    derived_dec_deg = np.full(75, -23.38149089)
    derived_dl_mpc = np.array([
        46.75847203,
        45.77873623,
        41.52210869,
        43.26622309,
        43.88671261,
        39.49157621,
        19.28519137,
        41.94769991,
        42.89321902,
        28.39683068,
        44.95082192,
        36.95666051,
        27.29144277,
        34.67295683,
        47.24461899,
        43.11595219,
        42.86081414,
        42.17785192,
        45.91468308,
        35.64024599,
        44.03971472,
        36.59155545,
        35.69079274,
        41.98989627,
        42.66393654,
        46.74923377,
        37.9570358,
        27.37949843,
        45.03297551,
        36.53665506,
        43.4719077,
        42.50402541,
        41.56969619,
        40.42483778,
        43.19102484,
        31.56886075,
        34.49847464,
        44.74725961,
        45.28648584,
        48.96765026,
        42.60336965,
        39.97862641,
        43.93860134,
        46.04031786,
        31.68063589,
        38.20645093,
        34.55678703,
        44.62508399,
        45.85552663,
        45.67177303,
        38.87002232,
        45.734932,
        37.30076946,
        45.99007885,
        41.32326228,
        41.41190384,
        40.06815323,
        47.82867778,
        48.83519933,
        40.77895788,
        44.50614983,
        46.50536149,
        40.68483997,
        40.5646827,
        46.97173835,
        43.10367564,
        45.09453533,
        43.67660806,
        45.39380695,
        42.40362707,
        43.47804755,
        40.09708223,
        46.78010998,
        35.56275441,
        34.39754969,
    ])

    derived_galaxy_subset_df = pd.DataFrame(
        {
            "PGC": [45657.0, 45550.0, 26377.0],
            "ra": [197.448776, 197.135, 139.945],
            "dec": [-23.383831, -23.34725, 33.74957],
            "z": [0.00968, 0.0102614451307, 0.0217207308604],
            "mass_proxy": [2.2, 0.03, 11.0],
        }
    )

    # Mock data fetching and extraction - target gwsiren.pipeline
    mock_gw_obj = MagicMock()
    mocker.patch(
        "gwsiren.pipeline.fetch_candidate_data",
        return_value=(True, mock_gw_obj),
    )
    mocker.patch(
        "gwsiren.pipeline.extract_gw_event_parameters",
        return_value=(derived_dl_mpc, derived_ra_deg, derived_dec_deg),
    )

    # Mock galaxy catalog functions - target gwsiren.pipeline
    mocker.patch(
        "gwsiren.pipeline.download_and_load_galaxy_catalog",
        return_value=derived_galaxy_subset_df,
    )
    mocker.patch(
        "gwsiren.pipeline.clean_galaxy_catalog",
        return_value=derived_galaxy_subset_df,
    )
    mocker.patch(
        "gwsiren.pipeline.apply_specific_galaxy_corrections",
        side_effect=lambda df, *args, **kwargs: df,
    )

    # Patch MCMC parameters for a quick run - target gwsiren.h0_mcmc_analyzer
    test_n_walkers = 8
    test_n_steps = 50
    test_burnin = 10
    test_thin_by = 2
    mocker.patch("gwsiren.h0_mcmc_analyzer.DEFAULT_MCMC_N_WALKERS", test_n_walkers)
    mocker.patch("gwsiren.h0_mcmc_analyzer.DEFAULT_MCMC_N_STEPS", test_n_steps)
    mocker.patch("gwsiren.h0_mcmc_analyzer.DEFAULT_MCMC_BURNIN", test_burnin)
    mocker.patch("gwsiren.h0_mcmc_analyzer.DEFAULT_MCMC_THIN_BY", test_thin_by)

    # Avoid creating real multiprocessing pools - target gwsiren.pipeline
    # as InterruptiblePool is imported in gwsiren.pipeline and likely used from there
    # by h0_mcmc_analyzer, consistent with test_h0_e2e_pipeline.py
    class DummyPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return None  # Run emcee in serial without a pool

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    mocker.patch("gwsiren.pipeline.InterruptiblePool", DummyPool)

    results = pipeline.run_full_analysis(
        event_name,
        perform_mcmc=True,
        nside_skymap=64,
    )

    assert results.get("error") is None
    candidate_hosts_df = results.get("candidate_hosts_df")
    assert not candidate_hosts_df.empty
    assert 45657.0 in candidate_hosts_df["PGC"].values
    assert 1 <= len(candidate_hosts_df) <= 2

    sampler = results.get("sampler")
    assert sampler is not None
    flat_samples = results.get("flat_h0_samples")
    assert flat_samples is not None
    assert flat_samples.ndim == 2 and flat_samples.shape[1] == 2
    assert np.all(np.isfinite(flat_samples))

    h0_median = float(np.median(flat_samples[:, 0]))
    alpha_median = float(np.median(flat_samples[:, 1]))
    assert np.isfinite(h0_median) and 10 < h0_median < 200
    alpha_min = h0_mcmc_analyzer_mod.DEFAULT_ALPHA_PRIOR_MIN
    alpha_max = h0_mcmc_analyzer_mod.DEFAULT_ALPHA_PRIOR_MAX
    assert np.isfinite(alpha_median)
    assert alpha_min <= alpha_median <= alpha_max

    pipeline.save_h0_samples_and_print_summary(
        flat_samples, event_name, len(candidate_hosts_df)
    )
    out_file = tmp_path / f"H0_samples_{event_name}.npy"
    assert out_file.exists()

