import importlib
from unittest.mock import MagicMock
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import logging


def test_h0_pipeline_e2e_workflow(mocker, tmp_path, capsys, caplog, mock_config):
    event_name = "MOCK_GW_EVENT"

    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Import pipeline after config fixture patched CONFIG
    pipeline = importlib.import_module("scripts.h0_e2e_pipeline")
    pipeline_mod = importlib.import_module("gwsiren.pipeline")

    # Ensure pipeline uses temporary output directory
    mocker.patch.object(pipeline, "OUTPUT_DIR", str(tmp_path))
    mocker.patch.object(pipeline_mod, "OUTPUT_DIR", str(tmp_path))

    # Mock cache configuration
    mocker.patch.object(
        pipeline_mod,
        "configure_astropy_cache",
        return_value=str(tmp_path / "cache"),
    )

    # Mock GW data fetching and parameter extraction
    mock_gw_obj = MagicMock()
    fetch_mock = mocker.patch.object(
        pipeline_mod,
        "fetch_candidate_data",
        return_value=(True, mock_gw_obj),
    )
    dl_samples = np.array([40.0, 45.0, 50.0])
    ra_samples = np.array([30.0, 30.1, 30.2])
    dec_samples = np.array([-10.0, -10.1, -9.9])
    mocker.patch.object(
        pipeline_mod,
        "extract_gw_event_parameters",
        return_value=(dl_samples, ra_samples, dec_samples),
    )

    # Mock galaxy catalog loading and processing
    gal_df = pd.DataFrame({
        "PGC": [1, 2],
        "ra": [30.05, 30.3],
        "dec": [-10.05, -9.95],
        "z": [0.01, 0.02],
        "mass_proxy": [1.0, 2.0],  # Added mass_proxy column
        "z_err": [0.001, 0.001],
    })
    mocker.patch.object(
        pipeline_mod,
        "download_and_load_galaxy_catalog",
        return_value=gal_df,
    )
    mocker.patch.object(pipeline_mod, "clean_galaxy_catalog", return_value=gal_df)
    mocker.patch.object(
        pipeline_mod,
        "apply_specific_galaxy_corrections",
        return_value=gal_df,
    )

    # Mock sky selection steps
    npix = 12 * pipeline.NSIDE_SKYMAP * pipeline.NSIDE_SKYMAP
    sky_mask = np.zeros(npix, dtype=bool)
    sky_mask[0] = True
    mocker.patch.object(
        pipeline_mod,
        "generate_sky_map_and_credible_region",
        return_value=(np.zeros(npix), sky_mask, 0.1),
    )
    mocker.patch.object(pipeline_mod, "select_galaxies_in_sky_region", return_value=gal_df)
    mocker.patch.object(pipeline_mod, "filter_galaxies_by_redshift", return_value=gal_df)
    mocker.patch.object(pipeline_mod, "estimate_event_specific_z_max", return_value=0.05)

    # Mock MCMC run
    mock_sampler = MagicMock()
    # Create a mock chain with 2 dimensions (H0 and alpha)
    mock_chain = np.random.normal(70, 5, size=(40, 2))  # Changed to 2 dimensions
    mock_sampler.get_chain.return_value = mock_chain
    mcmc_mock = mocker.patch.object(pipeline_mod, "run_mcmc_h0", return_value=mock_sampler)

    # Avoid spawning processes
    class DummyPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return MagicMock()

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    mocker.patch.object(pipeline_mod, "InterruptiblePool", DummyPool)

    # Simulate command-line arguments
    mocker.patch.object(sys, "argv", ["h0_e2e_pipeline.py", event_name])

    with caplog.at_level(logging.INFO):
        pipeline.main()

    fetch_mock.assert_called_once()
    mcmc_mock.assert_called_once()

    out_file = tmp_path / f"H0_samples_{event_name}.npy"
    assert out_file.exists()
    samples = np.load(out_file)
    assert samples.size > 0

    captured = caplog.text
    assert "H0 =" in captured 