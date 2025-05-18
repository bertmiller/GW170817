import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import sys
import pytest

import gwsiren.gw_data_fetcher as gw_data_fetcher
import gwsiren.event_data_extractor as event_data_extractor
import gwsiren.data.catalogs as galaxy_catalog_handler
import gwsiren.sky_analyzer as sky_analyzer


@pytest.fixture
def prevent_main_import(monkeypatch):
    """Prevent execution of main.py when mock_config imports modules."""
    monkeypatch.setitem(sys.modules, "main", MagicMock())


def test_successful_data_pipeline_flow(prevent_main_import, mocker, mock_config):
    # Mock GW data fetching
    mock_dL = np.array([100.0, 110.0])
    mock_ra_rad = np.deg2rad([10.1, 10.2])
    mock_dec_rad = np.deg2rad([5.0, 5.1])
    mock_gw_obj = MagicMock()
    mock_gw_obj.samples_dict = {
        "C01:IMRPhenomXPHM": {
            "luminosity_distance": mock_dL,
            "ra": mock_ra_rad,
            "dec": mock_dec_rad,
        }
    }
    mocker.patch(
        "gwsiren.gw_data_fetcher.fetch_candidate_data",
        return_value=(True, mock_gw_obj),
    )

    # Mock galaxy catalog loading/cleaning
    mock_raw_df = pd.DataFrame(
        {
            "PGC": [1, 2, 3, 4],
            "ra": [10.0, 15.0, 12.0, 180.0],
            "dec": [5.0, -5.0, 6.0, 0.0],
            "z": [0.01, 0.02, 0.015, 0.08],
        }
    )
    mock_clean_df = mock_raw_df.copy()
    mocker.patch(
        "gwsiren.data.catalogs.download_and_load_galaxy_catalog",
        return_value=mock_raw_df,
    )
    mocker.patch(
        "gwsiren.data.catalogs.clean_galaxy_catalog",
        return_value=mock_clean_df,
    )

    # Control z max estimation
    mocker.patch(
        "gwsiren.sky_analyzer.estimate_event_specific_z_max",
        return_value=0.05,
    )

    success, gw_data = gw_data_fetcher.fetch_candidate_data(
        "MOCK_EVENT", mock_config.catalog["data_dir"]
    )
    assert success is True

    dL_samples, ra_deg, dec_deg = event_data_extractor.extract_gw_event_parameters(
        gw_data, "MOCK_EVENT"
    )
    assert np.allclose(ra_deg, np.rad2deg(mock_ra_rad))
    assert np.allclose(dec_deg, np.rad2deg(mock_dec_rad))

    raw_cat = galaxy_catalog_handler.download_and_load_galaxy_catalog(
        catalog_type=mock_config.catalog.get("default_catalog_type", "glade+")
    )
    cleaned_cat_df = galaxy_catalog_handler.clean_galaxy_catalog(raw_cat)

    host_z_max = sky_analyzer.estimate_event_specific_z_max(dL_samples)
    prob_map, sky_mask, _ = sky_analyzer.generate_sky_map_and_credible_region(
        ra_deg, dec_deg, nside=16
    )
    spatial_df = sky_analyzer.select_galaxies_in_sky_region(
        cleaned_cat_df, sky_mask, nside=16
    )
    final_df = sky_analyzer.filter_galaxies_by_redshift(spatial_df, host_z_max)

    assert isinstance(prob_map, np.ndarray)
    assert isinstance(sky_mask, np.ndarray) and sky_mask.dtype == bool
    assert not spatial_df.empty
    assert all(final_df["z"] < host_z_max)


def test_pipeline_handles_failure_in_extraction(prevent_main_import, mocker, mock_config):
    mock_gw_obj = MagicMock()
    mock_gw_obj.samples_dict = {}
    mocker.patch(
        "gwsiren.gw_data_fetcher.fetch_candidate_data",
        return_value=(True, mock_gw_obj),
    )
    mocker.patch(
        "gwsiren.event_data_extractor.extract_gw_event_parameters",
        return_value=(None, None, None),
    )
    mock_raw_df = pd.DataFrame(
        {
            "PGC": [1],
            "ra": [10.0],
            "dec": [5.0],
            "z": [0.01],
        }
    )
    mocker.patch(
        "gwsiren.data.catalogs.download_and_load_galaxy_catalog",
        return_value=mock_raw_df,
    )
    mocker.patch(
        "gwsiren.data.catalogs.clean_galaxy_catalog",
        return_value=mock_raw_df,
    )
    mocker.patch(
        "gwsiren.sky_analyzer.estimate_event_specific_z_max",
        return_value=0.05,
    )
    npix = sky_analyzer.hp.nside2npix(16)
    empty_mask = np.zeros(npix, dtype=bool)
    mocker.patch(
        "gwsiren.sky_analyzer.generate_sky_map_and_credible_region",
        return_value=(np.zeros(npix), empty_mask, 0.0),
    )

    success, gw_data = gw_data_fetcher.fetch_candidate_data(
        "MOCK_EVENT", mock_config.catalog["data_dir"]
    )
    assert success

    dL_samples, ra_deg, dec_deg = event_data_extractor.extract_gw_event_parameters(
        gw_data, "MOCK_EVENT"
    )
    host_z_max = sky_analyzer.estimate_event_specific_z_max(dL_samples)
    prob_map, sky_mask, _ = sky_analyzer.generate_sky_map_and_credible_region(
        ra_deg, dec_deg, nside=16
    )
    spatial_df = sky_analyzer.select_galaxies_in_sky_region(
        mock_raw_df, sky_mask, nside=16
    )
    final_df = sky_analyzer.filter_galaxies_by_redshift(spatial_df, host_z_max)

    assert ra_deg is None and dec_deg is None
    assert sky_mask.sum() == 0
    assert spatial_df.empty
    assert final_df.empty
