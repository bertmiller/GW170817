import numpy as np
import pandas as pd
import textwrap

import pytest

from gwsiren.multi_event_data_manager import (
    load_multi_event_config,
    prepare_event_data,
    EventDataPackage,
)
from gwsiren.config import Config, MultiEventAnalysisSettings, MERunSettings
from gwsiren.pipeline import (
    NSIDE_SKYMAP,
    CDF_THRESHOLD,
    CATALOG_TYPE,
    HOST_Z_MAX_FALLBACK,
)


def test_load_multi_event_config(tmp_path):
    cfg_text = textwrap.dedent(
        """
        events_to_combine:
          - event_id: EV1
            gw_dl_samples_path: dl1.npy
          - event_id: EV2
        """
    )
    cfg_file = tmp_path / "events.yaml"
    cfg_file.write_text(cfg_text)

    events = load_multi_event_config(cfg_file)

    assert len(events) == 2
    assert events[0]["event_id"] == "EV1"
    assert events[1]["event_id"] == "EV2"


def test_prepare_event_data_loads_files(tmp_path, mock_config, monkeypatch):
    dl = np.array([1.0, 2.0])
    df = pd.DataFrame({"z": [0.01], "mass_proxy": [1.0], "z_err": [0.001]})
    dl_path = tmp_path / "dl.npy"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "EV1_cat_glade+_n128_cdf0.9_zfb0.05.csv"
    np.save(dl_path, dl)
    df.to_csv(cache_file, index=False)

    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(candidate_galaxy_cache_dir=str(cache_dir))
    )
    new_cfg = Config(
        catalog=mock_config.catalog,
        skymap=mock_config.skymap,
        mcmc=mock_config.mcmc,
        cosmology=mock_config.cosmology,
        fetcher=mock_config.fetcher,
        multi_event_analysis=me_cfg,
    )
    monkeypatch.setattr("gwsiren.multi_event_data_manager.CONFIG", new_cfg)

    cfg = {
        "event_id": "EV1",
        "gw_dl_samples_path": str(dl_path),
    }

    package = prepare_event_data(cfg)

    assert isinstance(package, EventDataPackage)
    assert np.allclose(package.dl_samples, dl)
    pd.testing.assert_frame_equal(package.candidate_galaxies_df, df)


def test_prepare_event_data_generation_path(tmp_path, mock_config, monkeypatch, mocker):
    df = pd.DataFrame({"z": [0.02], "mass_proxy": [2.0], "z_err": [0.002]})
    results = {"dL_samples": np.array([3.0, 4.0]), "candidate_hosts_df": df}
    run_mock = mocker.patch(
        "gwsiren.multi_event_data_manager.run_full_analysis",
        return_value=results,
    )

    cache_dir = tmp_path / "cache"
    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(candidate_galaxy_cache_dir=str(cache_dir))
    )
    new_cfg = Config(
        catalog=mock_config.catalog,
        skymap=mock_config.skymap,
        mcmc=mock_config.mcmc,
        cosmology=mock_config.cosmology,
        fetcher=mock_config.fetcher,
        multi_event_analysis=me_cfg,
    )
    monkeypatch.setattr("gwsiren.multi_event_data_manager.CONFIG", new_cfg)

    cfg = {"event_id": "EVGEN"}

    package = prepare_event_data(cfg)

    run_mock.assert_called_once_with(
        "EVGEN",
        perform_mcmc=False,
        nside_skymap=NSIDE_SKYMAP,
        cdf_threshold=CDF_THRESHOLD,
        catalog_type=CATALOG_TYPE,
        host_z_max_fallback=HOST_Z_MAX_FALLBACK,
    )
    assert np.allclose(package.dl_samples, results["dL_samples"])
    pd.testing.assert_frame_equal(package.candidate_galaxies_df, df)


def test_candidate_galaxy_cache(tmp_path, mock_config, mocker, monkeypatch):
    df = pd.DataFrame({"z": [0.03], "mass_proxy": [3.0], "z_err": [0.003]})
    results = {"dL_samples": np.array([5.0, 6.0]), "candidate_hosts_df": df}
    run_mock = mocker.patch(
        "gwsiren.multi_event_data_manager.run_full_analysis",
        return_value=results,
    )

    cache_dir = tmp_path / "cache"
    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(candidate_galaxy_cache_dir=str(cache_dir))
    )
    new_cfg = Config(
        catalog=mock_config.catalog,
        skymap=mock_config.skymap,
        mcmc=mock_config.mcmc,
        cosmology=mock_config.cosmology,
        fetcher=mock_config.fetcher,
        multi_event_analysis=me_cfg,
    )
    monkeypatch.setattr("gwsiren.multi_event_data_manager.CONFIG", new_cfg)

    dl_path = tmp_path / "dl.npy"
    np.save(dl_path, results["dL_samples"])
    cfg = {"event_id": "EVX", "gw_dl_samples_path": str(dl_path)}

    # First call generates and writes to cache
    package1 = prepare_event_data(cfg)
    run_mock.assert_called_once()
    cached = cache_dir / "EVX_cat_glade+_n128_cdf0.9_zfb0.05.csv"
    assert cached.exists()

    # Second call should load from cache
    run_mock.reset_mock()
    package2 = prepare_event_data(cfg)
    run_mock.assert_not_called()

    pd.testing.assert_frame_equal(package1.candidate_galaxies_df, df)
    pd.testing.assert_frame_equal(package2.candidate_galaxies_df, df)
