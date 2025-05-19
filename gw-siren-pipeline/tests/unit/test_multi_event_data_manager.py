import numpy as np
import pandas as pd
import textwrap

import pytest

from gwsiren.multi_event_data_manager import (
    load_multi_event_config,
    prepare_event_data,
    EventDataPackage,
    _load_or_fetch_gw_posteriors,
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
    ra = np.array([10.0, 11.0])
    dec = np.array([-1.0, -2.0])
    df = pd.DataFrame({"z": [0.01], "mass_proxy": [1.0], "z_err": [0.001]})
    cache_dir = tmp_path / "cache"
    gw_cache = tmp_path / "gw"
    cache_dir.mkdir()
    gw_cache.mkdir()
    cache_file = cache_dir / "EV1_cat_glade+_n128_cdf0.9_zfb0.05.csv"
    np.savez(gw_cache / "EV1_gw_posteriors.npz", dl=dl, ra=ra, dec=dec)
    df.to_csv(cache_file, index=False)

    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(
            candidate_galaxy_cache_dir=str(cache_dir),
            gw_posteriors_cache_dir=str(gw_cache),
        )
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

    cfg = {"event_id": "EV1"}

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
    gw_mock = mocker.patch(
        "gwsiren.multi_event_data_manager._load_or_fetch_gw_posteriors",
        return_value=(results["dL_samples"], np.array([]), np.array([])),
    )

    cache_dir = tmp_path / "cache"
    gw_cache = tmp_path / "gw"
    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(
            candidate_galaxy_cache_dir=str(cache_dir),
            gw_posteriors_cache_dir=str(gw_cache),
        )
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
    gw_mock.assert_called_once_with("EVGEN")


def test_candidate_galaxy_cache(tmp_path, mock_config, mocker, monkeypatch):
    df = pd.DataFrame({"z": [0.03], "mass_proxy": [3.0], "z_err": [0.003]})
    results = {"dL_samples": np.array([5.0, 6.0]), "candidate_hosts_df": df}
    run_mock = mocker.patch(
        "gwsiren.multi_event_data_manager.run_full_analysis",
        return_value=results,
    )
    gw_mock = mocker.patch(
        "gwsiren.multi_event_data_manager._load_or_fetch_gw_posteriors",
        return_value=(results["dL_samples"], np.array([]), np.array([])),
    )

    cache_dir = tmp_path / "cache"
    gw_cache = tmp_path / "gw"
    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(
            candidate_galaxy_cache_dir=str(cache_dir),
            gw_posteriors_cache_dir=str(gw_cache),
        )
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

    cfg = {"event_id": "EVX"}

    # First call generates and writes to cache
    package1 = prepare_event_data(cfg)
    run_mock.assert_called_once()
    cached = cache_dir / "EVX_cat_glade+_n128_cdf0.9_zfb0.05.csv"
    assert cached.exists()

    # Second call should load from cache
    run_mock.reset_mock()
    package2 = prepare_event_data(cfg)
    run_mock.assert_not_called()
    gw_mock.assert_called()

    pd.testing.assert_frame_equal(package1.candidate_galaxies_df, df)
    pd.testing.assert_frame_equal(package2.candidate_galaxies_df, df)


def test_gw_posteriors_cache_hit(tmp_path, mock_config, mocker, monkeypatch):
    gw_dir = tmp_path / "gw"
    gw_dir.mkdir()
    np.savez(gw_dir / "EV1_gw_posteriors.npz", dl=np.array([1.0]), ra=np.array([0.1]), dec=np.array([-0.1]))

    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(gw_posteriors_cache_dir=str(gw_dir))
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
    fetch_mock = mocker.patch("gwsiren.multi_event_data_manager.fetch_candidate_data")
    dl, ra, dec = _load_or_fetch_gw_posteriors("EV1")
    assert np.allclose(dl, [1.0])
    assert np.allclose(ra, [0.1])
    assert np.allclose(dec, [-0.1])
    fetch_mock.assert_not_called()


def test_gw_posteriors_cache_miss(tmp_path, mock_config, mocker, monkeypatch):
    gw_dir = tmp_path / "gw"
    gw_dir.mkdir()
    pes_dir = tmp_path / "pes"

    monkeypatch.setattr(
        "gwsiren.multi_event_data_manager.configure_astropy_cache",
        lambda *_args, **_kw: str(pes_dir),
    )
    fetch_mock = mocker.patch(
        "gwsiren.multi_event_data_manager.fetch_candidate_data",
        return_value=(True, "obj"),
    )
    extract_mock = mocker.patch(
        "gwsiren.multi_event_data_manager.extract_gw_event_parameters",
        return_value=(np.array([2.0]), np.array([0.2]), np.array([-0.2])),
    )

    me_cfg = MultiEventAnalysisSettings(
        run_settings=MERunSettings(gw_posteriors_cache_dir=str(gw_dir))
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

    dl, ra, dec = _load_or_fetch_gw_posteriors("EV2")

    assert np.allclose(dl, [2.0])
    assert (gw_dir / "EV2_gw_posteriors.npz").exists()
    fetch_mock.assert_called_once()
    extract_mock.assert_called_once()
