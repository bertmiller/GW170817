import numpy as np
import pandas as pd
import healpy as hp
import pytest
from unittest.mock import MagicMock

from sky_analyzer import (
    estimate_event_specific_z_max,
    generate_sky_map_and_credible_region,
    select_galaxies_in_sky_region,
    filter_galaxies_by_redshift,
    DEFAULT_NSIDE_SKYMAP,
    DEFAULT_PROB_THRESHOLD_CDF,
)




def test_estimate_z_max_valid_input(mocker):
    dl_samples = np.array([30.0, 40.0, 50.0, 60.0, 70.0])
    mock_z = mocker.patch("sky_analyzer.z_at_value", return_value=0.05)

    z_max = estimate_event_specific_z_max(dl_samples, percentile_dL=95.0, z_margin_factor=1.2)

    assert isinstance(z_max, float)
    assert z_max == pytest.approx(0.06)
    called_dl = mock_z.call_args[0][1]
    assert pytest.approx(called_dl.value) == 68.0
    assert str(called_dl.unit) == "Mpc"


@pytest.mark.parametrize("samples", [None, np.array([])])
def test_estimate_z_max_empty_or_none_input(samples, caplog):
    caplog.set_level("WARNING")
    result = estimate_event_specific_z_max(samples)
    assert result == pytest.approx(0.3)
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_estimate_z_max_non_finite_dl_characteristic(mocker, caplog):
    caplog.set_level("WARNING")
    samples = np.array([np.nan, np.inf])
    mock_z = mocker.patch("sky_analyzer.z_at_value", return_value=0.1)

    result = estimate_event_specific_z_max(samples)

    assert result == pytest.approx(0.12)
    called_dl = mock_z.call_args[0][1]
    assert pytest.approx(called_dl.value) == 1000.0
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_estimate_z_max_z_at_value_exception(mocker, caplog):
    caplog.set_level("WARNING")
    samples = np.array([10.0, 20.0, 30.0])
    mocker.patch("sky_analyzer.z_at_value", side_effect=RuntimeError("fail"))

    result = estimate_event_specific_z_max(samples)

    assert result == pytest.approx(0.3)
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


@pytest.mark.parametrize(
    "mock_return,expected",
    [(0.0001, 0.01), (1.0, 0.3)],
)
def test_estimate_z_max_clipping_behavior(mocker, mock_return, expected):
    samples = np.array([10.0, 20.0, 30.0])
    mocker.patch("sky_analyzer.z_at_value", return_value=mock_return)

    result = estimate_event_specific_z_max(samples)

    assert result == pytest.approx(expected)



def test_generate_skymap_valid_inputs():
    ra = np.array([0.0] * 6 + [90.0] * 4)
    dec = np.zeros_like(ra)
    nside = 16
    cdf_threshold = 0.5

    prob_map, mask, threshold = generate_sky_map_and_credible_region(ra, dec, nside=nside, cdf_threshold=cdf_threshold)

    assert isinstance(prob_map, np.ndarray)
    assert prob_map.size == hp.nside2npix(nside)
    assert np.isclose(prob_map.sum(), 1.0)
    assert mask.dtype == bool
    assert mask.size == prob_map.size
    assert isinstance(threshold, float)
    assert np.isclose(prob_map[mask].sum(), cdf_threshold, atol=0.1)


def test_generate_skymap_empty_or_none_samples(caplog):
    caplog.set_level("WARNING")
    ra_valid = np.array([0.0])
    dec_valid = np.array([0.0])
    nside = 8
    res1 = generate_sky_map_and_credible_region(None, dec_valid, nside=nside)
    res2 = generate_sky_map_and_credible_region(np.array([]), np.array([]), nside=nside)

    for prob_map, mask, threshold in (res1, res2):
        assert prob_map.size == hp.nside2npix(nside)
        assert mask.size == hp.nside2npix(nside)
        assert not mask.any()
        assert prob_map.sum() == 0
        assert threshold == 0.0
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_generate_skymap_all_zero_bincount(mocker, caplog):
    caplog.set_level("WARNING")
    ra = np.array([0.0, 1.0])
    dec = np.array([0.0, 1.0])
    nside = 8
    mocker.patch("sky_analyzer.np.bincount", return_value=np.zeros(hp.nside2npix(nside)))

    prob_map, mask, threshold = generate_sky_map_and_credible_region(ra, dec, nside=nside)

    assert np.all(prob_map == 0)
    assert not mask.any()
    assert threshold == 0.0
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_generate_skymap_different_cdf_thresholds():
    ra = np.linspace(0, 1, 10)
    dec = np.linspace(0, 1, 10)
    nside = 8

    pm50, mask50, thresh50 = generate_sky_map_and_credible_region(ra, dec, nside=nside, cdf_threshold=0.5)
    pm95, mask95, thresh95 = generate_sky_map_and_credible_region(ra, dec, nside=nside, cdf_threshold=0.95)

    assert mask95.sum() >= mask50.sum()
    assert thresh95 <= thresh50



def test_filter_galaxies_empty_df():
    df = pd.DataFrame()
    result = filter_galaxies_by_redshift(df, 0.1)
    assert result.empty


def test_filter_galaxies_missing_z_column(caplog):
    caplog.set_level("WARNING")
    df = pd.DataFrame({"ra": [10], "dec": [20]})
    result = filter_galaxies_by_redshift(df, 0.1)
    pd.testing.assert_frame_equal(result, df)
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_filter_galaxies_correct_filtering():
    df = pd.DataFrame({"z": [0.05, 0.1, 0.15, 0.099]})
    result = filter_galaxies_by_redshift(df, 0.1)
    expected = pd.DataFrame({"z": [0.05, 0.099]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)



def test_select_galaxies_empty_input_catalog(caplog):
    caplog.set_level("WARNING")
    nside = 8
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    result = select_galaxies_in_sky_region(pd.DataFrame(), mask, nside=nside)
    assert result.empty
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_select_galaxies_missing_ra_dec_columns(caplog):
    caplog.set_level("WARNING")
    nside = 8
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    df = pd.DataFrame({"PGC": [1], "z": [0.1]})
    result = select_galaxies_in_sky_region(df, mask, nside=nside)
    assert result.empty
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_select_galaxies_invalid_mask(caplog):
    caplog.set_level("WARNING")
    df = pd.DataFrame({"ra": [0], "dec": [0]})
    nside = 8
    mask_none = None
    result_none = select_galaxies_in_sky_region(df, mask_none, nside=nside)
    assert result_none.empty

    wrong_mask = np.zeros(hp.nside2npix(nside) - 1, dtype=bool)
    result_wrong = select_galaxies_in_sky_region(df, wrong_mask, nside=nside)
    assert result_wrong.empty
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_select_galaxies_correct_selection():
    nside = 4
    npix = hp.nside2npix(nside)
    df = pd.DataFrame({
        "PGC": ["A", "B", "C"],
        "ra": [0.0, 90.0, 180.0],
        "dec": [0.0, 0.0, 0.0],
    })
    mask = np.zeros(npix, dtype=bool)
    pa = hp.ang2pix(nside, df.loc[0, "ra"], df.loc[0, "dec"], lonlat=True)
    pb = hp.ang2pix(nside, df.loc[1, "ra"], df.loc[1, "dec"], lonlat=True)
    mask[pa] = True
    mask[pb] = True

    result = select_galaxies_in_sky_region(df, mask, nside=nside)

    assert set(result["PGC"]) == {"A", "B"}


def test_select_galaxies_index_error_handling(mocker, caplog):
    caplog.set_level("ERROR")
    df = pd.DataFrame({"ra": [0.0], "dec": [0.0]})
    nside = 8
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mocker.patch("sky_analyzer.hp.ang2pix", return_value=np.array([mask.size + 10]))

    result = select_galaxies_in_sky_region(df, mask, nside=nside)

    assert result.empty
    assert any("IndexError" in rec.message for rec in caplog.records)


