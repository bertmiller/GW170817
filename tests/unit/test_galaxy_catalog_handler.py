import os
os.environ["ASTROPY_IERS_AUTO_DOWNLOAD"] = "0"
import requests
from unittest.mock import MagicMock
# Prevent external network access triggered by pesummary or gwosc imports
import sys, types
for _mod in ["new","viz","main","plot_utils","analyze_candidates","sky_analyzer","h0_mcmc_analyzer"]:
    sys.modules[_mod] = types.ModuleType(_mod)
    sys.modules[_mod].CONFIG = None

import urllib.request
urllib.request.urlretrieve = MagicMock()
import gwosc.api
gwosc.api.fetch_event_json = MagicMock(return_value={"events": {}})
import pesummary.gw.fetch
pesummary.gw.fetch.fetch_open_samples = MagicMock(return_value=MagicMock(samples_dict={"luminosity_distance": [], "ra": [], "dec": []}))
requests.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {"events": {}}))

import pandas as pd
from galaxy_catalog_handler import download_and_load_galaxy_catalog, CATALOG_CONFIGS, DATA_DIR


def test_download_new_file_glade_plus(mocker, mock_config):
    cat_cfg = CATALOG_CONFIGS['glade+']
    target_path = os.path.join(DATA_DIR, cat_cfg['filename'])

    exists_mock = mocker.patch('galaxy_catalog_handler.os.path.exists', return_value=False)
    urlretrieve_mock = mocker.patch('galaxy_catalog_handler.urllib.request.urlretrieve')
    expected_df = pd.DataFrame({'PGC': [1], 'ra': [1.0], 'dec': [2.0], 'z': [0.01]})
    read_csv_mock = mocker.patch('galaxy_catalog_handler.pd.read_csv', return_value=expected_df)

    result = download_and_load_galaxy_catalog(catalog_type='glade+')

    exists_mock.assert_called_once_with(target_path)
    urlretrieve_mock.assert_called_once_with(cat_cfg['url'], target_path)
    read_csv_mock.assert_called_once_with(
        target_path,
        sep=r"\s+",
        usecols=cat_cfg['use_cols'],
        names=cat_cfg['col_names'],
        comment='#',
        low_memory=False,
        na_values=cat_cfg['na_vals'],
    )
    pd.testing.assert_frame_equal(result, expected_df)


def test_load_existing_file_glade24(mocker, mock_config):
    cat_cfg = CATALOG_CONFIGS['glade24']
    target_path = os.path.join(DATA_DIR, cat_cfg['filename'])

    exists_mock = mocker.patch('galaxy_catalog_handler.os.path.exists', return_value=True)
    urlretrieve_mock = mocker.patch('galaxy_catalog_handler.urllib.request.urlretrieve')
    expected_df = pd.DataFrame({'PGC': [2], 'ra': [3.0], 'dec': [4.0], 'z': [0.02]})
    read_csv_mock = mocker.patch('galaxy_catalog_handler.pd.read_csv', return_value=expected_df)

    result = download_and_load_galaxy_catalog(catalog_type='glade24')

    exists_mock.assert_called_once_with(target_path)
    urlretrieve_mock.assert_not_called()
    read_csv_mock.assert_called_once_with(
        target_path,
        sep=r"\s+",
        usecols=cat_cfg['use_cols'],
        names=cat_cfg['col_names'],
        comment='#',
        low_memory=False,
        na_values=cat_cfg['na_vals'],
    )
    pd.testing.assert_frame_equal(result, expected_df)


def test_download_fails_returns_empty_df(mocker, mock_config, caplog):
    cat_cfg = CATALOG_CONFIGS['glade+']
    target_path = os.path.join(DATA_DIR, cat_cfg['filename'])

    mocker.patch('galaxy_catalog_handler.os.path.exists', return_value=False)
    mocker.patch('galaxy_catalog_handler.urllib.request.urlretrieve', side_effect=Exception('fail'))
    read_csv_mock = mocker.patch('galaxy_catalog_handler.pd.read_csv')
    caplog.set_level('ERROR')

    result = download_and_load_galaxy_catalog(catalog_type='glade+')

    assert result.empty
    read_csv_mock.assert_not_called()
    assert 'Error downloading' in caplog.text


def test_read_csv_fails_returns_empty_df(mocker, mock_config, caplog):
    cat_cfg = CATALOG_CONFIGS['glade+']
    target_path = os.path.join(DATA_DIR, cat_cfg['filename'])

    mocker.patch('galaxy_catalog_handler.os.path.exists', return_value=True)
    mocker.patch('galaxy_catalog_handler.urllib.request.urlretrieve')
    mocker.patch('galaxy_catalog_handler.pd.read_csv', side_effect=Exception('parse'))
    caplog.set_level('ERROR')

    result = download_and_load_galaxy_catalog(catalog_type='glade+')

    assert result.empty
    assert 'Error reading' in caplog.text


def test_unknown_catalog_type_returns_empty_df(mock_config, caplog):
    caplog.set_level('ERROR')
    result = download_and_load_galaxy_catalog(catalog_type='non_existent_catalog')
    assert result.empty
    assert 'Unknown catalog type' in caplog.text
