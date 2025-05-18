import os
import pandas as pd
import pytest
from unittest.mock import MagicMock

from galaxy_catalog_handler import download_and_load_galaxy_catalog, CATALOG_CONFIGS, DATA_DIR


def test_download_new_file_glade_plus(mocker, mock_config):
    config = CATALOG_CONFIGS['glade+']
    target_file = os.path.join(DATA_DIR, config['filename'])

    mock_exists = mocker.patch('os.path.exists', return_value=False)
    mock_urlretrieve = mocker.patch('urllib.request.urlretrieve')
    expected_df = pd.DataFrame({'PGC': [1], 'ra': [1.0], 'dec': [2.0], 'z': [0.1]})
    mock_read = mocker.patch('pandas.read_csv', return_value=expected_df)

    result = download_and_load_galaxy_catalog(catalog_type='glade+')

    mock_exists.assert_called_once_with(target_file)
    mock_urlretrieve.assert_called_once_with(config['url'], target_file)
    mock_read.assert_called_once_with(
        target_file,
        sep=r"\s+",
        usecols=config['use_cols'],
        names=config['col_names'],
        comment='#',
        low_memory=False,
        na_values=config['na_vals'],
    )
    pd.testing.assert_frame_equal(result, expected_df)


def test_load_existing_file_glade24(mocker, mock_config):
    config = CATALOG_CONFIGS['glade24']
    target_file = os.path.join(DATA_DIR, config['filename'])

    mock_exists = mocker.patch('os.path.exists', return_value=True)
    mock_urlretrieve = mocker.patch('urllib.request.urlretrieve')
    expected_df = pd.DataFrame({'PGC': [2], 'ra': [3.0], 'dec': [4.0], 'z': [0.2]})
    mock_read = mocker.patch('pandas.read_csv', return_value=expected_df)

    result = download_and_load_galaxy_catalog(catalog_type='glade24')

    mock_exists.assert_called_once_with(target_file)
    mock_urlretrieve.assert_not_called()
    mock_read.assert_called_once_with(
        target_file,
        sep=r"\s+",
        usecols=config['use_cols'],
        names=config['col_names'],
        comment='#',
        low_memory=False,
        na_values=config['na_vals'],
    )
    pd.testing.assert_frame_equal(result, expected_df)


def test_download_fails_returns_empty_df(mocker, mock_config, caplog):
    config = CATALOG_CONFIGS['glade+']
    target_file = os.path.join(DATA_DIR, config['filename'])

    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('urllib.request.urlretrieve', side_effect=Exception('fail'))
    mock_read = mocker.patch('pandas.read_csv')

    with caplog.at_level('ERROR'):
        result = download_and_load_galaxy_catalog(catalog_type='glade+')

    assert result.empty
    mock_read.assert_not_called()
    assert 'Error downloading' in caplog.text


def test_read_csv_fails_returns_empty_df(mocker, mock_config, caplog):
    config = CATALOG_CONFIGS['glade+']
    target_file = os.path.join(DATA_DIR, config['filename'])

    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('urllib.request.urlretrieve')
    mocker.patch('pandas.read_csv', side_effect=Exception('read fail'))

    with caplog.at_level('ERROR'):
        result = download_and_load_galaxy_catalog(catalog_type='glade+')

    assert result.empty
    assert 'Error reading' in caplog.text


def test_unknown_catalog_type_returns_empty_df(mock_config, caplog):
    with caplog.at_level('ERROR'):
        result = download_and_load_galaxy_catalog(catalog_type='non_existent_catalog')

    assert result.empty
    assert 'Unknown catalog type' in caplog.text
