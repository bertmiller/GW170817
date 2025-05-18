import os
from unittest.mock import MagicMock

import pytest

from gw_data_fetcher import (
    configure_astropy_cache,
    fetch_candidate_data,
    DEFAULT_CACHE_DIR_NAME,
)


def test_configure_astropy_cache_creates_dir(temp_data_dir):
    path = configure_astropy_cache(cache_dir_base=temp_data_dir)
    assert os.path.exists(temp_data_dir)
    assert path == os.path.abspath(temp_data_dir)


def test_configure_astropy_cache_sets_astropy_config(mocker, temp_data_dir):
    mock_conf = MagicMock()
    mock_conf.allow_remote_data = False
    mock_conf.remote_data_strict = True
    mocker.patch('gw_data_fetcher.astropy_conf', mock_conf)

    path = configure_astropy_cache(cache_dir_base=temp_data_dir)
    assert mock_conf.cache_dir == os.path.abspath(temp_data_dir)
    assert mock_conf.allow_remote_data is True


def test_fetch_candidate_data_successful_fetch(mocker, mock_config):
    mock_result = MagicMock()
    mock_result.samples_dict = {"key": "value"}
    fetch_mock = mocker.patch('gw_data_fetcher.fetch_open_samples', return_value=mock_result)

    success, result = fetch_candidate_data('GW_TEST_SUCCESS', mock_config.catalog['data_dir'])

    fetch_mock.assert_called_once_with(
        'GW_TEST_SUCCESS',
        outdir=mock_config.catalog['data_dir'],
        download_kwargs={'cache': True}
    )
    assert success is True
    assert result is mock_result


def test_fetch_candidate_data_handles_multiple_tables(mocker, mock_config):
    retry_result = MagicMock()
    retry_result.samples_dict = {'a': 1}

    def side_effect(*args, **kwargs):
        if side_effect.call_count == 0:
            side_effect.call_count += 1
            raise Exception(
                'Found multiple posterior sample tables in file: table1.dat, table2.dat. Not sure which to load.'
            )
        return retry_result

    side_effect.call_count = 0
    fetch_mock = mocker.patch('gw_data_fetcher.fetch_open_samples', side_effect=side_effect)

    success, result = fetch_candidate_data('GW_TEST_MULTITABLE', mock_config.catalog['data_dir'])

    assert fetch_mock.call_count == 2
    assert fetch_mock.call_args_list[1].kwargs.get('path_to_samples') == 'table1.dat'
    assert success is True
    assert result is retry_result


def test_fetch_candidate_data_handles_unknown_url(mocker, mock_config):
    fetch_mock = mocker.patch('gw_data_fetcher.fetch_open_samples', side_effect=Exception('Unknown URL problem'))

    success, msg = fetch_candidate_data('GW_TEST_BADURL', mock_config.catalog['data_dir'])

    fetch_mock.assert_called_once()
    assert success is False
    assert msg == 'Unknown URL for GW_TEST_BADURL.'


def test_fetch_candidate_data_handles_other_exception(mocker, mock_config):
    fetch_mock = mocker.patch('gw_data_fetcher.fetch_open_samples', side_effect=RuntimeError('Some other PESummary error'))

    success, msg = fetch_candidate_data('GW_TEST_OTHER_ERROR', mock_config.catalog['data_dir'])

    fetch_mock.assert_called_once()
    assert success is False
    assert msg == 'Unexpected error for GW_TEST_OTHER_ERROR: Some other PESummary error'


def test_fetch_candidate_data_empty_samples_dict(mocker, mock_config):
    mock_result = MagicMock()
    mock_result.samples_dict = {}
    mocker.patch('gw_data_fetcher.fetch_open_samples', return_value=mock_result)

    success, msg = fetch_candidate_data('GW_TEST_EMPTY_SAMPLES', mock_config.catalog['data_dir'])

    assert success is False
    assert msg.startswith('Fetched GW_TEST_EMPTY_SAMPLES')
