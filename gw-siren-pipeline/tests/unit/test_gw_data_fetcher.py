import os
import sys
from unittest.mock import MagicMock
import pytest

from gwsiren.gw_data_fetcher import (
    configure_astropy_cache,
    fetch_candidate_data,
    DEFAULT_CACHE_DIR_NAME,
)


@pytest.fixture
def prevent_main_import(monkeypatch):
    """Avoid executing main.py when mock_config imports modules."""
    monkeypatch.setitem(sys.modules, "main", MagicMock())


def test_configure_astropy_cache_creates_dir(temp_data_dir):
    cache_dir = configure_astropy_cache(cache_dir_base=temp_data_dir)
    assert os.path.isdir(cache_dir)


def test_configure_astropy_cache_sets_astropy_config(mocker, temp_data_dir):
    dummy_conf = MagicMock()
    dummy_conf.allow_remote_data = False
    dummy_conf.remote_data_strict = True
    mocker.patch("gwsiren.gw_data_fetcher.astropy_conf", dummy_conf)

    abs_dir = os.path.abspath(temp_data_dir)
    returned = configure_astropy_cache(cache_dir_base=temp_data_dir)

    assert dummy_conf.cache_dir == abs_dir
    assert returned == abs_dir
    assert dummy_conf.allow_remote_data is True


def test_fetch_candidate_data_successful_fetch(prevent_main_import, mocker, mock_config):
    mock_result = MagicMock()
    mock_result.samples_dict = {"key": "val"}
    fetch_mock = mocker.patch(
        "gwsiren.gw_data_fetcher.fetch_open_samples", return_value=mock_result
    )

    success, result = fetch_candidate_data("GW_TEST_SUCCESS", mock_config.catalog["data_dir"])

    fetch_mock.assert_called_once_with(
        "GW_TEST_SUCCESS",
        outdir=mock_config.catalog["data_dir"],
        download_kwargs={"cache": True},
    )
    assert success is True and result is mock_result


def test_fetch_candidate_data_handles_multiple_tables(prevent_main_import, mocker, mock_config):
    msg = (
        "Found multiple posterior sample tables in file.h5: table1.dat, table2.dat. "
        "Not sure which to load."
    )
    mock_result = MagicMock()
    mock_result.samples_dict = {"x": 1}
    fetch_mock = mocker.patch(
        "gwsiren.gw_data_fetcher.fetch_open_samples", side_effect=[Exception(msg), mock_result]
    )

    success, result = fetch_candidate_data("GW_TEST_MULTITABLE", mock_config.catalog["data_dir"])

    assert fetch_mock.call_count == 2
    # Second call should specify the first table name
    second_kwargs = fetch_mock.call_args_list[1].kwargs
    assert second_kwargs.get("path_to_samples") == "table1.dat"
    assert success is True and result is mock_result


def test_fetch_candidate_data_handles_unknown_url(prevent_main_import, mocker, mock_config):
    fetch_mock = mocker.patch(
        "gwsiren.gw_data_fetcher.fetch_open_samples", side_effect=Exception("Unknown URL")
    )

    success, message = fetch_candidate_data("GW_TEST_BADURL", mock_config.catalog["data_dir"])

    fetch_mock.assert_called_once()
    assert success is False
    assert message == "Unknown URL for GW_TEST_BADURL."


def test_fetch_candidate_data_handles_other_exception(prevent_main_import, mocker, mock_config):
    fetch_mock = mocker.patch(
        "gwsiren.gw_data_fetcher.fetch_open_samples",
        side_effect=RuntimeError("Some other PESummary error"),
    )

    success, message = fetch_candidate_data("GW_TEST_OTHER_ERROR", mock_config.catalog["data_dir"])

    fetch_mock.assert_called_once()
    assert success is False
    assert message == "Unexpected error for GW_TEST_OTHER_ERROR: Some other PESummary error"


def test_fetch_candidate_data_empty_samples_dict(prevent_main_import, mocker, mock_config):
    mock_result = MagicMock()
    mock_result.samples_dict = {}
    mocker.patch("gwsiren.gw_data_fetcher.fetch_open_samples", return_value=mock_result)

    success, message = fetch_candidate_data(
        "GW_TEST_EMPTY_SAMPLES", mock_config.catalog["data_dir"]
    )

    assert not success
    assert "no sample_dict" in message
