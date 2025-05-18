import numpy as np
import pytest
from unittest.mock import MagicMock

from gwsiren.event_data_extractor import extract_gw_event_parameters


def test_extract_from_ideal_pesummary_object():
    mock_obj = MagicMock()
    mock_obj.samples_dict = {
        "C01:IMRPhenomXPHM": {
            "luminosity_distance": np.array([100.0, 110.0]),
            "ra": np.array([np.pi / 4, np.pi / 2]),
            "dec": np.array([-np.pi / 6, -np.pi / 3]),
        }
    }

    dL, ra, dec = extract_gw_event_parameters(mock_obj, "MOCK_EVENT")

    assert np.allclose(dL, [100.0, 110.0])
    assert np.allclose(ra, np.rad2deg([np.pi / 4, np.pi / 2]))
    assert np.allclose(dec, np.rad2deg([-np.pi / 6, -np.pi / 3]))


def test_extract_from_direct_samples_dictionary():
    data = {
        "luminosity_distance": np.array([200.0, 210.0]),
        "ra": np.array([0.0, np.pi]),
        "dec": np.array([0.1, -0.1]),
    }

    dL, ra, dec = extract_gw_event_parameters(data, "MOCK_EVENT")

    assert np.allclose(dL, [200.0, 210.0])
    assert np.allclose(ra, np.rad2deg([0.0, np.pi]))
    assert np.allclose(dec, np.rad2deg([0.1, -0.1]))


def test_extract_from_dict_of_dicts_selects_valid_analysis(caplog):
    caplog.set_level('INFO')
    data = {
        "analysis_A": {"param": np.array([1])},
        "preferred_analysis_B": {
            "luminosity_distance": np.array([300.0]),
            "ra": np.array([0.2]),
            "dec": np.array([-0.2]),
        },
    }

    dL, ra, dec = extract_gw_event_parameters(data, "MOCK_EVENT")

    assert np.allclose(dL, [300.0])
    assert np.allclose(ra, np.rad2deg([0.2]))
    assert np.allclose(dec, np.rad2deg([-0.2]))
    assert "Using: 'preferred_analysis_B'" in "".join(caplog.messages)


def test_extract_from_dict_of_dicts_fallback_if_no_preferred(caplog):
    caplog.set_level('INFO')
    data = {
        "analysis1": {
            "luminosity_distance": np.array([400.0]),
            "ra": np.array([0.5]),
            "dec": np.array([-0.5]),
        },
        "analysis2": {"param": np.array([1])},
    }

    dL, ra, dec = extract_gw_event_parameters(data, "MOCK_EVENT")

    assert np.allclose(dL, [400.0])
    assert np.allclose(ra, np.rad2deg([0.5]))
    assert np.allclose(dec, np.rad2deg([-0.5]))
    assert "Multiple analyses found in samples_dict for MOCK_EVENT. Using: 'analysis1'" in "".join(caplog.messages)


@pytest.mark.parametrize("missing_key", ["luminosity_distance", "ra", "dec"])
def test_extract_handles_missing_essential_key(missing_key):
    base = {
        "luminosity_distance": np.array([1.0]),
        "ra": np.array([0.1]),
        "dec": np.array([-0.1]),
    }
    del base[missing_key]
    data = {"analysis": base}

    dL, ra, dec = extract_gw_event_parameters(data, "MOCK_EVENT")
    assert dL is None and ra is None and dec is None


def test_extract_handles_empty_samples_dict():
    mock_obj = MagicMock()
    mock_obj.samples_dict = {}

    dL, ra, dec = extract_gw_event_parameters(mock_obj, "MOCK_EVENT")
    assert dL is None and ra is None and dec is None


def test_extract_handles_top_level_empty_dict_or_none():
    dL, ra, dec = extract_gw_event_parameters({}, "MOCK_EVENT")
    assert dL is None and ra is None and dec is None

    dL, ra, dec = extract_gw_event_parameters(None, "MOCK_EVENT")
    assert dL is None and ra is None and dec is None


def test_extract_handles_invalid_input_type():
    dL, ra, dec = extract_gw_event_parameters(42, "MOCK_EVENT")
    assert dL is None and ra is None and dec is None


def test_extract_empty_gw_samples_values():
    data = {
        "luminosity_distance": np.array([]),
        "ra": np.array([]),
        "dec": np.array([]),
    }

    dL, ra, dec = extract_gw_event_parameters(data, "MOCK_EVENT")

    assert isinstance(dL, np.ndarray) and dL.size == 0
    assert isinstance(ra, np.ndarray) and ra.size == 0
    assert isinstance(dec, np.ndarray) and dec.size == 0
