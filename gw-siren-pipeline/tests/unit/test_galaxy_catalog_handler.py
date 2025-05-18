import pandas as pd
import numpy as np
import pytest

from gwsiren.data.catalogs import (
    clean_galaxy_catalog,
    apply_specific_galaxy_corrections,
    DEFAULT_GALAXY_CORRECTIONS,
    DEFAULT_RANGE_CHECKS,
)


def test_clean_empty_dataframe():
    empty_df = pd.DataFrame()
    result = clean_galaxy_catalog(empty_df)
    assert result.empty


def test_clean_numeric_conversion_and_coercion():
    df = pd.DataFrame(
        {
            "PGC": ["123", "bad", 456, None],
            "ra": ["10.5", "bad_val", 15.0, 20.0],
            "dec": ["20.5", 30, 45.0, 50],
            "z": ["0.05", 0.1, "bad", 0.02],
        }
    )

    cleaned = clean_galaxy_catalog(df, numeric_cols=["PGC", "ra", "dec", "z"])

    expected = pd.DataFrame(
        {
            "PGC": [123.0, np.nan],
            "ra": [10.5, 20.0],
            "dec": [20.5, 50.0],
            "z": [0.05, 0.02],
        }
    )

    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), expected)


def test_clean_dropna_specific_columns():
    df = pd.DataFrame(
        {
            "PGC": [1, 2, 3, 4, None],
            "ra": [10.0, np.nan, 20.0, 30.0, 50.0],
            "dec": [0.0, 5.0, np.nan, 10.0, 20.0],
            "z": [0.01, 0.02, 0.03, np.nan, 0.05],
        }
    )

    cleaned = clean_galaxy_catalog(df, cols_to_dropna=["ra", "dec", "z"])

    expected = pd.DataFrame(
        {
            "PGC": [1.0, np.nan],
            "ra": [10.0, 50.0],
            "dec": [0.0, 20.0],
            "z": [0.01, 0.05],
        }
    )

    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), expected)


def test_clean_range_filters_default():
    df = pd.DataFrame(
        {
            "PGC": [1, 2, 3, 4, 5],
            "ra": [10, 370, 20, 30, 40],
            "dec": [0, 0, -100, 10, 20],
            "z": [0.1, 0.1, 0.1, 0.0, 2.5],
        }
    )

    cleaned = clean_galaxy_catalog(df)

    expected = pd.DataFrame({"PGC": [1], "ra": [10], "dec": [0], "z": [0.1]})

    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), expected)


def test_clean_range_filters_custom():
    df = pd.DataFrame(
        {
            "PGC": [1, 2, 3],
            "ra": [15, 50, 120],
            "dec": [0, 10, -20],
            "z": [0.4, 0.6, 0.3],
        }
    )

    filters = {"ra_min": 10, "ra_max": 100, "z_max": 0.5}
    cleaned = clean_galaxy_catalog(df, range_filters=filters)

    expected = pd.DataFrame({"PGC": [1], "ra": [15], "dec": [0], "z": [0.4]})

    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), expected)


def test_clean_missing_critical_range_columns(caplog):
    caplog.set_level("WARNING")
    df = pd.DataFrame({"PGC": [1, 2], "ra": [10, 20], "dec": [0, 10]})

    cleaned = clean_galaxy_catalog(df)

    assert "Critical columns for range checks" in "".join(caplog.messages)
    expected = pd.DataFrame({"PGC": [1, 2], "ra": [10, 20], "dec": [0, 10]})
    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), expected)


def test_clean_all_rows_dropped_after_nans(caplog):
    caplog.set_level("INFO")
    df = pd.DataFrame({"PGC": [1], "ra": [np.nan], "dec": [np.nan], "z": [np.nan]})

    cleaned = clean_galaxy_catalog(df)

    assert cleaned.empty
    assert "No galaxies remaining after dropping NaNs" in "".join(caplog.messages)


def test_clean_all_rows_dropped_after_range_checks(caplog):
    caplog.set_level("INFO")
    df = pd.DataFrame({"PGC": [1], "ra": [370], "dec": [0], "z": [0.1]})

    cleaned = clean_galaxy_catalog(df)

    assert cleaned.empty
    assert "No galaxies remaining after range checks" in "".join(caplog.messages)


def test_apply_correction_success():
    pgc_id = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"]
    df = pd.DataFrame(
        {
            "PGC": [pgc_id, 99999],
            "ra": [30.0, 40.0],
            "dec": [10.0, 20.0],
            "z": [0.009, 0.1],
        }
    )

    corrected = apply_specific_galaxy_corrections(
        df, "GW170817", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    expected_z = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["LITERATURE_Z"]
    assert corrected.loc[corrected["PGC"] == pgc_id, "z"].iloc[0] == pytest.approx(
        expected_z
    )


def test_apply_correction_event_not_in_dict():
    pgc_id = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"]
    df = pd.DataFrame({"PGC": [pgc_id], "z": [0.1]})

    result = apply_specific_galaxy_corrections(
        df, "FAKE_EVENT", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    pd.testing.assert_frame_equal(result, df)


def test_apply_correction_pgc_id_not_in_hosts(caplog):
    caplog.set_level("INFO")
    df = pd.DataFrame({"PGC": [12345], "z": [0.1]})

    result = apply_specific_galaxy_corrections(
        df, "GW170817", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    pd.testing.assert_frame_equal(result, df)
    assert "not found in candidate hosts" in "".join(caplog.messages)


def test_apply_correction_missing_pgc_column(caplog):
    caplog.set_level("WARNING")
    df = pd.DataFrame({"z": [0.1]})

    result = apply_specific_galaxy_corrections(
        df, "GW170817", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    pd.testing.assert_frame_equal(result, df)
    assert "PGC' or 'z' column missing" in "".join(caplog.messages)


def test_apply_correction_missing_z_column(caplog):
    caplog.set_level("WARNING")
    pgc = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"]
    df = pd.DataFrame({"PGC": [pgc]})

    result = apply_specific_galaxy_corrections(
        df, "GW170817", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    pd.testing.assert_frame_equal(result, df)
    assert "PGC' or 'z' column missing" in "".join(caplog.messages)


def test_apply_correction_pgc_column_needs_coercion():
    pgc_id = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"]
    df = pd.DataFrame({"PGC": [str(pgc_id), "foo"], "z": [0.1, 0.2]})

    result = apply_specific_galaxy_corrections(
        df, "GW170817", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    expected_z = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["LITERATURE_Z"]
    assert len(result) == 1
    assert result.iloc[0]["PGC"] == pgc_id
    assert result.iloc[0]["z"] == pytest.approx(expected_z)


def test_apply_correction_z_already_close(caplog):
    caplog.set_level("INFO")
    pgc_id = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"]
    nearly = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["LITERATURE_Z"] + 5e-8
    df = pd.DataFrame({"PGC": [pgc_id], "z": [nearly]})

    result = apply_specific_galaxy_corrections(
        df, "GW170817", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    pd.testing.assert_frame_equal(result, df)
    assert "close enough" in "".join(caplog.messages)


def test_apply_correction_empty_hosts_df():
    empty_df = pd.DataFrame()

    result = apply_specific_galaxy_corrections(
        empty_df, "GW170817", corrections_dict=DEFAULT_GALAXY_CORRECTIONS
    )

    assert result.empty
