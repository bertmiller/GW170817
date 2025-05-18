import pandas as pd
import numpy as np
import pytest

from galaxy_catalog_handler import clean_galaxy_catalog, apply_specific_galaxy_corrections, DEFAULT_GALAXY_CORRECTIONS, DEFAULT_RANGE_CHECKS


def test_clean_galaxy_catalog_basic():
    df = pd.DataFrame({
        'PGC': ['1', '2', '3'],
        'ra': [10.0, 370.0, 20.0],
        'dec': [0.0, 0.0, -100.0],
        'z': [0.01, 0.02, 0.03]
    })
    filters = DEFAULT_RANGE_CHECKS.copy()
    filters.update({'z_max': 0.05})

    cleaned = clean_galaxy_catalog(df, range_filters=filters)

    # Only first row should remain after range checks (ra<360, dec>-90)
    assert len(cleaned) == 1
    assert np.isclose(cleaned.iloc[0]['ra'], 10.0)
    assert cleaned['PGC'].dtype.kind in {'i', 'f'}


def test_apply_specific_galaxy_corrections():
    pgc_id = DEFAULT_GALAXY_CORRECTIONS['GW170817']['PGC_ID']
    df = pd.DataFrame({
        'PGC': [pgc_id, 99999],
        'ra': [30.0, 40.0],
        'dec': [10.0, 20.0],
        'z': [0.009, 0.1]
    })
    corrected = apply_specific_galaxy_corrections(df, 'GW170817')
    expected_z = DEFAULT_GALAXY_CORRECTIONS['GW170817']['LITERATURE_Z']
    assert corrected.loc[corrected['PGC'] == pgc_id, 'z'].iloc[0] == pytest.approx(expected_z)

