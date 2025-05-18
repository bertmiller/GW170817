import pytest
from pathlib import Path
import pandas as pd
import logging
import os
import sys
import importlib
import time
from datetime import datetime
import shutil
import numpy as np
from gwsiren.data.catalogs import (
    clean_galaxy_catalog,
    apply_specific_galaxy_corrections,
    DEFAULT_GALAXY_CORRECTIONS,
    DEFAULT_RANGE_CHECKS,
)

logger = logging.getLogger(__name__)

def log_step(message):
    """Helper function to log steps with timestamps"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    logger.info(f"[{timestamp}] {message}")

def test_load_glade_plus_from_local_mock_file(mock_config, caplog, monkeypatch):
    """Test loading GLADE+ catalog from a local mock file."""
    # Set up logging capture
    caplog.set_level(logging.INFO)
    
    log_step("Starting test")
    
    # Get the temporary catalog directory path
    log_step("Getting temporary directory path")
    temp_catalog_dir = Path(mock_config.catalog["data_dir"])
    log_step(f"Using temporary directory: {temp_catalog_dir}")
    
    # Ensure the directory exists
    log_step("Creating directory")
    temp_catalog_dir.mkdir(parents=True, exist_ok=True)
    
    # Patch the DATA_DIR in the catalogs module
    log_step("Patching DATA_DIR")
    from gwsiren.data import catalogs
    monkeypatch.setattr(catalogs, "DATA_DIR", str(temp_catalog_dir))
    
    # Create the mock GLADE+ file with the exact test data
    log_step("Creating mock GLADE+ file")
    target_file = temp_catalog_dir / "GLADE+.txt"
    mock_data = """1 43495 NGC4736 NGC4736 12505314+4107125 J125053.14+410712.7 null G 192.721451 41.120152 8.8 null 0 -19.4135 6.084 0.015 5.418 0.015 5.169 0.015 5.611 null 6.12 null 0 null null 0.000990570495285 0.001733 0 null 2.87265e-05 4.392418 null 3 null null 1 null null
2 41934 NGC4548 NGC4548 12352642+1429467 J123526.45+142946.9 null G 188.860123 14.49632 10.85 null 0 -20.1537 8.266 0.016 7.6 0.017 7.368 0.018 9.416 null 9.306 null 0 null null 0.00411957205979 0.0035732712644517 1 0.0007320200807231 0.000119468 15.876007 3.263033 3 0.3 0.2 1 3.7 0.8
3 60921 NGC6503 NGC6503 17492651+7008396 J174926.45+700840.8 null G 267.360474 70.144341 10.28 null 0 -20.1953 8.3 0.016 7.615 0.016 7.382 0.017 10.18 null 10.102 null 0 null null 0.000999810499905 0.0028030488171053 1 0.0002219416591614 2.89945e-05 12.4466 0.987781 3 null null 1 null null
4 40950 NGC4442 NGC4442 12280389+0948130 J122803.90+094813.3 null G 187.01622 9.80362 11.29 null 0 -19.0062 8.317 0.016 7.593 0.016 7.381 0.017 8.476 null 8.504 null 0 null null 0.00355677177839 0.0025816059470629 1 0.0005582724573868 0.000103146 11.461371 2.484465 3 0.68 0.09 0 4.1 0.8
5 41164 NGC4469 NGC4469 null J122928.05+084500.8 null G 187.367 8.74989 12.4 null 0 -18.5065 null null null null null null 9.784 null 9.706 null 0 null null 0.00413868206934 0.003417232088195 1 0.0007453747623336 0.000120022 15.18092 3.321809 3 0.2 0.1 1 3.5 0.8
6 42241 NGC4586 NGC4586 12382843+0419087 J123828.39+041909 null G 189.618484 4.319099 12.05 null 0 -18.0978 9.597 0.021 8.904 0.024 8.634 0.037 10.055 null 10.041 null 0 null null 0.00331989165995 0.0024114067579639 1 0.0003947569965642 9.62769e-05 10.704358 1.756102 3 0.08 0.04 null 3.0 0.8
7 40927 NGC4440 NGC4440 12275357+1217354 J122753.56+121735.8 null G 186.973221 12.293191 12.63 null 0 -18.6497 9.901 0.014 9.225 0.018 9.005 0.021 9.872 null 9.947 null 0 null null 0.00434028217014 0.0040560518230427 1 0.0007966356449514 0.000125868 18.02763 3.553836 3 0.46 0.06 0 3.9 0.8
8 40562 NGC4387 NGC4387 12254171+1248372 J122541.67+124838.1 null G 186.423813 12.810359 12.82 null 0 -18.2542 10.131 0.013 9.439 0.018 9.234 0.02 9.979 null 10.034 null 0 null null 0.00417669208835 0.0036908499636745 1 0.0007748687414523 0.000121124 16.399878 3.454762 3 0.35 0.05 0 3.8 0.8
9 40809 NGC4424 NGC4424 null J122711.57+092513.9 null G 186.799 9.42031 12.02 null 0 -16.9814 null null null null null null 10.814 null 10.638 null 0 null null 0.00134211067106 0.001423400210514 1 0.0002210352913706 3.89212e-05 6.313776 0.98168 3 null null 1 null null
10 39206 NGC4207 NGC4207 12153043+0935057 J121530.38+093505.9 null G 183.876816 9.58494 13.36 null 0 -18.2926 10.562 0.02 9.816 0.023 9.528 0.034 10.294 null 10.16 null 0 null null 0.00464184232092 0.0048133031423996 1 0.0007643283631875 0.000134613 21.40567 3.41354 3 0.3 0.1 1 3.6 0.8"""
    
    with open(target_file, "w") as f:
        f.write(mock_data)
    log_step(f"Mock catalog written to {target_file}")
    
    try:
        # Call the function to load the catalog
        log_step("Calling download_and_load_galaxy_catalog...")
        start_time = time.time()
        df = catalogs.download_and_load_galaxy_catalog(catalog_type='glade+')
        end_time = time.time()
        log_step(f"Catalog loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Perform assertions
        log_step("Performing assertions")
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) == 10, f"Should have exactly 10 rows from test_GLADE+.txt, got {len(df)}"
        assert list(df.columns) == ['PGC', 'ra', 'dec', 'z', 'mass_proxy'], "Column names should match expected"
        # Check a few sample values for correctness
        first_row = df.iloc[0]
        assert first_row['PGC'] == 43495, "First row PGC should match test data"
        assert abs(first_row['ra'] - 192.721451) < 1e-6, "First row RA should match test data"
        assert abs(first_row['dec'] - 41.120152) < 1e-6, "First row Dec should match test data"
        assert abs(first_row['z'] - 0.000990570495285) < 1e-6, "First row z should match test data"
        # mass_proxy may be NaN for some rows, so just check dtype
        assert 'mass_proxy' in df.columns, "mass_proxy column should exist"
        assert pd.api.types.is_numeric_dtype(df['mass_proxy']), "mass_proxy should be numeric"
        log_step("All assertions passed")
        
    except Exception as e:
        log_step(f"Test failed with error: {str(e)}")
        raise
    finally:
        # Clean up the mock file
        log_step("Cleaning up")
        if target_file.exists():
            target_file.unlink()
            log_step("Mock file cleaned up")
        log_step("Test completed")

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