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
    
    # Use the actual test_GLADE+.txt file as the data source
    log_step("Copying test_GLADE+.txt to temp directory")
    test_data_path = Path(__file__).parent / "test_GLADE+.txt"
    target_file = temp_catalog_dir / "GLADE+.txt"
    shutil.copy(test_data_path, target_file)
    log_step(f"Copied {test_data_path} to {target_file}")
    
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