import os
import sys
import urllib.request
import pandas as pd
import logging
from gwsiren import CONFIG

logger = logging.getLogger(__name__)

# --- Data Directory from central config ---
DATA_DIR = CONFIG.catalog["data_dir"]

# --- Catalog Configurations ---
CATALOG_CONFIGS = {
    'glade+': {
        'url': CONFIG.catalog['glade_plus_url'],
        'filename': "GLADE+.txt",
        # PGC(col 2), RA(col 9), Dec(col 10), z_helio(col 28), placeholder mass proxy column
        'use_cols': [1, 8, 9, 27, 35],
        'col_names': ['PGC', 'ra', 'dec', 'z', 'mass_proxy'],
        'dtypes': {
            'PGC': 'int32',
            'ra': 'float32',
            'dec': 'float32',
            'z': 'float32',
            'mass_proxy': 'float32',
        },
        'na_vals': ['-99.0', '-999.0', '-9999.0', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'N/A', 'n/a', 'None', '...', 'no_value'],
        'display_name': "GLADE+"
    },
    'glade24': {
        'url': CONFIG.catalog['glade24_url'],
        'filename': "GLADE_2.4.txt",
        # PGC, RA, Dec, z, placeholder mass proxy column
        'use_cols': [0, 6, 7, 15, 35],
        'col_names': ['PGC', 'ra', 'dec', 'z', 'mass_proxy'],
        'dtypes': {
            'PGC': 'int32',
            'ra': 'float32',
            'dec': 'float32',
            'z': 'float32',
            'mass_proxy': 'float32',
        },
        'na_vals': ['-99.0', '-999.0', '-9999.0', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'N/A', 'n/a', 'None', '...', 'no_value'],
        'display_name': "GLADE 2.4"
    }
}

# --- Removed old individual constants as they are now in CATALOG_CONFIGS ---
# GLADE_PLUS_URL, GLADE_PLUS_FILE, GLADE_PLUS_USE_COLS, GLADE_PLUS_COL_NAMES, GLADE_PLUS_NA_VALUES
# GLADE24_URL, GLADE24_FILE, GLADE24_USE_COLS, GLADE24_COL_NAMES, GLADE24_NA_VALUES

# Default range checks, can be used by both or overridden
DEFAULT_RANGE_CHECKS = {
    'dec_min': -90, 'dec_max': 90,
    'ra_min': 0, 'ra_max': 360,
    'z_min': 0, 'z_max': 2.0 # Initial broad z range for catalog (can be overridden)
}

# Default specific galaxy corrections (can be extended or overridden by passing a dict to apply_specific_galaxy_corrections)
DEFAULT_GALAXY_CORRECTIONS = {
    "GW170817": {
        "PGC_ID": 45657.0, # NGC 4993
        "LITERATURE_Z": 0.009783
    }
}

def download_and_load_galaxy_catalog(catalog_type='glade+'):
    """
    Downloads (if not present) and loads the specified galaxy catalog using configurations.
    Files will be stored in and read from the DATA_DIR.

    Args:
        catalog_type (str): Type of catalog to load. Keys from CATALOG_CONFIGS (e.g., 'glade+', 'glade24').

    Returns:
        pd.DataFrame: Loaded galaxy catalog, or an empty DataFrame on failure.

    Notes:
        Explicit dtypes are used when reading to minimize parsing overhead and
        memory usage for large catalogs like GLADE+.
    """
    config = CATALOG_CONFIGS.get(catalog_type.lower())

    if not config:
        logger.error(f"❌ Unknown catalog type: {catalog_type}. Available types: {list(CATALOG_CONFIGS.keys())}")
        return pd.DataFrame()

    url = config['url']
    base_filename = config['filename'] # Renamed from filename to base_filename
    use_cols = config['use_cols']
    col_names = config['col_names']
    dtypes = config.get('dtypes')
    na_vals = config['na_vals']
    catalog_name_print = config['display_name']

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    target_filepath = os.path.join(DATA_DIR, base_filename) # Path inside DATA_DIR

    if not os.path.exists(target_filepath):
        logger.info(f"Downloading {catalog_name_print} catalogue ({url}) to {target_filepath} …")
        # Note: GLADE+ is ~6GB, GLADE 2.4 is ~450MB. Download message is generic.
        try:
            urllib.request.urlretrieve(url, target_filepath)
        except Exception as e:
            logger.error(f"❌ Error downloading {catalog_name_print} catalog to {target_filepath}: {e}")
            return pd.DataFrame()

    logger.info(f"Reading {catalog_name_print} from {target_filepath}...")
    try:
        glade_df = pd.read_csv(
            target_filepath,
            sep=r"\s+",
            usecols=use_cols,
            names=col_names,
            dtype=dtypes,
            comment='#',
            low_memory=False,
            na_values=na_vals,
        )
        logger.info(f"  {len(glade_df):,} total rows read from {catalog_name_print} specified columns.")
        return glade_df
    except Exception as e:
        logger.error(f"❌ Error reading {catalog_name_print} catalog: {e}")
        return pd.DataFrame()

def clean_galaxy_catalog(
    glade_df,
    numeric_cols=['PGC', 'ra', 'dec', 'z', 'z_err', 'mass_proxy'],
    cols_to_dropna=['ra', 'dec', 'z', 'mass_proxy'],
    range_filters=DEFAULT_RANGE_CHECKS,
):
    """
    Cleans the galaxy catalog: converts to numeric, drops NaNs, and applies range
    checks. The function now also supports a ``mass_proxy`` column used for
    hierarchical host weighting.

    Args:
        glade_df (pd.DataFrame): The raw galaxy catalog DataFrame.
        numeric_cols (list): Columns to convert to numeric. Defaults include
            ``'z_err'`` and ``'mass_proxy'`` which should represent positive,
            linear quantities (``'mass_proxy'`` being a proxy for stellar mass).
        cols_to_dropna (list): Columns where NaNs should be dropped. ``'mass_proxy'``
            is included by default so galaxies lacking this information are removed.
        range_filters (dict): Dictionary with min/max values for 'ra', 'dec', 'z'.
                              Example: {'dec_min': -90, 'dec_max': 90, ...}

    Notes:
        After initial cleaning, galaxies lacking a valid, positive ``z_err``
        value will have one imputed using ``0.015 * (1 + z)``.

    Returns:
        pd.DataFrame: Cleaned galaxy catalog, or an empty DataFrame if all rows are dropped.
    """
    if glade_df.empty:
        logger.info("  Input DataFrame for cleaning is empty. Returning empty.")
        return pd.DataFrame()

    logger.info("Cleaning GLADE data...")
    df_cleaned = glade_df.copy()

    # ``mass_proxy`` values are assumed to be provided as a positive, linear
    # stellar-mass proxy (e.g., flux rather than magnitude). Any transformation
    # from magnitudes should be applied prior to calling this function.

    for c in numeric_cols:
        if c in df_cleaned:
            # PGC column might contain non-numeric identifiers for some catalogs,
            # but for GLADE it's usually numeric. Coerce errors for robustness.
            df_cleaned[c] = pd.to_numeric(df_cleaned[c], errors='coerce')
        else:
            logger.warning(f"⚠️ Warning: Column {c} not found for numeric conversion during cleaning.")

    initial_count = len(df_cleaned)
    # Only attempt to drop NaNs for columns that actually exist
    subset_for_dropna = [c for c in cols_to_dropna if c in df_cleaned.columns]
    missing_drop_cols = set(cols_to_dropna) - set(subset_for_dropna)
    if missing_drop_cols:
        logger.warning(
            f"⚠ Columns {sorted(missing_drop_cols)} missing for dropna during cleaning."
        )
    df_cleaned = df_cleaned.dropna(subset=subset_for_dropna)
    logger.info(
        f"  {len(df_cleaned):,} galaxies kept after dropping NaNs in {subset_for_dropna} (from {initial_count})."
    )

    # ------------------------------------------------------------------
    # Fallback handling for missing or non-positive z_err values
    # ------------------------------------------------------------------
    if "z_err" in df_cleaned.columns and "z" in df_cleaned.columns:
        fallback_mask = df_cleaned["z_err"].isna() | (df_cleaned["z_err"] <= 0)
        valid_z_mask = df_cleaned["z"].notna()
        apply_mask = fallback_mask & valid_z_mask
        num_fallback = int(apply_mask.sum())
        if num_fallback > 0:
            logger.info(f"Applying z_err fallback for {num_fallback} galaxies...")
            df_cleaned.loc[apply_mask, "z_err"] = 0.015 * (
                1 + df_cleaned.loc[apply_mask, "z"]
            )
            floor_mask = apply_mask & (
                df_cleaned["z_err"].isna() | (df_cleaned["z_err"] <= 0)
            )
            num_floor = int(floor_mask.sum())
            if num_floor > 0:
                logger.info(
                    f"Applying z_err floor for an additional {num_floor} galaxies after fallback calculation."
                )
                df_cleaned.loc[floor_mask, "z_err"] = 0.001
            logger.info("Finished applying z_err fallback logic.")
    else:
        logger.warning(
            "'z_err' or 'z' column not found or not suitable for fallback. Cannot apply z_err fallback logic."
        )

    if df_cleaned.empty:
        logger.info("  No galaxies remaining after dropping NaNs. Cannot proceed with range checks.")
        return df_cleaned

    # Ensure necessary columns for range checks exist after potential drops/coercions
    required_range_cols = ['ra', 'dec', 'z']
    if not all(col in df_cleaned.columns for col in required_range_cols):
        logger.warning(f"  Critical columns for range checks ({required_range_cols}) are missing. Skipping range checks.")
        return df_cleaned
    
    # Filter out rows where any of these critical columns became NaN after coercion but weren't in cols_to_dropna initially
    df_cleaned = df_cleaned.dropna(subset=required_range_cols)
    if df_cleaned.empty:
        logger.info(f"  No galaxies remaining after ensuring {required_range_cols} are not NaN post-coercion.")
        return df_cleaned

    count_before_range_checks = len(df_cleaned)
    
    # Apply range checks
    # Use .get() for flexibility if a filter key is missing, though GLADE_RANGE_CHECKS is fairly standard
    ra_min = range_filters.get('ra_min', 0)
    ra_max = range_filters.get('ra_max', 360)
    dec_min = range_filters.get('dec_min', -90)
    dec_max = range_filters.get('dec_max', 90)
    z_min = range_filters.get('z_min', 0) # Default z_min to 0 if not specified
    z_max = range_filters.get('z_max', 2.0)

    df_cleaned = df_cleaned[
        (df_cleaned['dec'] >= dec_min) & (df_cleaned['dec'] <= dec_max) &
        (df_cleaned['ra'] >= ra_min) & (df_cleaned['ra'] < ra_max) & # Note: ra < ra_max
        (df_cleaned['z'] > z_min) & (df_cleaned['z'] < z_max)       # Note: z > z_min
    ]
    logger.info(f"  {len(df_cleaned):,} clean galaxies kept after range checks (from {count_before_range_checks}).")
    
    if not df_cleaned.empty:
        logger.info("  Sample of cleaned galaxy data (head):")
        # Convert head to string to log it; avoids multi-line issues with some log formatters
        logger.info(df_cleaned.head().to_string())
    else:
        logger.info("  No galaxies remaining after range checks.")
    return df_cleaned

def apply_specific_galaxy_corrections(hosts_df, event_name, corrections_dict=DEFAULT_GALAXY_CORRECTIONS):
    """
    Applies specific redshift corrections for known galaxies for a given event.

    Args:
        hosts_df (pd.DataFrame): DataFrame of candidate host galaxies. Must contain 'PGC' and 'z'.
        event_name (str): The name of the GW event to look up in corrections_dict.
        corrections_dict (dict): Dictionary defining corrections.
                                 Format: {"EVENT_NAME": {"PGC_ID": float, "LITERATURE_Z": float}}

    Returns:
        pd.DataFrame: The hosts_df with corrections applied, or original if no corrections needed/found.
    """
    if event_name not in corrections_dict or hosts_df.empty:
        return hosts_df

    if 'PGC' not in hosts_df.columns or 'z' not in hosts_df.columns:
        logger.warning("⚠️ 'PGC' or 'z' column missing in hosts_df. Cannot apply specific galaxy corrections.")
        return hosts_df
        
    df_corrected = hosts_df.copy()
    correction_info = corrections_dict[event_name]
    pgc_id_to_correct = float(correction_info["PGC_ID"]) # Ensure PGC_ID is float for comparison
    literature_z = correction_info["LITERATURE_Z"]

    # Ensure PGC column is numeric if it's not already, to match pgc_id_to_correct type
    if not pd.api.types.is_numeric_dtype(df_corrected['PGC']):
        df_corrected['PGC'] = pd.to_numeric(df_corrected['PGC'], errors='coerce')
        df_corrected = df_corrected.dropna(subset=['PGC']) # Remove rows where PGC couldn't be coerced

    galaxy_mask = df_corrected['PGC'] == pgc_id_to_correct
    is_galaxy_present = galaxy_mask.any()

    if is_galaxy_present:
        current_z = df_corrected.loc[galaxy_mask, 'z'].iloc[0]
        logger.info(f"\nFound galaxy PGC {pgc_id_to_correct} in candidate hosts for {event_name}.")
        logger.info(f"  Its current redshift from GLADE is: {current_z:.5f}")
        # Apply correction if significantly different or if a policy is to always update
        # Using a small tolerance for floating point comparison
        if abs(current_z - literature_z) > 1e-6: 
            logger.info(f"  Correcting its redshift to the literature value: {literature_z:.5f}")
            df_corrected.loc[galaxy_mask, 'z'] = literature_z
        else:
            logger.info(f"  Its current redshift {current_z:.5f} is close enough to the literature value. No correction applied.")
    else:
        logger.info(f"\nNote: Galaxy PGC {pgc_id_to_correct} (for {event_name} correction) not found in candidate hosts.")
    return df_corrected

if __name__ == '__main__':
    # Configure basic logging for standalone testing
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.info("--- Testing galaxy_catalog_handler.py ---")

    # 1. Test download and load for GLADE+
    logger.info("\n--- Testing GLADE+ ---   ")
    raw_galaxies_plus = download_and_load_galaxy_catalog(catalog_type='glade+')
    # To avoid downloading the large GLADE+ file during routine tests, we can mock or use a small subset.
    # For this example, we'll proceed assuming it might download or use an existing small test file.
    # The function now uses DATA_DIR/CATALOG_CONFIGS['glade+']['filename'] by default.

    if not raw_galaxies_plus.empty:
        logger.info(f"Successfully loaded {len(raw_galaxies_plus)} raw galaxies from GLADE+.")
        test_range_filters_plus = DEFAULT_RANGE_CHECKS.copy()
        test_range_filters_plus['z_max'] = 0.8 # Example z_max for GLADE+
        cleaned_galaxies_plus = clean_galaxy_catalog(
            raw_galaxies_plus,
            range_filters=test_range_filters_plus
        )
        if not cleaned_galaxies_plus.empty:
            logger.info(f"Successfully cleaned GLADE+, {len(cleaned_galaxies_plus)} galaxies remaining.")
        else:
            logger.warning("GLADE+ cleaning resulted in an empty DataFrame.")
    else:
        logger.error("Failed to load raw galaxies from GLADE+. Check connection or file.")

    # 2. Test download and load for GLADE 2.4
    logger.info("\n--- Testing GLADE 2.4 ---    ")
    # The function now uses DATA_DIR/CATALOG_CONFIGS['glade24']['filename'] by default.

    raw_galaxies_24 = download_and_load_galaxy_catalog(catalog_type='glade24')

    if not raw_galaxies_24.empty:
        logger.info(f"Successfully loaded {len(raw_galaxies_24)} raw galaxies from GLADE 2.4.")

        test_range_filters_24 = DEFAULT_RANGE_CHECKS.copy()
        test_range_filters_24['z_max'] = 0.5

        cleaned_galaxies_24 = clean_galaxy_catalog(
            raw_galaxies_24,
            range_filters=test_range_filters_24
        )

        if not cleaned_galaxies_24.empty:
            logger.info(f"Successfully cleaned GLADE 2.4, {len(cleaned_galaxies_24)} galaxies remaining.")

            # 3. Test specific galaxy corrections (using GLADE 2.4 cleaned data as example)
            ngc4993_like_data = {
                'PGC': [DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"], 12345.0],
                'ra': [30.0, 45.0],
                'dec': [10.0, 20.0],
                'z': [0.009000, 0.1]
            }
            dummy_hosts = pd.DataFrame(ngc4993_like_data)

            logger.info("\nTesting galaxy corrections on dummy data...")
            corrected_hosts = apply_specific_galaxy_corrections(dummy_hosts, "GW170817")

            if not corrected_hosts.empty:
                logger.info("Corrections applied (or checked). Resulting dummy data:")
                logger.info("\n" + corrected_hosts.to_string())
                corrected_z_val = corrected_hosts[corrected_hosts['PGC'] == DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"]]['z'].iloc[0]
                expected_z_val = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["LITERATURE_Z"]
                if abs(corrected_z_val - expected_z_val) < 1e-6:
                    logger.info(f"  PGC {DEFAULT_GALAXY_CORRECTIONS['GW170817']['PGC_ID']} redshift successfully corrected to {expected_z_val:.6f}")
                else:
                    logger.error(f"  PGC {DEFAULT_GALAXY_CORRECTIONS['GW170817']['PGC_ID']} redshift is {corrected_z_val:.6f}, expected {expected_z_val:.6f}. Check logic.")

            logger.info("\nTesting corrections for an event not in DEFAULT_GALAXY_CORRECTIONS ('FAKE_EVENT')...")
            uncorrected_hosts = apply_specific_galaxy_corrections(dummy_hosts, "FAKE_EVENT")
            if uncorrected_hosts.equals(dummy_hosts):
                logger.info("  Correctly returned original dataframe as FAKE_EVENT not in corrections dict.")
        else:
            logger.warning("GLADE 2.4 cleaning resulted in an empty DataFrame. Further tests might be affected.")
    else:
        logger.error("Failed to load raw galaxies from GLADE 2.4. Check connection or file.")

    # Clean up test files downloaded by the functions
    logger.info("\n--- Cleaning up downloaded test files ---")
    glade_plus_actual_file = os.path.join(DATA_DIR, CATALOG_CONFIGS['glade+']['filename'])
    if os.path.exists(glade_plus_actual_file):
        os.remove(glade_plus_actual_file)
        logger.info(f"Removed test GLADE+ file: {glade_plus_actual_file}")

    glade24_actual_file = os.path.join(DATA_DIR, CATALOG_CONFIGS['glade24']['filename'])
    if os.path.exists(glade24_actual_file):
        os.remove(glade24_actual_file)
        logger.info(f"Removed test GLADE 2.4 file: {glade24_actual_file}")

    logger.info("\n--- Finished testing galaxy_catalog_handler.py ---") 