import os
import sys
import urllib.request
import pandas as pd

# GLADE Catalog specific constants
GLADE_URL = "https://glade.elte.hu/GLADE-2.4.txt"
GLADE_FILE = "GLADE_2.4.txt" # Default filename
GLADE_USE_COLS = [0, 6, 7, 15]  # PGC, RA, Dec, z
GLADE_COL_NAMES = ['PGC', 'ra', 'dec', 'z']
GLADE_NA_VALUES = ['-99.0', '-999.0', '-9999.0', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'N/A', 'n/a', 'None', '...', 'no_value']
GLADE_RANGE_CHECKS = {
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

def download_and_load_galaxy_catalog(url=GLADE_URL, filename=GLADE_FILE, use_cols=GLADE_USE_COLS, col_names=GLADE_COL_NAMES, na_vals=GLADE_NA_VALUES):
    """
    Downloads (if not present) and loads the galaxy catalog.

    Args:
        url (str): URL to download the catalog from.
        filename (str): Local filename to save/read the catalog.
        use_cols (list): List of column indices to use.
        col_names (list): List of names for the used columns.
        na_vals (list): List of strings to recognize as NaN.

    Returns:
        pd.DataFrame: Loaded galaxy catalog, or an empty DataFrame on failure.
    """
    if not os.path.exists(filename):
        print(f"Downloading GLADE catalogue ({url}, ~450 MB) …")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"❌ Error downloading GLADE catalog: {e}")
            return pd.DataFrame() # Return empty DataFrame on download failure

    print(f"Reading GLADE from {filename}...")
    try:
        glade_df = pd.read_csv(
            filename,
            sep=r"\s+",
            usecols=use_cols,
            names=col_names,
            comment='#',
            low_memory=False,
            na_values=na_vals,
        )
        print(f"  {len(glade_df):,} total rows read from GLADE specified columns.")
        return glade_df
    except Exception as e:
        print(f"❌ Error reading GLADE catalog: {e}")
        return pd.DataFrame() # Return empty DataFrame on read failure

def clean_galaxy_catalog(glade_df, numeric_cols=GLADE_COL_NAMES, cols_to_dropna=['ra', 'dec', 'z'], range_filters=GLADE_RANGE_CHECKS):
    """
    Cleans the galaxy catalog: converts to numeric, drops NaNs, applies range checks.

    Args:
        glade_df (pd.DataFrame): The raw galaxy catalog DataFrame.
        numeric_cols (list): Columns to convert to numeric.
        cols_to_dropna (list): Columns where NaNs should be dropped.
        range_filters (dict): Dictionary with min/max values for 'ra', 'dec', 'z'.
                              Example: {'dec_min': -90, 'dec_max': 90, ...}

    Returns:
        pd.DataFrame: Cleaned galaxy catalog, or an empty DataFrame if all rows are dropped.
    """
    if glade_df.empty:
        print("  Input DataFrame for cleaning is empty. Returning empty.")
        return pd.DataFrame()

    print("Cleaning GLADE data...")
    df_cleaned = glade_df.copy()

    for c in numeric_cols:
        if c in df_cleaned:
            # PGC column might contain non-numeric identifiers for some catalogs,
            # but for GLADE it's usually numeric. Coerce errors for robustness.
            df_cleaned[c] = pd.to_numeric(df_cleaned[c], errors='coerce')
        else:
            print(f"⚠️ Warning: Column {c} not found for numeric conversion during cleaning.")

    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=cols_to_dropna)
    print(f"  {len(df_cleaned):,} galaxies kept after dropping NaNs in {cols_to_dropna} (from {initial_count}).")

    if df_cleaned.empty:
        print("  No galaxies remaining after dropping NaNs. Cannot proceed with range checks.")
        return df_cleaned

    # Ensure necessary columns for range checks exist after potential drops/coercions
    required_range_cols = ['ra', 'dec', 'z']
    if not all(col in df_cleaned.columns for col in required_range_cols):
        print(f"  Critical columns for range checks ({required_range_cols}) are missing. Skipping range checks.")
        return df_cleaned
    
    # Filter out rows where any of these critical columns became NaN after coercion but weren't in cols_to_dropna initially
    df_cleaned = df_cleaned.dropna(subset=required_range_cols)
    if df_cleaned.empty:
        print(f"  No galaxies remaining after ensuring {required_range_cols} are not NaN post-coercion.")
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
    print(f"  {len(df_cleaned):,} clean galaxies kept after range checks (from {count_before_range_checks}).")
    
    if not df_cleaned.empty:
        print("  Sample of cleaned galaxy data (head):")
        print(df_cleaned.head())
    else:
        print("  No galaxies remaining after range checks.")
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
        print("⚠️ 'PGC' or 'z' column missing in hosts_df. Cannot apply specific galaxy corrections.")
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
        print(f"\nFound galaxy PGC {pgc_id_to_correct} in candidate hosts for {event_name}.")
        print(f"  Its current redshift from GLADE is: {current_z:.5f}")
        # Apply correction if significantly different or if a policy is to always update
        # Using a small tolerance for floating point comparison
        if abs(current_z - literature_z) > 1e-6: 
            print(f"  Correcting its redshift to the literature value: {literature_z:.5f}")
            df_corrected.loc[galaxy_mask, 'z'] = literature_z
        else:
            print(f"  Its current redshift {current_z:.5f} is close enough to the literature value. No correction applied.")
    else:
        print(f"\nNote: Galaxy PGC {pgc_id_to_correct} (for {event_name} correction) not found in candidate hosts.")
    return df_corrected

if __name__ == '__main__':
    # Example Usage and Test for this module
    print("--- Testing galaxy_catalog_handler.py ---")
    
    # 1. Test download and load
    # Using a temporary filename for test to avoid overwriting a real GLADE_FILE if it exists
    test_glade_file = "GLADE_2.4_test.txt"
    if os.path.exists(test_glade_file):
        os.remove(test_glade_file)

    raw_galaxies = download_and_load_galaxy_catalog(filename=test_glade_file)
    
    if not raw_galaxies.empty:
        print(f"Successfully loaded {len(raw_galaxies)} raw galaxies for test.")
        
        # 2. Test cleaning
        # Modify range checks for testing purposes to ensure some data passes/fails
        test_range_filters = GLADE_RANGE_CHECKS.copy()
        test_range_filters['z_max'] = 0.5 # A more restrictive z_max for testing
        
        cleaned_galaxies = clean_galaxy_catalog(
            raw_galaxies, 
            range_filters=test_range_filters
        )
        
        if not cleaned_galaxies.empty:
            print(f"Successfully cleaned catalog, {len(cleaned_galaxies)} galaxies remaining.")
            
            # 3. Test specific galaxy corrections
            # Create a dummy event and hosts_df for testing corrections
            # Add a row that matches the GW170817 correction PGC ID
            ngc4993_like_data = {
                'PGC': [DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"], 12345.0], 
                'ra': [30.0, 45.0], 
                'dec': [10.0, 20.0], 
                'z': [0.009000, 0.1] # Slightly different z for the one to be corrected
            }
            dummy_hosts = pd.DataFrame(ngc4993_like_data)
            
            print("\nTesting galaxy corrections on dummy data...")
            corrected_hosts = apply_specific_galaxy_corrections(dummy_hosts, "GW170817")
            
            if not corrected_hosts.empty:
                print("Corrections applied (or checked). Resulting dummy data:")
                print(corrected_hosts)
                # Verify if the redshift was corrected
                corrected_z_val = corrected_hosts[corrected_hosts['PGC'] == DEFAULT_GALAXY_CORRECTIONS["GW170817"]["PGC_ID"]]['z'].iloc[0]
                expected_z_val = DEFAULT_GALAXY_CORRECTIONS["GW170817"]["LITERATURE_Z"]
                if abs(corrected_z_val - expected_z_val) < 1e-6:
                    print(f"  PGC {DEFAULT_GALAXY_CORRECTIONS['GW170817']['PGC_ID']} redshift successfully corrected to {expected_z_val:.6f}")
                else:
                    print(f"  PGC {DEFAULT_GALAXY_CORRECTIONS['GW170817']['PGC_ID']} redshift is {corrected_z_val:.6f}, expected {expected_z_val:.6f}. Check logic.")

            # Test with an event not in corrections
            print("\nTesting corrections for an event not in DEFAULT_GALAXY_CORRECTIONS ('FAKE_EVENT')...")
            uncorrected_hosts = apply_specific_galaxy_corrections(dummy_hosts, "FAKE_EVENT")
            if uncorrected_hosts.equals(dummy_hosts):
                print("  Correctly returned original dataframe as FAKE_EVENT not in corrections dict.")

        else:
            print("Cleaning resulted in an empty DataFrame. Further tests might be affected.")
        
        # Clean up test file
        if os.path.exists(test_glade_file):
            os.remove(test_glade_file)
            print(f"Removed test GLADE file: {test_glade_file}")
            
    else:
        print("Failed to load raw galaxies. Aborting further tests in galaxy_catalog_handler.")
    print("--- Finished testing galaxy_catalog_handler.py ---") 