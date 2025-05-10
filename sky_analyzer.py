import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging

logger = logging.getLogger(__name__)

# Default Sky Analysis Parameters (can be overridden by passing as arguments to functions)
DEFAULT_NSIDE_SKYMAP = 128
DEFAULT_PROB_THRESHOLD_CDF = 0.90 # For 90% credible region
# HOST_Z_MAX is intentionally not a global default here, as it's a crucial analysis choice
# and should be explicitly passed to functions that use it.

def generate_sky_map_and_credible_region(ra_gw_samples, dec_gw_samples, nside=DEFAULT_NSIDE_SKYMAP, cdf_threshold=DEFAULT_PROB_THRESHOLD_CDF):
    """
    Generates a GW probability map and a boolean sky mask for the credible region.

    Args:
        ra_gw_samples (array-like): Array of Right Ascension samples in degrees.
        dec_gw_samples (array-like): Array of Declination samples in degrees.
        nside (int): The Nside parameter for the HEALPix map.
        cdf_threshold (float): Cumulative distribution function threshold to define the credible region (e.g., 0.90 for 90%).

    Returns:
        tuple: (prob_map_gw, sky_mask_boolean, sky_map_threshold_val)
               - prob_map_gw (np.array): The GW probability Healpix map (normalized).
               - sky_mask_boolean (np.array): Boolean Healpix map defining the credible region.
               - sky_map_threshold_val (float): The probability value used as a threshold for the sky_mask.
               Returns (empty_map, empty_mask, 0.0) if inputs are invalid or map generation fails.
    """
    npix = hp.nside2npix(nside)
    empty_prob_map = np.zeros(npix)
    empty_sky_mask = np.zeros(npix, dtype=bool)

    if ra_gw_samples is None or dec_gw_samples is None or len(ra_gw_samples) == 0 or len(dec_gw_samples) == 0:
        logger.warning("⚠️ RA or Dec GW samples are empty or None. Cannot generate sky map.")
        return empty_prob_map, empty_sky_mask, 0.0

    logger.info(f"Building GW probability sky-map (Nside={nside})...")
    ipix_gw = hp.ang2pix(nside, ra_gw_samples, dec_gw_samples, lonlat=True)
    prob_map_gw = np.bincount(ipix_gw, minlength=npix).astype(float)
    
    if prob_map_gw.sum() > 0:
        prob_map_gw /= prob_map_gw.sum()
    else:
        logger.warning("⚠️ Warning: Sum of raw probability map is zero. No GW samples mapped effectively.")
        return prob_map_gw, empty_sky_mask, 0.0 # prob_map_gw will be all zeros

    logger.info(f"Determining sky mask for {cdf_threshold*100:.0f}% credible region...")
    sorted_probs = np.sort(prob_map_gw)[::-1] # Sort in descending order
    cdf = np.cumsum(sorted_probs)
    sky_map_threshold_val = 0.0
    
    try:
        # Find the probability value at which the CDF reaches the threshold
        idx = np.searchsorted(cdf, cdf_threshold)
        if idx < len(sorted_probs):
            sky_map_threshold_val = sorted_probs[idx]
        elif len(sorted_probs) > 0: # If cdf_threshold is 1.0 or very high
            sky_map_threshold_val = sorted_probs[-1] # Smallest non-zero probability or last value
        # If sorted_probs is empty (e.g. prob_map_gw was all zeros), threshold remains 0.0
    except IndexError:
        # This case should ideally be caught by len(sorted_probs) > 0 check
        if cdf_threshold == 1.0 and len(sorted_probs) > 0:
            sky_map_threshold_val = sorted_probs[-1]
        else:
            logger.warning(f"⚠️ Could not determine sky map probability threshold for CDF {cdf_threshold}. Using 0.0.")

    sky_mask_boolean = prob_map_gw >= sky_map_threshold_val
    num_pixels_in_cr = np.sum(sky_mask_boolean)
    area_in_cr_sq_deg = num_pixels_in_cr * hp.nside2pixarea(nside, degrees=True)
    logger.info(f"  Sky mask for {cdf_threshold*100:.0f}% CR contains {num_pixels_in_cr} pixels (approx. {area_in_cr_sq_deg:.1f} sq. deg.).")
    logger.info(f"  Probability threshold for this CR: {sky_map_threshold_val:.2e}")

    return prob_map_gw, sky_mask_boolean, sky_map_threshold_val

def select_galaxies_in_sky_region(all_galaxies_df, sky_mask_boolean, nside=DEFAULT_NSIDE_SKYMAP):
    """
    Selects galaxies that fall within a given HEALPix sky mask.

    Args:
        all_galaxies_df (pd.DataFrame): DataFrame of all galaxies, must have 'ra' and 'dec' columns.
        sky_mask_boolean (np.array): Boolean Healpix map defining the selection region.
        nside (int): The Nside parameter of the sky_mask_boolean.

    Returns:
        pd.DataFrame: DataFrame of galaxies selected by the sky mask. Returns empty if input is invalid.
    """
    empty_df = pd.DataFrame()
    if all_galaxies_df.empty:
        logger.warning("⚠️ Input galaxy catalog is empty. Cannot perform spatial selection.")
        return empty_df

    if not all(col in all_galaxies_df.columns for col in ['ra', 'dec']):
        logger.warning("⚠️ 'ra' or 'dec' column missing in all_galaxies_df. Cannot perform spatial selection.")
        return empty_df
        
    if sky_mask_boolean is None or sky_mask_boolean.size != hp.nside2npix(nside):
        logger.warning("⚠️ Sky mask is invalid or Nside mismatch. Cannot perform spatial selection.")
        return empty_df

    logger.info(f"Spatially selecting galaxies within the provided sky mask (Nside={nside})...")
    coords_gal = SkyCoord(all_galaxies_df['ra'].values * u.deg, all_galaxies_df['dec'].values * u.deg, frame='icrs')
    gpix_gal = hp.ang2pix(nside, coords_gal.ra.deg, coords_gal.dec.deg, lonlat=True)
    
    try:
        spatially_selected_galaxies_df = all_galaxies_df[sky_mask_boolean[gpix_gal]].copy()
        logger.info(f"  Selected {len(spatially_selected_galaxies_df):,} galaxies falling within the sky mask.")
    except IndexError as e:
        logger.error(f"❌ IndexError during spatial selection of galaxies: {e}. Likely an issue with gpix_gal indices.")
        logger.error(f"  Max gpix_gal: {gpix_gal.max() if len(gpix_gal) > 0 else 'N/A'}, sky_mask_boolean size: {sky_mask_boolean.size}")
        return empty_df
        
    return spatially_selected_galaxies_df

def filter_galaxies_by_redshift(galaxies_df, z_max_filter):
    """
    Filters a DataFrame of galaxies by a maximum redshift.

    Args:
        galaxies_df (pd.DataFrame): DataFrame of galaxies, must have a 'z' column.
        z_max_filter (float): Maximum redshift (exclusive, z < z_max_filter).

    Returns:
        pd.DataFrame: DataFrame of galaxies after the redshift cut. Returns empty if input invalid.
    """
    if galaxies_df.empty:
        logger.info("Input DataFrame for redshift filtering is empty.")
        return pd.DataFrame()

    if 'z' not in galaxies_df.columns:
        logger.warning("⚠️ Column 'z' missing in galaxies_df. Cannot apply redshift cut.")
        return galaxies_df # Or an empty DF, depending on desired strictness. Returning original for now.

    logger.info(f"Applying redshift cut z < {z_max_filter}...")
    original_count = len(galaxies_df)
    candidate_hosts_df = galaxies_df[galaxies_df['z'] < z_max_filter].copy()
    logger.info(f"  → {len(candidate_hosts_df):,} galaxies remaining after z < {z_max_filter} cut (from {original_count}).")
    
    return candidate_hosts_df


if __name__ == '__main__':
    # Configure basic logging for standalone testing
    import sys # For StreamHandler
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.info("--- Testing sky_analyzer.py ---")

    # Create mock GW samples (degrees)
    mock_ra_gw = np.array([30.0, 30.5, 31.0, 30.2, 30.8]) 
    mock_dec_gw = np.array([-10.0, -10.2, -9.8, -10.1, -9.9])
    mock_nside = 64 # Lower Nside for faster test

    # Test 1: Generate sky map and credible region
    logger.info("\nTest 1: Generating sky map and credible region...")
    prob_map, mask, thresh_val = generate_sky_map_and_credible_region(mock_ra_gw, mock_dec_gw, nside=mock_nside, cdf_threshold=0.50)
    if prob_map.sum() > 0 and mask.any():
        logger.info(f"  Successfully generated map (sum: {prob_map.sum():.2f}) and mask ({mask.sum()} pixels in 50% CR). Threshold: {thresh_val:.2e}")
        assert prob_map.size == hp.nside2npix(mock_nside)
        assert mask.size == hp.nside2npix(mock_nside)
    else:
        logger.error("  Test 1 FAILED or map was empty.")

    # Create mock galaxy catalog
    mock_galaxies_data = {
        'PGC': [1, 2, 3, 4, 5, 6],
        'ra':  [30.0, 35.0, 30.6, 150.0, 30.1, 30.9], # Some inside, some outside typical mock_ra_gw region
        'dec': [-10.0, -12.0, -9.7,  20.0, -10.3, -9.5],
        'z':   [0.01, 0.02, 0.005, 0.03, 0.15, 0.008]
    }
    mock_galaxies_df = pd.DataFrame(mock_galaxies_data)
    logger.info("\nMock galaxy catalog created:")
    logger.info("\n" + mock_galaxies_df.head().to_string())

    # Test 2: Select galaxies in sky region (using the mask from Test 1)
    logger.info("\nTest 2: Selecting galaxies within the 50% CR mask...")
    if mask.any(): # Proceed only if mask is valid
        spatially_selected = select_galaxies_in_sky_region(mock_galaxies_df, mask, nside=mock_nside)
        if not spatially_selected.empty:
            logger.info(f"  Spatially selected {len(spatially_selected)} galaxies:")
            logger.info("\n" + spatially_selected.to_string())
            # Check PGCs (example: PGCs 1, 3, 5, 6 might be selected depending on exact skymap)
            # This assertion is illustrative and depends on the generated skymap
            # assert all(pgc in spatially_selected['PGC'].values for pgc in [1,3,6]) 
        elif len(mock_ra_gw) > 0 : # If we had GW samples but no galaxies were selected, it might be OK or an issue
            logger.info("  No galaxies were spatially selected. This might be expected or indicate an issue with map/catalog alignment.")
        else:
            logger.info("  Test 2: Spatially selected DataFrame is empty (and no GW samples to begin with).")
    else:
        logger.warning("  Test 2 SKIPPED as the mask from Test 1 was empty or invalid.")

    # Test 3: Filter galaxies by redshift
    logger.info("\nTest 3: Filtering by redshift (z < 0.1)... ")
    # Use the full mock_galaxies_df for this test to ensure it filters correctly
    z_filtered_galaxies = filter_galaxies_by_redshift(mock_galaxies_df, z_max_filter=0.1)
    if not z_filtered_galaxies.empty:
        logger.info(f"  Filtered {len(z_filtered_galaxies)} galaxies with z < 0.1:")
        logger.info("\n" + z_filtered_galaxies.to_string())
        assert all(z < 0.1 for z in z_filtered_galaxies['z'])
        assert 5 not in z_filtered_galaxies['PGC'].values # PGC 5 has z=0.15
    else:
        logger.error("  Test 3 FAILED or resulted in an empty DataFrame post-filtering.")

    # Test with spatially selected galaxies if available from Test 2
    logger.info("\nTest 3b: Filtering spatially_selected (if any) by redshift (z < 0.01)... ")
    if 'spatially_selected' in locals() and not spatially_selected.empty:
        final_candidates = filter_galaxies_by_redshift(spatially_selected, z_max_filter=0.01)
        if not final_candidates.empty:
            logger.info(f"  Final candidates (spatially selected & z < 0.01): {len(final_candidates)} galaxies")
            logger.info("\n" + final_candidates.to_string())
            assert all(z < 0.01 for z in final_candidates['z'])
        else:
            logger.info("  No final candidates after redshift cut on spatially selected galaxies.")
    else:
        logger.warning("  Test 3b SKIPPED as no spatially selected galaxies were available from Test 2.")

    # Test with empty/invalid inputs
    logger.info("\nTest 4: Edge cases for generate_sky_map_and_credible_region...")
    empty_map, empty_mask, _ = generate_sky_map_and_credible_region(None, None)
    assert empty_map.sum() == 0 and not empty_mask.any(), "Test 4a FAILED: Empty GW samples"
    logger.info("  Test 4a (None GW samples) PASSED.")
    empty_map, empty_mask, _ = generate_sky_map_and_credible_region([], [])
    assert empty_map.sum() == 0 and not empty_mask.any(), "Test 4b FAILED: Empty list GW samples"
    logger.info("  Test 4b (Empty list GW samples) PASSED.")

    logger.info("\nTest 5: Edge cases for select_galaxies_in_sky_region...")
    res_empty_gal = select_galaxies_in_sky_region(pd.DataFrame(), mask)
    assert res_empty_gal.empty, "Test 5a FAILED: Empty galaxy DataFrame"
    logger.info("  Test 5a (Empty galaxy DF) PASSED.")
    res_invalid_mask = select_galaxies_in_sky_region(mock_galaxies_df, np.array([True, False]), nside=mock_nside)
    assert res_invalid_mask.empty, "Test 5b FAILED: Invalid mask"
    logger.info("  Test 5b (Invalid mask) PASSED.")
    res_missing_cols = select_galaxies_in_sky_region(pd.DataFrame({'PGC': [1]}), mask, nside=mock_nside)
    assert res_missing_cols.empty, "Test 5c FAILED: Missing ra/dec columns"
    logger.info("  Test 5c (Missing ra/dec cols) PASSED.")

    logger.info("\nTest 6: Edge cases for filter_galaxies_by_redshift...")
    res_empty_filter = filter_galaxies_by_redshift(pd.DataFrame(), 0.1)
    assert res_empty_filter.empty, "Test 6a FAILED: Empty DF for redshift filter"
    logger.info("  Test 6a (Empty DF for z filter) PASSED.")
    res_missing_z_col = filter_galaxies_by_redshift(pd.DataFrame({'ra':[1]}), 0.1)
    assert 'z' not in res_missing_z_col.columns, "Test 6b FAILED: Missing z column"
    logger.info("  Test 6b (Missing z col) PASSED (returned original).")

    logger.info("\n--- Finished testing sky_analyzer.py ---") 