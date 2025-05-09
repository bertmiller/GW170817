#!/usr/bin/env python
"""
Visualizes the GW Sky Localization Probability Map using HEALPix.

This script fetches GW posterior samples (RA, Dec) and generates a Mollweide
projection of the sky, where color intensity represents probability density.
"""
import sys
import os # Make sure os is imported
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd # Added for DataFrame operations
from matplotlib.lines import Line2D # Added for custom legend
import urllib.request # For downloading GLADE
from astropy.coordinates import SkyCoord # For select_candidate_hosts
from astropy import units as u # For select_candidate_hosts

# Import from our new module
from gw_data_fetcher import fetch_candidate_data, configure_astropy_cache, DEFAULT_CACHE_DIR_NAME

# Configuration (defaults, can be overridden by command line args)
# GW Event specific
DEFAULT_EVENT_NAME = "GW170608"
# DEFAULT_SAMPLES_PATH_PREFIX removed as gw_data_fetcher handles table selection

# HEALPix Sky Map parameters
NSIDE_SKYMAP = 128  # HEALPix Nside parameter, determines resolution
# CACHE_DIR will be configured by the module

# GLADE Catalog specific
GLADE_URL = "https://glade.elte.hu/GLADE-2.4.txt"
GLADE_FILE = "GLADE_2.4.txt"
# PGC, RA, Dec, z, dist, B, J, H, K (PGC is col 0, B is col 8, dist is col 18 if needed later)
GLADE_USE_COLS = [0, 6, 7, 15]  # PGC, RA, Dec, z
GLADE_COL_NAMES = ['PGC', 'ra', 'dec', 'z']
GLADE_NA_VALUES = ['-99.0', '-999.0', '-9999.0', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'N/A', 'n/a', 'None', '...', 'no_value']
GLADE_RANGE_CHECKS = {
    'dec_min': -90, 'dec_max': 90,
    'ra_min': 0, 'ra_max': 360,
    'z_min': 1e-5, 'z_max': 2.0 # Initial broad z range for catalog, z_min >0 to avoid issues
}

# Analysis specific for host selection
PROB_THRESHOLD_CDF = 0.90 # To define credible region (e.g., 0.90 for 90%)
HOST_Z_MAX = 0.15 # Final redshift cut for candidate hosts

# Specific galaxy corrections (can be extended or moved to a config file)
# Make sure PGC_ID is float if GLADE PGC column becomes float after to_numeric
GALAXY_CORRECTIONS = {
    "GW170817": {
        "PGC_ID": 45657.0, # NGC 4993 (PGC needs to be float if data becomes float)
        "LITERATURE_Z": 0.009783
    }
}

# -------------------------------------------------------------------
# Fetch posterior samples using the gw_data_fetcher module
# -------------------------------------------------------------------
def get_ra_dec_samples_from_event(event_name, configured_cache_dir):
    """
    Fetches GW event data using gw_data_fetcher and extracts RA/Dec samples.

    Args:
        event_name (str): Name of the GW event.
        configured_cache_dir (str): Absolute path to the configured cache directory.

    Returns:
        tuple: (ra_samples, dec_samples) or (None, None) if fetching or extraction fails.
    """
    print(f"Fetching data for {event_name} using gw_data_fetcher module...")
    success, data_or_error = fetch_candidate_data(event_name, configured_cache_dir)

    if not success:
        print(f"❌ Failed to fetch data for {event_name} via module: {data_or_error}")
        return None, None

    # If successful, data_or_error is the pesummary object (or a dict in some cases)
    pesummary_object = data_or_error
    
    try:
        if hasattr(pesummary_object, 'samples_dict') and pesummary_object.samples_dict:
            samples_dict = pesummary_object.samples_dict
        elif isinstance(pesummary_object, dict): # Handles if pesummary_object is already a SamplesDict
            samples_dict = pesummary_object
        else:
            print(f"❌ Fetched data for {event_name}, but it's not a recognized pesummary object or samples dictionary. Type: {type(pesummary_object)}")
            return None, None

        # Try to get a default posterior if samples_dict is a dictionary of posteriors
        # This can happen if the main 'post' object has multiple analyses (e.g. post.samples_dict['C01:IMRPhenomXPHM'])
        # We need to dive one level deeper if necessary.
        # The gw_data_fetcher returns the 'post' object, which might contain multiple analyses,
        # or it might return a specific analysis if a retry with path_to_samples occurred.
        
        actual_samples = None
        if 'ra' in samples_dict and 'dec' in samples_dict: # Direct samples found
            actual_samples = samples_dict
        elif samples_dict: # It's a dict of dicts (multiple analyses)
            # Try to get the first available set of samples
            first_key = next(iter(samples_dict))
            print(f"  Multiple analyses found in samples_dict, using first one: '{first_key}'")
            actual_samples = samples_dict[first_key]
        
        if actual_samples and 'ra' in actual_samples and 'dec' in actual_samples:
            ra_samples = np.rad2deg(actual_samples["ra"])
            dec_samples = np.rad2deg(actual_samples["dec"])
            print(f"  Successfully extracted {len(ra_samples)} RA/Dec samples for {event_name}.")
            return ra_samples, dec_samples
        else:
            print(f"❌ Error: 'ra' or 'dec' keys not found in the extracted samples for {event_name}.")
            print(f"  Available keys in actual_samples: {list(actual_samples.keys()) if actual_samples else 'None'}")
            return None, None
            
    except KeyError as e:
        print(f"❌ Error: Key {e} (e.g., 'ra' or 'dec') not found in posterior samples for {event_name}.")
        return None, None
    except Exception as e:
        print(f"❌ An unexpected error occurred extracting RA/Dec samples for {event_name}: {e}")
        return None, None

# -------------------------------------------------------------------
# Galaxy Catalog Processing (from new.py)
# -------------------------------------------------------------------
def download_and_load_galaxy_catalog(url, filename, use_cols, col_names, na_vals):
    """Downloads (if not present) and loads the galaxy catalog."""
    if not os.path.exists(filename):
        print(f"Downloading GLADE catalogue ({url}, ~450 MB) …")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"❌ Error downloading GLADE catalog: {e}")
            sys.exit(1)

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
        sys.exit(1)

def clean_galaxy_catalog(glade_df, numeric_cols, cols_to_dropna, range_filters):
    """Cleans the galaxy catalog: converts to numeric, drops NaNs, applies range checks."""
    print("Cleaning GLADE data...")
    for c in numeric_cols:
        if c in glade_df:
            # Convert PGC to string first if it might contain non-numeric identifiers,
            # then to numeric with errors='coerce' if it should be purely numeric.
            # For now, assuming PGC can be coerced to numeric for matching with GALAXY_CORRECTIONS PGC_ID
            glade_df[c] = pd.to_numeric(glade_df[c], errors='coerce')
        else:
            print(f"⚠️ Warning: Column {c} not found for numeric conversion.")

    initial_count = len(glade_df)
    glade_df = glade_df.dropna(subset=cols_to_dropna)
    print(f"  {len(glade_df):,} galaxies kept after dropping NaNs in {cols_to_dropna} (from {initial_count}).")

    if glade_df.empty:
        print("  No galaxies remaining after dropping NaNs. Cannot proceed.")
        return pd.DataFrame() # Return empty DataFrame

    count_before_range_checks = len(glade_df)
    # Ensure columns exist before filtering
    if not all(col in glade_df.columns for col in ['ra', 'dec', 'z']):
        print("Critical columns for range checks are missing. Aborting cleaning.")
        return pd.DataFrame()
        
    glade_df = glade_df[
        (glade_df['dec'] >= range_filters['dec_min']) & (glade_df['dec'] <= range_filters['dec_max']) &
        (glade_df['ra'] >= range_filters['ra_min']) & (glade_df['ra'] < range_filters['ra_max']) &
        (glade_df['z'] > range_filters['z_min']) & (glade_df['z'] < range_filters['z_max'])
    ]
    print(f"  {len(glade_df):,} clean galaxies kept after range checks (from {count_before_range_checks}).")
    
    if not glade_df.empty:
        print("  Sample of cleaned galaxy data (head):")
        print(glade_df.head())
    else:
        print("  No galaxies remaining after range checks.")
    return glade_df

# -------------------------------------------------------------------
# Candidate Host Selection and Map Generation (adapted from new.py)
# -------------------------------------------------------------------
def select_candidate_hosts_and_gen_maps(ra_gw_samples, dec_gw_samples, all_galaxy_df, 
                                      nside, cdf_threshold, z_max_filter):
    """
    Generates GW probability map, sky mask for credible region, 
    and selects candidate host galaxies based on sky localization and redshift.
    Returns: prob_map_gw, sky_mask_boolean, selected_hosts_df
    """
    if all_galaxy_df.empty:
        print("⚠️ Galaxy catalog is empty. Cannot select candidate hosts.")
        npix = hp.nside2npix(nside)
        return np.zeros(npix), np.zeros(npix, dtype=bool), pd.DataFrame()

    print("\nBuilding sky-map and selecting candidate hosts...")
    # Generate GW probability map
    ipix_gw = hp.ang2pix(nside, ra_gw_samples, dec_gw_samples, lonlat=True)
    prob_map_gw = np.bincount(ipix_gw, minlength=hp.nside2npix(nside)).astype(float)
    if prob_map_gw.sum() > 0:
        prob_map_gw /= prob_map_gw.sum()
    else:
        print("⚠️ Warning: Sum of probability map is zero. No GW samples mapped.")
        # Return empty results if no GW signal
        npix = hp.nside2npix(nside)
        return prob_map_gw, np.zeros(npix, dtype=bool), pd.DataFrame()

    # Determine sky mask for the credible region
    sorted_probs = np.sort(prob_map_gw)[::-1]
    cdf = np.cumsum(sorted_probs)
    sky_map_threshold_val = 0.0
    try:
        idx = np.searchsorted(cdf, cdf_threshold)
        if idx < len(sorted_probs):
            sky_map_threshold_val = sorted_probs[idx]
        elif len(sorted_probs) > 0: # If cdf_threshold is 1.0 or very high
             sky_map_threshold_val = sorted_probs[-1] # Smallest non-zero probability or last value
        # If sorted_probs is empty (e.g. prob_map_gw was all zeros), threshold remains 0.0
    except IndexError:
        if cdf_threshold == 1.0 and len(sorted_probs) > 0:
            sky_map_threshold_val = sorted_probs[-1]
        else:
            print(f"⚠️ Could not determine sky map threshold for CDF {cdf_threshold}. Using 0.")

    sky_mask_boolean = prob_map_gw >= sky_map_threshold_val
    num_pixels_in_cr = np.sum(sky_mask_boolean)
    area_in_cr_sq_deg = num_pixels_in_cr * hp.nside2pixarea(nside, degrees=True)
    print(f"  Sky mask for {cdf_threshold*100:.0f}% CR contains {num_pixels_in_cr} pixels (approx. {area_in_cr_sq_deg:.1f} sq. deg.).")

    # Select galaxies within this sky_mask_boolean
    # Ensure galaxy_df has ra, dec columns
    if not all(col in all_galaxy_df.columns for col in ['ra', 'dec']):
        print("⚠️ 'ra' or 'dec' missing in all_galaxy_df. Cannot perform spatial selection.")
        return prob_map_gw, sky_mask_boolean, pd.DataFrame()
        
    coords_gal = SkyCoord(all_galaxy_df.ra.values * u.deg, all_galaxy_df.dec.values * u.deg, frame='icrs')
    gpix_gal = hp.ang2pix(nside, coords_gal.ra.deg, coords_gal.dec.deg, lonlat=True)
    
    selected_hosts_df = all_galaxy_df[sky_mask_boolean[gpix_gal]].copy()
    print(f"  Selected {len(selected_hosts_df):,} galaxies within the initial {cdf_threshold*100:.0f}% sky area.")

    # Apply redshift cut to these spatially selected galaxies
    if 'z' not in selected_hosts_df.columns:
        print("⚠️ Column 'z' missing in selected_hosts_df. Cannot apply redshift cut.")
    else:
        selected_hosts_df = selected_hosts_df[selected_hosts_df['z'] < z_max_filter]
        print(f"  → {len(selected_hosts_df):,} candidate host galaxies after z < {z_max_filter} cut.")
    
    return prob_map_gw, sky_mask_boolean, selected_hosts_df

def apply_specific_galaxy_corrections(hosts_df, event_name, corrections_dict):
    """Applies specific redshift corrections for known galaxies for a given event."""
    if event_name not in corrections_dict or hosts_df.empty:
        return hosts_df

    correction_info = corrections_dict[event_name]
    pgc_id_to_correct = correction_info["PGC_ID"]
    literature_z = correction_info["LITERATURE_Z"]

    # Ensure PGC column exists and handle potential type mismatches (e.g. float vs int)
    if 'PGC' not in hosts_df.columns:
        print(f"⚠️ Warning: 'PGC' column not found in hosts_df. Cannot apply correction for {event_name}.")
        return hosts_df
    
    # Make sure PGC_ID to correct is of the same type as the PGC column after pd.to_numeric
    # If hosts_df['PGC'] is float (due to NaNs and coerce), pgc_id_to_correct should be float.
    galaxy_mask = hosts_df['PGC'] == float(pgc_id_to_correct) 
    is_galaxy_present = galaxy_mask.any()

    if is_galaxy_present:
        current_z = hosts_df.loc[galaxy_mask, 'z'].iloc[0]
        print(f"\nFound galaxy PGC {pgc_id_to_correct} in candidate hosts for {event_name}.")
        print(f"  Its current redshift from GLADE is: {current_z:.5f}")
        if abs(current_z - literature_z) > 1e-6: # Tolerance for floating point comparison
            print(f"  Correcting its redshift to the literature value: {literature_z:.5f}")
            hosts_df.loc[galaxy_mask, 'z'] = literature_z
        else:
            print(f"  Its current redshift {current_z:.5f} is close enough to the literature value. No correction applied.")
    else:
        print(f"\nNote: Galaxy PGC {pgc_id_to_correct} (for {event_name} correction) not found in candidate hosts.")
    return hosts_df

# -------------------------------------------------------------------
# Plot sky probability map
# -------------------------------------------------------------------
def plot_sky_probability_map(ra_gw_samples, dec_gw_samples, nside, event_name, output_filename=None):
    """
    Generates and plots a HEALPix sky map from RA and Dec samples.

    Args:
        ra_gw_samples (array-like): Array of Right Ascension samples in degrees.
        dec_gw_samples (array-like): Array of Declination samples in degrees.
        nside (int): The Nside parameter for the HEALPix map.
        event_name (str): Name of the GW event, used for title and filename.
        output_filename (str, optional): Path to save the plot. If None, defaults to
                                         'skymap_<event_name>.pdf'. Plot is also shown.
    """
    if ra_gw_samples is None or dec_gw_samples is None or len(ra_gw_samples) == 0:
        print("⚠️ RA or Dec samples are empty. Cannot generate sky map.")
        return

    print(f"\nGenerating sky map for {event_name} using {len(ra_gw_samples)} samples and Nside={nside}...")

    # Convert RA/Dec from degrees to HEALPix theta/phi (radians)
    # HEALPix uses colatitude (theta) from 0 (North Pole) to pi (South Pole)
    # and longitude (phi) from 0 to 2pi.
    # lonlat=True in ang2pix expects RA (lon) and Dec (lat) in degrees.
    ipix = hp.ang2pix(nside, ra_gw_samples, dec_gw_samples, lonlat=True)

    # Count samples in each pixel
    prob_map = np.bincount(ipix, minlength=hp.nside2npix(nside))

    # Normalize to get probability density per pixel
    total_samples = np.sum(prob_map)
    if total_samples > 0:
        prob_map = prob_map.astype(float) / total_samples
    else:
        print("⚠️ No samples found in any HEALPix pixels. Map will be empty.")
        prob_map = np.zeros(hp.nside2npix(nside)) # Empty map

    # Plotting using Mollweide projection
    plt.figure(figsize=(10, 7))
    
    cmap = plt.cm.viridis 
    cmap.set_under("w") # Set color for pixels with zero probability (or below min) to white

    hp.mollview(
        prob_map,
        title=f"GW Sky Localization Probability Map - {event_name}",
        unit="Probability Density",
        norm="hist", 
        cmap=cmap,
        min=0 
    )
    hp.graticule()

    if output_filename is None:
        output_filename = f"skymap_{event_name}.pdf"

    try:
        plt.savefig(output_filename)
        print(f"Sky map saved to {output_filename}")
    except Exception as e:
        print(f"❌ Error saving sky map: {e}")

    plt.show()

# -------------------------------------------------------------------
# Plot sky probability map with galaxies and credible region
# -------------------------------------------------------------------
def plot_skymap_with_galaxies(
    prob_map_gw,       # The original GW probability map (Healpix array)
    sky_mask_boolean,  # Boolean Healpix map defining the credible region
    all_galaxies_df,   # DataFrame of all cleaned galaxies (must have 'ra', 'dec', 'z', and 'PGC' if distinguishing selected)
    selected_hosts_df, # DataFrame of candidate host galaxies (must have 'ra', 'dec', 'PGC')
    nside,             # Nside for Healpix operations
    event_name,        # Name of the GW event (string)
    cred_level_percent,# Credible level as a percentage (e.g., 90 for 90%)
    host_z_max,        # Redshift cut-off for host galaxies (float)
    output_filename=None # Optional output filename (string)
):
    """
    Plots the GW probability skymap, highlighting the credible region,
    and overlays cataloged galaxies, distinguishing selected candidates.
    """
    plt.figure(figsize=(12, 9)) # Adjusted figure size for legend

    # Create a version of the prob_map that only shows the credible region
    prob_map_credible_region = np.copy(prob_map_gw)
    if not (
        sky_mask_boolean.ndim == 1 and 
        sky_mask_boolean.size == hp.nside2npix(nside) and 
        sky_mask_boolean.dtype == bool
    ):
        print("⚠️ Warning: sky_mask_boolean is not a valid boolean Healpix map. \n            Credible region might not display as intended. Plotting full prob_map_gw instead.")
        # Fallback to plotting the original prob_map if sky_mask_boolean is invalid
        # Or one could choose to plot nothing for the prob_map, or raise an error
    else:
        prob_map_credible_region[~sky_mask_boolean] = hp.UNSEEN # Or a very small number if UNSEEN conflicts with cmap

    title = (
        f"GW Skymap ({event_name}) with Galaxies\n"
        f"{cred_level_percent}% Credible Region & z < {host_z_max:.2f} Selected Hosts"
    )

    hp.mollview(
        prob_map_credible_region,
        title=title,
        unit="Probability Density (in C.R.)",
        norm="hist",
        cmap=plt.cm.Blues, # Using Blues for the skymap to make galaxies stand out
        min=0,
        cbar=True,
        sub=None,
        nest=False # Assuming RING ordering, common for GW skymaps
    )
    hp.graticule()

    galaxies_to_plot_far = pd.DataFrame()
    galaxies_to_plot_other_in_z_cut = pd.DataFrame()

    if not all_galaxies_df.empty:
        # Ensure essential columns exist
        for col in ['ra', 'dec', 'z']:
            if col not in all_galaxies_df.columns:
                print(f"⚠️ Warning: Column '{col}' missing in all_galaxies_df. Cannot plot all galaxies.")
                return

        # Galaxies beyond the redshift cut for hosts
        galaxies_to_plot_far = all_galaxies_df[all_galaxies_df['z'] >= host_z_max]
        if not galaxies_to_plot_far.empty:
            hp.projplot(
                galaxies_to_plot_far['ra'].values, galaxies_to_plot_far['dec'].values,
                '.', lonlat=True, color='gray', alpha=0.5, markersize=2
            )

        # Galaxies within redshift cut, but not selected as hosts
        galaxies_within_z_cut = all_galaxies_df[all_galaxies_df['z'] < host_z_max]
        if not galaxies_within_z_cut.empty:
            if not selected_hosts_df.empty and 'PGC' in galaxies_within_z_cut.columns and 'PGC' in selected_hosts_df.columns:
                pgc_selected = selected_hosts_df['PGC'].astype(str).unique()
                galaxies_within_z_cut_pgc_str = galaxies_within_z_cut['PGC'].astype(str)
                is_not_selected = ~galaxies_within_z_cut_pgc_str.isin(pgc_selected)
                galaxies_to_plot_other_in_z_cut = galaxies_within_z_cut[is_not_selected]
            else:
                # If PGC info is missing for comparison or no selected hosts, all within z-cut are "other"
                galaxies_to_plot_other_in_z_cut = galaxies_within_z_cut
            
            if not galaxies_to_plot_other_in_z_cut.empty:
                hp.projplot(
                    galaxies_to_plot_other_in_z_cut['ra'].values, galaxies_to_plot_other_in_z_cut['dec'].values,
                    '.', lonlat=True, color='darkorange', alpha=0.7, markersize=3 # Changed color for better visibility
                )

    # Plot selected_hosts (most prominent)
    if not selected_hosts_df.empty:
        if not all(col in selected_hosts_df.columns for col in ['ra', 'dec']):
            print("⚠️ Warning: 'ra' or 'dec' missing in selected_hosts_df. Cannot plot selected hosts.")
        else:
            hp.projplot(
                selected_hosts_df['ra'].values, selected_hosts_df['dec'].values,
                '*', lonlat=True, color='red', markersize=10, markeredgecolor='black'
            )

    # Custom Legend
    legend_elements = []
    if not galaxies_to_plot_far.empty:
        legend_elements.append(Line2D([0], [0], marker='.', color='w', 
                                  label=f'Galaxies (z ≥ {host_z_max:.2f})',
                                  markerfacecolor='gray', alpha=0.5, markersize=5))
    if not galaxies_to_plot_other_in_z_cut.empty:
        legend_elements.append(Line2D([0], [0], marker='.', color='w', 
                                  label=f'Other Galaxies (z < {host_z_max:.2f})',
                                  markerfacecolor='darkorange', alpha=0.7, markersize=6))
    if not selected_hosts_df.empty and all(col in selected_hosts_df.columns for col in ['ra', 'dec']):
        legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                  label=f'Selected Hosts (in C.R. & z < {host_z_max:.2f})',
                                  markerfacecolor='red', markeredgecolor='black', markersize=9))

    if legend_elements:
        plt.legend(handles=legend_elements, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize='medium', frameon=True)

    if output_filename is None:
        output_filename = f"skymap_galaxies_{event_name}_cr{int(cred_level_percent)}.pdf"
    
    try:
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Sky map with galaxies saved to {output_filename}")
    except Exception as e:
        print(f"❌ Error saving sky map with galaxies: {e}")
    
    plt.show()

# -------------------------------------------------------------------
# Main script execution
# -------------------------------------------------------------------
def main():
    """Main function to fetch samples, process catalog, and plot sky maps."""
    
    # Configure cache once using the module
    # DEFAULT_CACHE_DIR_NAME is imported from gw_data_fetcher
    effective_cache_dir = configure_astropy_cache(DEFAULT_CACHE_DIR_NAME)
    if not effective_cache_dir:
        print("❌ CRITICAL: Failed to configure cache. Exiting.")
        sys.exit(1)

    current_event_name = DEFAULT_EVENT_NAME
    # current_samples_approximant and related CLI arg removed

    if len(sys.argv) > 1:
        current_event_name = sys.argv[1]
    # Removed sys.argv[2] processing for samples_approximant

    print(f"--- Starting Analysis for Event: {current_event_name} ---")
    # Removed print for samples_approximant

    # 1. Fetch GW posterior samples (RA, Dec) using the new function
    ra_gw_samples, dec_gw_samples = get_ra_dec_samples_from_event(current_event_name, effective_cache_dir)
    
    if ra_gw_samples is None or dec_gw_samples is None:
        print(f"❌ Failed to get posterior samples for {current_event_name}. Exiting.")
        return

    # 2. Plot the basic sky probability map (using original function)
    plot_sky_probability_map(
        ra_gw_samples,
        dec_gw_samples,
        NSIDE_SKYMAP,
        current_event_name,
        output_filename=f"skymap_basic_{current_event_name}_nside{NSIDE_SKYMAP}.pdf"
    )

    print("\n--- Preparing data for skymap with galaxies ---")
    # 3. Load and clean galaxy catalog
    glade_raw_df = download_and_load_galaxy_catalog(
        GLADE_URL, GLADE_FILE, GLADE_USE_COLS, GLADE_COL_NAMES, GLADE_NA_VALUES
    )
    if glade_raw_df.empty:
        print("❌ GLADE catalog is empty after loading. Cannot proceed with galaxy overlay. Exiting.")
        return
        
    glade_cleaned_df = clean_galaxy_catalog(
        glade_raw_df,
        numeric_cols=GLADE_COL_NAMES, # Attempt to convert all loaded columns, PGC might become float
        cols_to_dropna=['ra', 'dec', 'z'], # Essential for sky position and redshift
        range_filters=GLADE_RANGE_CHECKS
    )
    if glade_cleaned_df.empty:
        print("❌ GLADE catalog is empty after cleaning. Cannot proceed with galaxy overlay. Exiting.")
        return

    # 4. Generate GW probability map, sky mask, and select candidate hosts
    # This function now also applies the redshift cut internally as part of selection
    prob_map_gw, sky_mask_boolean, selected_hosts_df = select_candidate_hosts_and_gen_maps(
        ra_gw_samples, dec_gw_samples, glade_cleaned_df,
        NSIDE_SKYMAP, PROB_THRESHOLD_CDF, HOST_Z_MAX
    )

    # 5. Apply specific galaxy corrections (e.g., NGC 4993 for GW170817)
    # This modifies selected_hosts_df in place
    selected_hosts_df = apply_specific_galaxy_corrections(
        selected_hosts_df, current_event_name, GALAXY_CORRECTIONS
    )

    if selected_hosts_df.empty:
        print(f"⚠️ No candidate host galaxies found for {current_event_name} after selection and/or correction. Overlay plot might be sparse.")
    else:
        print(f"Final number of candidate hosts for {current_event_name} for overlay: {len(selected_hosts_df)}")

    # 6. Plot sky map with credible region and galaxies
    cred_level_percent_for_plot = PROB_THRESHOLD_CDF * 100
    plot_skymap_with_galaxies(
        prob_map_gw=prob_map_gw,
        sky_mask_boolean=sky_mask_boolean,
        all_galaxies_df=glade_cleaned_df, # Pass the full cleaned catalog
        selected_hosts_df=selected_hosts_df, # Pass the final selected hosts
        nside=NSIDE_SKYMAP,
        event_name=current_event_name,
        cred_level_percent=cred_level_percent_for_plot,
        host_z_max=HOST_Z_MAX,
        output_filename=f"skymap_galaxies_{current_event_name}_cr{int(cred_level_percent_for_plot)}.pdf"
    )

    print(f"\n--- Script finished for {current_event_name} ---")

if __name__ == "__main__":
    main()
