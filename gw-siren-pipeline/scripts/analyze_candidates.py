#!/usr/bin/env python
"""
Analyzes GW candidates to select potential host galaxies and gathers statistics.

This script iterates through a list of gravitational-wave (GW) event candidates,
fetches their data, determines sky localization and an event-specific redshift cut,
and then cross-matches with the GLADE+ galaxy catalog to find potential host
galaxies. Statistics for each event are compiled and displayed in a table.
This script does not perform MCMC H0 estimation.
"""
import os
import sys
import logging
from gwsiren import CONFIG
import numpy as np
import pandas as pd

# Assuming the script is in the same directory as viz.py and its helper modules,
# or that these modules are in the Python path.
try:
    from gw_data_fetcher import fetch_candidate_data, configure_astropy_cache
    from event_data_extractor import extract_gw_event_parameters
    from gwsiren.data.catalogs import (
        download_and_load_galaxy_catalog,
        clean_galaxy_catalog,
        apply_specific_galaxy_corrections,
        DEFAULT_GALAXY_CORRECTIONS,  # Used as VIZ_GALAXY_CORRECTIONS
        DEFAULT_RANGE_CHECKS,
    )
    from sky_analyzer import (
        generate_sky_map_and_credible_region,
        select_galaxies_in_sky_region,
        filter_galaxies_by_redshift,
        estimate_event_specific_z_max
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(
        "Please ensure that gw_data_fetcher.py, event_data_extractor.py, "
        "gwsiren.data.catalogs, and sky_analyzer.py are accessible "
        "in your Python path or the same directory."
    )
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CANDIDATE_LIST = [
    "GW170817", "GW190814", "GW190425", "GW170814", "GW170818", "GW170608",
    "GW200105_162426", "GW200115_042309", "GW190412",
    "GW190924_021846", "GW200202_154313"
] # From candidates.txt

OUTPUT_DIR = "output_candidate_analysis" # Define a specific output directory for this script
VIZ_CATALOG_TYPE = 'glade+'
VIZ_NSIDE_SKYMAP = CONFIG.skymap["default_nside"]
VIZ_PROB_THRESHOLD_CDF = CONFIG.skymap["credible_level"]
VIZ_HOST_Z_MAX_FALLBACK = 0.05
VIZ_GALAXY_CORRECTIONS = DEFAULT_GALAXY_CORRECTIONS  # Use defaults from gwsiren.data.catalogs


def analyze_candidates():
    """
    Main function to process candidates and display galaxy selection statistics.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensuring output directory exists: {os.path.abspath(OUTPUT_DIR)}")

    effective_cache_dir = configure_astropy_cache(CONFIG.fetcher["cache_dir_name"])
    if not effective_cache_dir:
        logger.critical("❌ CRITICAL: Failed to configure cache. Exiting.")
        sys.exit(1)
    logger.info(f"Using Astropy cache directory: {effective_cache_dir}")

    # 1. Load and clean galaxy catalog (once)
    logger.info(f"--- Loading and Cleaning Galaxy Catalog ({VIZ_CATALOG_TYPE}) ---")
    glade_raw_df = download_and_load_galaxy_catalog(catalog_type=VIZ_CATALOG_TYPE)
    if glade_raw_df.empty:
        logger.error(f"❌ {VIZ_CATALOG_TYPE.upper()} catalog empty after loading. Cannot proceed.")
        return
    
    glade_cleaned_df = clean_galaxy_catalog(glade_raw_df, range_filters=DEFAULT_RANGE_CHECKS)
    if glade_cleaned_df.empty:
        logger.error(f"❌ {VIZ_CATALOG_TYPE.upper()} catalog empty after cleaning. Cannot proceed.")
        return
    num_cleaned_glade_galaxies = len(glade_cleaned_df)
    logger.info(f"Successfully loaded and cleaned {VIZ_CATALOG_TYPE.upper()} catalog: {num_cleaned_glade_galaxies} galaxies.")

    all_event_data = []

    for event_name in CANDIDATE_LIST:
        logger.info(f"\n--- Processing Event: {event_name} ---")
        
        event_metrics = {
            "Event Name": event_name,
            "Num GW Samples (RA/Dec)": 0,
            "Est. Host z_max": np.nan,
            "Num Cleaned GLADE+ Galaxies": num_cleaned_glade_galaxies,
            "Num Spatially Selected (in C.R.)": 0,
            "Num After z-cut (in C.R. & z < z_max)": 0,
            "Num Final Candidates (after corrections)": 0
        }

        try:
            # 2. Fetch GW data and extract parameters
            logger.info(f"Fetching GW data for {event_name}...")
            success, gw_data_obj = fetch_candidate_data(event_name, effective_cache_dir)
            if not success:
                logger.error(f"Failed to fetch GW data for {event_name}: {gw_data_obj}")
                all_event_data.append(event_metrics)
                continue
            
            dL_gw_samples, ra_gw_samples, dec_gw_samples = extract_gw_event_parameters(gw_data_obj, event_name)

            if ra_gw_samples is None or dec_gw_samples is None:
                logger.error(f"Failed to get essential RA/Dec samples for {event_name}.")
                all_event_data.append(event_metrics)
                continue
            event_metrics["Num GW Samples (RA/Dec)"] = len(ra_gw_samples)

            # 3. Estimate event-specific host_z_max
            current_host_z_max_dynamic = VIZ_HOST_Z_MAX_FALLBACK
            if dL_gw_samples is not None and len(dL_gw_samples) > 0:
                current_host_z_max_dynamic = estimate_event_specific_z_max(
                    dL_gw_samples,
                    percentile_dL=95.0, z_margin_factor=1.2,
                    min_z_max_val=0.01, max_z_max_val=0.3 
                )
                logger.info(f"Dynamically estimated host_z_max: {current_host_z_max_dynamic:.4f}")
            else:
                logger.warning(f"dL_gw_samples not available. Using static fallback host_z_max: {current_host_z_max_dynamic:.4f}")
            event_metrics["Est. Host z_max"] = round(current_host_z_max_dynamic, 4)

            # 4. Generate sky map and credible region
            _, sky_mask_for_selection, _ = generate_sky_map_and_credible_region(
                ra_gw_samples, dec_gw_samples,
                nside=VIZ_NSIDE_SKYMAP,
                cdf_threshold=VIZ_PROB_THRESHOLD_CDF
            )

            if sky_mask_for_selection is None or not sky_mask_for_selection.any():
                logger.warning(f"No valid sky mask generated for {event_name}. Galaxy selection in C.R. will be skipped.")
            else:
                # 5. Select galaxies in sky region
                spatially_selected_galaxies_df = select_galaxies_in_sky_region(
                    glade_cleaned_df, sky_mask_for_selection, nside=VIZ_NSIDE_SKYMAP
                )
                event_metrics["Num Spatially Selected (in C.R.)"] = len(spatially_selected_galaxies_df)
                logger.info(f"Found {len(spatially_selected_galaxies_df)} galaxies in the credible region.")

                if not spatially_selected_galaxies_df.empty:
                    # 6. Filter by redshift
                    galaxies_after_z_cut_df = filter_galaxies_by_redshift(
                        spatially_selected_galaxies_df, current_host_z_max_dynamic
                    )
                    event_metrics["Num After z-cut (in C.R. & z < z_max)"] = len(galaxies_after_z_cut_df)
                    logger.info(f"Found {len(galaxies_after_z_cut_df)} galaxies after redshift cut (z < {current_host_z_max_dynamic:.4f}).")

                    if not galaxies_after_z_cut_df.empty:
                        # 7. Apply specific galaxy corrections
                        final_candidate_hosts_df = apply_specific_galaxy_corrections(
                            galaxies_after_z_cut_df, event_name, VIZ_GALAXY_CORRECTIONS
                        )
                        event_metrics["Num Final Candidates (after corrections)"] = len(final_candidate_hosts_df)
                        logger.info(f"Found {len(final_candidate_hosts_df)} final candidate hosts after corrections.")
                    else:
                        logger.info("No galaxies remained after redshift cut for final corrections.")
                else:
                    logger.info("No galaxies spatially selected to apply redshift cut or corrections.")
        
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while processing {event_name}: {e}", exc_info=True)
        
        all_event_data.append(event_metrics)

    # 8. Display results in a table
    logger.info("\n\n--- Candidate Host Galaxy Selection Summary ---")
    if not all_event_data:
        logger.info("No data processed to display.")
        return

    results_df = pd.DataFrame(all_event_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000) 
    
    # Reorder columns for clarity if needed
    column_order = [
        "Event Name", "Num GW Samples (RA/Dec)", "Est. Host z_max",
        "Num Cleaned GLADE+ Galaxies", "Num Spatially Selected (in C.R.)",
        "Num After z-cut (in C.R. & z < z_max)", "Num Final Candidates (after corrections)"
    ]
    results_df = results_df[column_order]

    logger.info("\n" + results_df.to_string(index=False))

    # Optionally, save to CSV
    csv_filename = os.path.join(OUTPUT_DIR, "candidate_host_selection_summary.csv")
    try:
        results_df.to_csv(csv_filename, index=False)
        logger.info(f"\nSummary table saved to: {csv_filename}")
    except Exception as e:
        logger.error(f"Error saving summary table to CSV: {e}")

if __name__ == "__main__":
    analyze_candidates()