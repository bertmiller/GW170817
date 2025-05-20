#!/usr/bin/env python
"""
Analyzes GW candidates to select potential host galaxies and gathers statistics.

This script first applies a set of selection criteria (S1-S5) to a list of 
gravitational-wave (GW) event candidates fetched from GWOSC. For the selected
candidates, it then fetches their data, determines sky localization and an 
event-specific redshift cut, and cross-matches with the GLADE+ galaxy catalog 
to find potential host galaxies. Statistics for each analyzed event are compiled 
and displayed in a table. This script does not perform MCMC H0 estimation.
"""
# Standard library imports
import json
import logging
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd
import requests

# Local application/library specific imports
from gwsiren import CONFIG

# Assuming the script is in the same directory as viz.py and its helper modules,
# or that these modules are in the Python path.
try:
    from gwsiren.gw_data_fetcher import fetch_candidate_data, configure_astropy_cache
    from gwsiren.event_data_extractor import extract_gw_event_parameters
    from gwsiren.data.catalogs import (
        download_and_load_galaxy_catalog,
        clean_galaxy_catalog,
        apply_specific_galaxy_corrections,
        DEFAULT_GALAXY_CORRECTIONS,  # Used as VIZ_GALAXY_CORRECTIONS
        DEFAULT_RANGE_CHECKS,
    )
    from gwsiren.sky_analyzer import (
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
# CANDIDATE_LIST will be dynamically generated based on S1-S5 criteria.

OUTPUT_DIR = "output_candidate_analysis" # Define a specific output directory for this script
VIZ_CATALOG_TYPE = 'glade+'
VIZ_NSIDE_SKYMAP = CONFIG.skymap["default_nside"]
VIZ_PROB_THRESHOLD_CDF = CONFIG.skymap["credible_level"]
VIZ_HOST_Z_MAX_FALLBACK = 0.05
VIZ_GALAXY_CORRECTIONS = DEFAULT_GALAXY_CORRECTIONS

# --- GWOSC API and Selection Criteria Configuration ---
GWOSC_API_BASE = "https://gwosc.org/eventapi/v1.0"
# S1: Confident event catalogs from GWOSC
CONFIDENT_CATALOGS_S1 = ["GWTC-2.1-confident", "GWTC-3-confident"]
# S2: Distance cut
MAX_DISTANCE_MPC_S2 = 2500.0
# S3: Sky area cut
MAX_SKY_AREA_S3 = 3000.0
# S5: Known bright sirens
BRIGHT_SIRENS_S5 = ["GW170817"]


def fetch_all_confident_event_names_s1():
    """
    Fetches a list of confident event names from GWOSC based on CONFIDENT_CATALOGS_S1.
    (Implements S1)
    """
    all_event_names = set()
    for catalog_search_key in CONFIDENT_CATALOGS_S1:
        try:
            # Corrected URL structure based on GWOSC API documentation for /eventapi/jsonfull/<cat_name>/
            url = f"https://gwosc.org/eventapi/jsonfull/{catalog_search_key}/" # Using the documented base for event portal
            logger.info(f"Fetching event names from GWOSC: {url}")
            response = requests.get(url, timeout=60) # Increased timeout slightly
            response.raise_for_status()
            data = response.json()
            
            # Expected structure: data["events"] = {"EVENT_NAME_1-vX": {...}, "EVENT_NAME_2-vY": {...}}
            # We need to extract the event names (commonName or the dict key without version)
            events_dict = data.get("events")
            
            retrieved_names_for_catalog = []
            if events_dict and isinstance(events_dict, dict):
                for event_version_key, event_details in events_dict.items():
                    # Attempt to get commonName, otherwise parse from key
                    common_name = event_details.get("commonName")
                    if common_name:
                        retrieved_names_for_catalog.append(common_name)
                    else:
                        # Fallback: parse from key like "GW190425-v1" -> "GW190425"
                        # This is a simplified parsing, assuming standard naming
                        name_part = event_version_key.split('-v')[0]
                        retrieved_names_for_catalog.append(name_part)
                # Remove duplicates that might arise if commonName and parsed key are the same
                retrieved_names_for_catalog = sorted(list(set(retrieved_names_for_catalog)))
            else:
                logger.warning(
                    f"Could not find 'events' dictionary for {catalog_search_key} in the response or it's not a dict. "
                    f"Response keys: {data.keys()}, events content type: {type(events_dict)}"
                )
            
            if retrieved_names_for_catalog:
                logger.info(f"Fetched {len(retrieved_names_for_catalog)} event names from {catalog_search_key}")
                all_event_names.update(retrieved_names_for_catalog)
            else:
                logger.warning(f"No event names retrieved for {catalog_search_key}. Response: {data}")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error fetching event names for {catalog_search_key}: {e.response.status_code} - {e.response.text}")
        except requests.RequestException as e: # Covers ConnectTimeout, ReadTimeout etc.
            logger.error(f"RequestException fetching event names for {catalog_search_key}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {catalog_search_key} events: {e}")
            
    if not all_event_names:
        logger.warning("S1 Fetch: No confident events could be fetched from any specified GWOSC catalog.")
    else:
        logger.info(f"S1 Fetch: Total {len(all_event_names)} unique confident events fetched from {', '.join(CONFIDENT_CATALOGS_S1)}.")
    return sorted(list(all_event_names))


def get_event_filter_criteria(event_name, effective_cache_dir):
    """
    Fetches and extracts metadata for a single event relevant to S2, S3, S4 filters.
    - S2: Median luminosity distance (from PE samples).
    - S3: Sky area A_90 (from event metadata).
    - S4: BNS/NSBH classification (from event metadata, HasNS or classification probabilities).
    """
    median_dL_mpc = np.nan
    sky_area_90 = np.nan
    is_bns_or_nsbh_s4 = False

    # --- Caching for event metadata JSON ---
    cache_filename = f"{event_name}_metadata.json"
    cache_filepath = os.path.join(effective_cache_dir, cache_filename)
    event_json_data = None

    if os.path.exists(cache_filepath):
        logger.debug(f"Attempting to load cached metadata for {event_name} from {cache_filepath}")
        try:
            with open(cache_filepath, 'r') as f_cache:
                event_json_data = json.load(f_cache)
            logger.info(f"Successfully loaded cached metadata for {event_name}.")
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load cached metadata for {event_name} from {cache_filepath}: {e}. Will re-fetch.")
            event_json_data = None # Ensure it's None so we re-fetch

    # --- Fetching logic (if not cached or cache load failed) ---
    if event_json_data is None:
        try:
            # Fetch main event JSON from GWOSC API (provides sky_area, classification)
            event_api_url = f"https://gwosc.org/eventapi/json/event/{event_name}/" 
            logger.debug(f"Fetching metadata for {event_name} from {event_api_url}")
            response = requests.get(event_api_url, timeout=30)
            response.raise_for_status()
            event_json_data = response.json()
            logger.info(f"Successfully fetched metadata for {event_name} from API.")

            # Save to cache
            try:
                with open(cache_filepath, 'w') as f_cache:
                    json.dump(event_json_data, f_cache, indent=4)
                logger.info(f"Saved metadata for {event_name} to cache: {cache_filepath}")
            except IOError as e:
                logger.warning(f"Failed to save metadata for {event_name} to cache {cache_filepath}: {e}")
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Filter Criteria Fetch ({event_name}): GWOSC API returned 404 from {event_api_url}. No metadata to cache.")
            else:
                logger.error(f"Filter Criteria Fetch ({event_name}): HTTP Error from {event_api_url}: {e.response.status_code} - {e.response.text}")
            return median_dL_mpc, sky_area_90, is_bns_or_nsbh_s4 # Return defaults if API fetch fails
        except requests.RequestException as e:
            logger.error(f"Filter Criteria Fetch ({event_name}): RequestException (e.g. timeout) from {event_api_url}: {e}")
            return median_dL_mpc, sky_area_90, is_bns_or_nsbh_s4
        except (json.JSONDecodeError, TypeError, ValueError) as e: 
            logger.error(f"Filter Criteria Fetch ({event_name}): Data Parsing Error (JSON/Type/Value) for API response: {e}")
            return median_dL_mpc, sky_area_90, is_bns_or_nsbh_s4
        except Exception as e:
            logger.error(f"Filter Criteria Fetch ({event_name}): Unexpected error during API fetch: {e}", exc_info=False)
            return median_dL_mpc, sky_area_90, is_bns_or_nsbh_s4

    # --- Processing (now using event_json_data which is either from cache or fresh) ---
    if not event_json_data:
        logger.warning(f"No event_json_data available for {event_name} (either from cache or API). Cannot extract criteria.")
        return median_dL_mpc, sky_area_90, is_bns_or_nsbh_s4

    try:
        # Event data is usually nested under "events": {"EVENT_NAME" (possibly with -vX): {...}}
        # The event_json_data from /eventapi/json/event/{event_name}/ should directly contain the details
        # for the specific event, potentially nested under an event key (that might include version).
        
        actual_event_data_key = None
        potential_event_data_root = event_json_data.get("events", {})
        
        if event_name in potential_event_data_root: # Exact match (e.g. GW170817)
            actual_event_data_key = event_name
        else: # Check for keys like GW170817-v1, etc.
            for key in potential_event_data_root.keys():
                if key.startswith(event_name): # e.g. key is GW170817-v1 and event_name is GW170817
                    actual_event_data_key = key
                    break
        
        if actual_event_data_key:
            event_data_dict = potential_event_data_root.get(actual_event_data_key)
        else: # Fallback: sometimes the root of the JSON is the event data itself
            event_data_dict = event_json_data 
            # This case might occur if /eventapi/json/event/EVENT_NAME/ returns the event data directly
            # without the top-level "events" nesting if there's only one event version for that name.
            # We check if it looks like an event dict by checking for a common key like 'commonName' or 'parameters'
            if not (event_data_dict.get("commonName") == event_name or "parameters" in event_data_dict or "GPS" in event_data_dict):
                 # If it doesn't look like event data, try to find it within a possible single-item dict from 'events'
                if isinstance(potential_event_data_root, dict) and len(potential_event_data_root) == 1:
                    event_data_dict = list(potential_event_data_root.values())[0]
                else:
                    event_data_dict = None # Could not reliably determine event data root

        if not event_data_dict:
            logger.warning(f"No data found for event '{event_name}' in GWOSC API response (after cache/fetch). Tried direct key and iterating. Data: {event_json_data}")
            return median_dL_mpc, sky_area_90, is_bns_or_nsbh_s4

        # S3: Sky Area (A_90)
        params_dict = event_data_dict.get("parameters", {})
        sky_loc_dict = event_data_dict.get("sky_localization", {})

        if "sky_area_90" in params_dict:
            sky_area_90 = float(params_dict["sky_area_90"])
        elif "sky_area_90_deg2" in sky_loc_dict: # Another common key
            sky_area_90 = float(sky_loc_dict["sky_area_90_deg2"])
        elif "sky_area" in params_dict: # More generic fallback
            sky_area_90 = float(params_dict["sky_area"])
        else:
            logger.warning(f"S3 Filter: Could not find sky_area_90 for {event_name}. Searched in 'parameters' and 'sky_localization'.")

        # S4: BNS/NSBH classification
        ext_coinc_dict = event_data_dict.get("external_coinc", {})
        has_ns_val = ext_coinc_dict.get("HasNS")

        if has_ns_val is not None: # Prioritize HasNS if available
            if isinstance(has_ns_val, (int, float)): is_bns_or_nsbh_s4 = has_ns_val > 0.0
            elif isinstance(has_ns_val, str): is_bns_or_nsbh_s4 = has_ns_val.lower() == "true"
            else: is_bns_or_nsbh_s4 = bool(has_ns_val) # Assume boolean otherwise
            logger.debug(f"S4 Filter ({event_name}): HasNS = {has_ns_val}, interpreted as BNS/NSBH: {is_bns_or_nsbh_s4}")
        else: # Fallback to classification probabilities
            classification_dict = params_dict.get("classification", {})
            if not classification_dict and "classification" in event_data_dict: # Check top-level too
                 classification_dict = event_data_dict["classification"]
            
            if classification_dict:
                p_bns = float(classification_dict.get("BNS", 0.0))
                p_nsbh = float(classification_dict.get("NSBH", 0.0))
                if p_bns > 0.5 or p_nsbh > 0.5: # Threshold for probability
                    is_bns_or_nsbh_s4 = True
                logger.debug(f"S4 Filter ({event_name}): P(BNS)={p_bns:.2f}, P(NSBH)={p_nsbh:.2f}. Classified BNS/NSBH: {is_bns_or_nsbh_s4}")
            else:
                logger.warning(f"S4 Filter: No 'HasNS' or 'classification' data found for {event_name}.")
        
        # S2: Median Luminosity Distance (requires PE samples)
        # This part is potentially slow as it fetches PE files.
        logger.debug(f"S2 Filter ({event_name}): Fetching PE data to determine median dL.")
        success_pe, gw_pe_data_obj = fetch_candidate_data(event_name, effective_cache_dir)
        if success_pe:
            dL_samples, _, _ = extract_gw_event_parameters(gw_pe_data_obj, event_name)
            if dL_samples is not None and len(dL_samples) > 0:
                median_dL_mpc = np.median(dL_samples)
                logger.debug(f"S2 Filter ({event_name}): Median dL = {median_dL_mpc:.2f} Mpc from {len(dL_samples)} samples.")
            else:
                logger.warning(f"S2 Filter ({event_name}): Could not extract dL_samples (samples empty/None).")
        else:
            logger.warning(f"S2 Filter ({event_name}): Failed to fetch/process PE data: {gw_pe_data_obj}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Filter Criteria Fetch ({event_name}): GWOSC API returned 404 from {event_api_url}.")
        else:
            logger.error(f"Filter Criteria Fetch ({event_name}): HTTP Error from {event_api_url}: {e.response.status_code} - {e.response.text}")
    except requests.RequestException as e:
        logger.error(f"Filter Criteria Fetch ({event_name}): RequestException (e.g. timeout) from {event_api_url}: {e}")
    except (json.JSONDecodeError, TypeError, ValueError) as e: # Catch parsing/conversion errors
        logger.error(f"Filter Criteria Fetch ({event_name}): Data Parsing Error (JSON/Type/Value): {e}")
    except Exception as e:
        logger.error(f"Filter Criteria Fetch ({event_name}): Unexpected error: {e}", exc_info=False) # Keep log cleaner

    return median_dL_mpc, sky_area_90, is_bns_or_nsbh_s4


def analyze_candidates():
    """
    Main function to select candidates based on S1-S5 criteria, then process
    them and display galaxy selection statistics.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensuring output directory exists: {os.path.abspath(OUTPUT_DIR)}")

    effective_cache_dir = configure_astropy_cache(CONFIG.fetcher["cache_dir_name"])
    if not effective_cache_dir:
        logger.critical("❌ CRITICAL: Failed to configure cache. Exiting.")
        sys.exit(1)
    logger.info(f"Using Astropy cache directory: {effective_cache_dir}")

    # --- Candidate Selection (S1-S5) ---
    logger.info("\n--- Applying Candidate Selection Criteria (S1-S5) ---")
    
    # S1: Get all confident event names
    initial_event_names_s1 = fetch_all_confident_event_names_s1()
    if not initial_event_names_s1:
        logger.error("❌ No confident events fetched (S1). Cannot proceed with analysis.")
        return

    selected_candidates_for_analysis = []
    bright_sirens_identified = []
    
    selection_stats = {
        "0_Initial_Confident_Events_S1": len(initial_event_names_s1),
        "1_S5_Bright_Sirens_Separated": 0,
        "2_S4_Kept_BNS_NSBH": 0,
        "3_Passed_S2_Distance_Cut (Non-BNS/NSBH)": 0,
        "4_Passed_S3_SkyArea_Cut (Non-BNS/NSBH, Post-S2)": 0,
        "5_Final_Selected_Dark_Sirens_for_Analysis": 0
    }

    logger.info(f"Processing {len(initial_event_names_s1)} initial confident events for filtering...")
    for event_name in initial_event_names_s1:
        logger.info(f"\nFiltering event: {event_name}")

        # S5: Treat bright sirens separately
        if event_name in BRIGHT_SIRENS_S5:
            logger.info(f"S5 Filter: Event {event_name} is a known bright siren. Separating.")
            bright_sirens_identified.append(event_name)
            selection_stats["1_S5_Bright_Sirens_Separated"] += 1
            continue # Skip further dark siren filtering for bright sirens

        # Fetch metadata for S2, S3, S4 filters
        # This call involves network requests and PE data processing, can be slow per event.
        median_dL_mpc, sky_area_90, is_bns_or_nsbh = get_event_filter_criteria(event_name, effective_cache_dir)

        # S4: Always keep BNS/NSBH candidates (unless it's a bright siren already handled)
        if is_bns_or_nsbh:
            logger.info(f"S4 Filter: Event {event_name} is BNS/NSBH. Adding to analysis list.")
            selected_candidates_for_analysis.append(event_name)
            selection_stats["2_S4_Kept_BNS_NSBH"] += 1
            continue # Added due to S4, skip S2/S3 checks for these specific events

        # S2: Cut at distance <= MAX_DISTANCE_MPC_S2 (for non-BNS/NSBH)
        if pd.isna(median_dL_mpc):
            logger.warning(f"S2 Filter: Event {event_name} - Median dL not available. Cannot pass S2. Discarding.")
            continue
        if median_dL_mpc > MAX_DISTANCE_MPC_S2:
            logger.info(f"S2 Filter: Event {event_name} failed distance cut (dL={median_dL_mpc:.2f} Mpc > {MAX_DISTANCE_MPC_S2} Mpc). Discarding.")
            continue
        logger.info(f"S2 Filter: Event {event_name} passed distance cut (dL={median_dL_mpc:.2f} Mpc).")
        selection_stats["3_Passed_S2_Distance_Cut (Non-BNS/NSBH)"] += 1

        # S3: Skip very poorly localised events (A_90 > MAX_SKY_AREA_S3) (for non-BNS/NSBH, after S2)
        if pd.isna(sky_area_90):
            logger.warning(f"S3 Filter: Event {event_name} - Sky area A90 not available. Cannot pass S3. Discarding.")
            continue
        if sky_area_90 > MAX_SKY_AREA_S3:
            logger.info(f"S3 Filter: Event {event_name} failed sky area cut (A90={sky_area_90:.2f} deg^2 > {MAX_SKY_AREA_S3} deg^2). Discarding.")
            continue
        logger.info(f"S3 Filter: Event {event_name} passed sky area cut (A90={sky_area_90:.2f} deg^2).")
        selection_stats["4_Passed_S3_SkyArea_Cut (Non-BNS/NSBH, Post-S2)"] += 1
        
        # If event passed S2 and S3 (and was not S5 bright siren or S4 BNS/NSBH)
        selected_candidates_for_analysis.append(event_name)

    CANDIDATE_LIST_TO_PROCESS = sorted(list(set(selected_candidates_for_analysis))) # Ensure unique and sorted
    selection_stats["5_Final_Selected_Dark_Sirens_for_Analysis"] = len(CANDIDATE_LIST_TO_PROCESS)

    logger.info("\n--- Candidate Selection Criteria Summary ---")
    for key, value in sorted(selection_stats.items()): # Sort keys for ordered output
        logger.info(f"{key}: {value}")
    logger.info(f"Identified Bright Sirens (S5): {bright_sirens_identified if bright_sirens_identified else 'None'}")
    logger.info(f"Final list of {len(CANDIDATE_LIST_TO_PROCESS)} dark sirens for detailed analysis: {CANDIDATE_LIST_TO_PROCESS if CANDIDATE_LIST_TO_PROCESS else 'None'}")

    if not CANDIDATE_LIST_TO_PROCESS:
        logger.warning("No candidates selected for analysis after applying all filters. Exiting.")
        return

    # --- Main Analysis Section (using CANDIDATE_LIST_TO_PROCESS) ---

    # 1. Load and clean galaxy catalog (once) - This can be done after candidate selection if preferred,
    # but current script structure loads it early. Keep as is for now.
    logger.info(f"\n--- Loading and Cleaning Galaxy Catalog ({VIZ_CATALOG_TYPE}) for Selected Candidates ---")
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

    for event_name in CANDIDATE_LIST_TO_PROCESS: # Iterate over the *filtered* list
        logger.info(f"\n--- Processing Selected Event: {event_name} ---")
        
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
            # Note: PE data might have been fetched already in get_event_filter_criteria for dL.
            # fetch_candidate_data handles caching, so re-fetch should be fast if already cached.
            success, gw_data_obj = fetch_candidate_data(event_name, effective_cache_dir)
            if not success:
                logger.error(f"Failed to fetch GW data for {event_name}: {gw_data_obj}")
                all_event_data.append(event_metrics) # Append with default/error values
                continue
            
            dL_gw_samples, ra_gw_samples, dec_gw_samples = extract_gw_event_parameters(gw_data_obj, event_name)

            if ra_gw_samples is None or dec_gw_samples is None:
                logger.error(f"Failed to get essential RA/Dec samples for {event_name}.")
                all_event_data.append(event_metrics)
                continue
            event_metrics["Num GW Samples (RA/Dec)"] = len(ra_gw_samples)

            # 3. Estimate event-specific host_z_max
            current_host_z_max_dynamic = VIZ_HOST_Z_MAX_FALLBACK # Fallback
            if dL_gw_samples is not None and len(dL_gw_samples) > 0:
                # Ensure dL_gw_samples are valid numbers for estimation
                valid_dl_samples = dL_gw_samples[~np.isnan(dL_gw_samples) & ~np.isinf(dL_gw_samples)]
                if len(valid_dl_samples) > 0:
                    current_host_z_max_dynamic = estimate_event_specific_z_max(
                        valid_dl_samples, # Use cleaned samples
                        percentile_dL=95.0, z_margin_factor=1.2,
                        min_z_max_val=0.01, max_z_max_val=0.3 
                    )
                    logger.info(f"Dynamically estimated host_z_max for {event_name}: {current_host_z_max_dynamic:.4f}")
                else:
                    logger.warning(f"No valid dL_gw_samples for {event_name} after cleaning. Using fallback host_z_max: {current_host_z_max_dynamic:.4f}")
            else:
                logger.warning(f"dL_gw_samples not available or empty for {event_name}. Using fallback host_z_max: {current_host_z_max_dynamic:.4f}")
            event_metrics["Est. Host z_max"] = round(current_host_z_max_dynamic, 4)

            # 4. Generate sky map and credible region
            _, sky_mask_for_selection, _ = generate_sky_map_and_credible_region(
                ra_gw_samples, dec_gw_samples,
                nside=VIZ_NSIDE_SKYMAP,
                cdf_threshold=VIZ_PROB_THRESHOLD_CDF
            )

            if sky_mask_for_selection is None or not sky_mask_for_selection.any():
                logger.warning(f"No valid sky mask generated for {event_name}. Galaxy selection in C.R. will be skipped.")
                # event_metrics will retain 0 for spatial/z-cut/final counts
            else:
                # 5. Select galaxies in sky region
                spatially_selected_galaxies_df = select_galaxies_in_sky_region(
                    glade_cleaned_df, sky_mask_for_selection, nside=VIZ_NSIDE_SKYMAP
                )
                num_spatially_selected = len(spatially_selected_galaxies_df)
                event_metrics["Num Spatially Selected (in C.R.)"] = num_spatially_selected
                logger.info(f"Found {num_spatially_selected} galaxies in the credible region for {event_name}.")

                if not spatially_selected_galaxies_df.empty:
                    # 6. Filter by redshift
                    galaxies_after_z_cut_df = filter_galaxies_by_redshift(
                        spatially_selected_galaxies_df, current_host_z_max_dynamic
                    )
                    num_after_z_cut = len(galaxies_after_z_cut_df)
                    event_metrics["Num After z-cut (in C.R. & z < z_max)"] = num_after_z_cut
                    logger.info(f"Found {num_after_z_cut} galaxies after redshift cut (z < {current_host_z_max_dynamic:.4f}) for {event_name}.")

                    if not galaxies_after_z_cut_df.empty:
                        # 7. Apply specific galaxy corrections
                        final_candidate_hosts_df = apply_specific_galaxy_corrections(
                            galaxies_after_z_cut_df, event_name, VIZ_GALAXY_CORRECTIONS
                        )
                        num_final_candidates = len(final_candidate_hosts_df)
                        event_metrics["Num Final Candidates (after corrections)"] = num_final_candidates
                        logger.info(f"Found {num_final_candidates} final candidate hosts after corrections for {event_name}.")
                    else:
                        logger.info(f"No galaxies remained after redshift cut for final corrections for {event_name}.")
                else:
                    logger.info(f"No galaxies spatially selected for {event_name} to apply redshift cut or corrections.")
        
        except Exception as e: # Catch-all for unexpected errors during a single event's processing
            logger.error(f"❌ An unexpected error occurred while processing {event_name}: {e}", exc_info=True)
            # event_metrics will have its defaults or partially filled values
        
        all_event_data.append(event_metrics)

    # 8. Display results in a table
    logger.info("\n\n--- Selected Candidate Host Galaxy Analysis Summary ---")
    if not all_event_data: # Should not happen if CANDIDATE_LIST_TO_PROCESS was non-empty, but good check
        logger.info("No data processed from selected candidates to display.")
        return

    results_df = pd.DataFrame(all_event_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1200) # Increased width for better table display
    
    # Ensure consistent column order, same as before
    column_order = [
        "Event Name", "Num GW Samples (RA/Dec)", "Est. Host z_max",
        "Num Cleaned GLADE+ Galaxies", "Num Spatially Selected (in C.R.)",
        "Num After z-cut (in C.R. & z < z_max)", "Num Final Candidates (after corrections)"
    ]
    # Filter out columns not present in results_df, if any (e.g. if all_event_data was empty)
    results_df_cols = [col for col in column_order if col in results_df.columns]
    results_df = results_df[results_df_cols]


    if results_df.empty:
        logger.info("Results DataFrame is empty. No summary to display or save.")
    else:
        logger.info("\n" + results_df.to_string(index=False))

        # Optionally, save to CSV
        csv_filename = os.path.join(OUTPUT_DIR, "selected_candidate_host_analysis_summary.csv")
        try:
            results_df.to_csv(csv_filename, index=False)
            logger.info(f"\nSummary table for selected candidates saved to: {csv_filename}")
        except Exception as e:
            logger.error(f"Error saving summary table to CSV: {e}")

if __name__ == "__main__":
    analyze_candidates()