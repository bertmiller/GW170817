#!/usr/bin/env python
"""
Reproduce Fishbach et al. (2019) statistical-standard-siren analysis
for GW170817 using the GLADE v2.4 galaxy catalogue.

Refactored for modularity and reusability.

Outputs:
  - 'H0_samples_<event_name>.npy'
  - 'H0_posterior_<event_name>.pdf'
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from gwsiren import CONFIG

# Import from our new module
from gw_data_fetcher import fetch_candidate_data, configure_astropy_cache, DEFAULT_CACHE_DIR_NAME
from event_data_extractor import extract_gw_event_parameters

# Galaxy Catalog Handling
from galaxy_catalog_handler import (
    download_and_load_galaxy_catalog,
    clean_galaxy_catalog,
    apply_specific_galaxy_corrections,
    DEFAULT_GALAXY_CORRECTIONS, # If new.py needs to pass this specifically
    GLADE_URL, GLADE_FILE, GLADE_USE_COLS, GLADE_COL_NAMES, GLADE_NA_VALUES, GLADE_RANGE_CHECKS
)

# Sky Analysis and Candidate Selection
from sky_analyzer import (
    generate_sky_map_and_credible_region,
    select_galaxies_in_sky_region,
    filter_galaxies_by_redshift,
    DEFAULT_NSIDE_SKYMAP,
    DEFAULT_PROB_THRESHOLD_CDF
)

# H0 MCMC Analysis
from h0_mcmc_analyzer import (
    get_log_likelihood_h0,
    run_mcmc_h0,
    process_mcmc_samples,
    DEFAULT_SIGMA_V_PEC, DEFAULT_C_LIGHT, DEFAULT_OMEGA_M, # Cosmological params
    DEFAULT_MCMC_N_WALKERS, DEFAULT_MCMC_N_DIM, DEFAULT_MCMC_INITIAL_H0_MEAN, 
    DEFAULT_MCMC_INITIAL_H0_STD, DEFAULT_MCMC_N_STEPS, DEFAULT_MCMC_BURNIN, 
    DEFAULT_MCMC_THIN_BY, DEFAULT_H0_PRIOR_MIN, DEFAULT_H0_PRIOR_MAX # MCMC params
)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
# GW Event specific
DEFAULT_EVENT_NAME = "GW170817"
# DEFAULT_SAMPLES_PATH_PREFIX removed as gw_data_fetcher handles table selection

# GLADE Catalog specific
GLADE_URL = CONFIG.catalog["glade24_url"]
GLADE_FILE = "GLADE_2.4.txt"
GLADE_USE_COLS = [0, 6, 7, 15]  # PGC, RA, Dec, z
GLADE_COL_NAMES = ['PGC', 'ra', 'dec', 'z']
GLADE_NA_VALUES = ['-99.0', '-999.0', '-9999.0', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'N/A', 'n/a', 'None', '...', 'no_value']
GLADE_RANGE_CHECKS = {
    'dec_min': -90, 'dec_max': 90,
    'ra_min': 0, 'ra_max': 360,
    'z_min': 0, 'z_max': 2.0 # Initial broad z range for catalog
}

# Analysis specific
NSIDE_SKYMAP = CONFIG.skymap["default_nside"]
PROB_THRESHOLD_CDF = CONFIG.skymap["credible_level"]
HOST_Z_MAX = 0.15 # Final redshift cut for candidate hosts

# Cosmological parameters for likelihood
SIGMA_V_PEC = CONFIG.cosmology["sigma_v_pec"]  # km/s, peculiar velocity uncertainty
C_LIGHT = CONFIG.cosmology["c_light"]
OMEGA_M = CONFIG.cosmology["omega_m"]

# MCMC parameters
MCMC_N_DIM = 1
MCMC_N_WALKERS = CONFIG.mcmc["walkers"]
MCMC_N_STEPS = CONFIG.mcmc["steps"]
MCMC_BURNIN = CONFIG.mcmc["burnin"]
MCMC_THIN_BY = CONFIG.mcmc["thin_by"]
MCMC_INITIAL_H0_MEAN = 70
MCMC_INITIAL_H0_STD = 10

# Specific galaxy corrections (can be extended or moved to a config file)
GALAXY_CORRECTIONS = {
    "GW170817": {
        "PGC_ID": 45657.0, # NGC 4993
        "LITERATURE_Z": 0.009783
    }
}

# -------------------------------------------------------------------
# Configuration specific to this script (can override module defaults)
# -------------------------------------------------------------------
DEFAULT_EVENT_NAME = "GW170817"

# Analysis specific parameters for this script (can override module defaults if needed by passing to functions)
# Example: current_nside_skymap = 256 # to override DEFAULT_NSIDE_SKYMAP from sky_analyzer
CURRENT_NSIDE_SKYMAP = DEFAULT_NSIDE_SKYMAP
CURRENT_PROB_THRESHOLD_CDF = DEFAULT_PROB_THRESHOLD_CDF
CURRENT_HOST_Z_MAX = 0.15 # Redshift cut specific to this analysis pipeline

# Galaxy corrections for this script (can use module default or specify a custom one)
CURRENT_GALAXY_CORRECTIONS = DEFAULT_GALAXY_CORRECTIONS

# -------------------------------------------------------------------
# 5.  Plot and save posterior
# -------------------------------------------------------------------
def save_and_plot_h0_posterior(h0_samples, event_name, num_candidate_hosts):
    """Saves H0 samples and plots the posterior distribution."""
    if h0_samples is None or len(h0_samples) == 0:
        print(f"⚠️ No H0 samples provided for {event_name}. Cannot save or plot posterior.")
        return

    output_samples_file = f"H0_samples_{event_name}.npy"
    output_plot_file = f"H0_posterior_{event_name}.pdf"

    np.save(output_samples_file, h0_samples)
    print(f"MCMC H0 samples saved to {output_samples_file}")

    q16, q50, q84 = np.percentile(h0_samples, [16, 50, 84])
    err_minus = q50 - q16
    err_plus = q84 - q50
    print(f"\n{event_name} H0 = {q50:.1f} +{err_plus:.1f} / -{err_minus:.1f} km s⁻¹ Mpc⁻¹ (68% C.I.)")

    plt.figure(figsize=(8, 6))
    plt.hist(h0_samples, bins=50, density=True, histtype="stepfilled", alpha=0.6, label="Posterior Samples")
    plt.axvline(q50, color='k', ls='--', label=f'Median: {q50:.1f} km s⁻¹ Mpc⁻¹')
    plt.axvline(q16, color='k', ls=':', alpha=0.7)
    plt.axvline(q84, color='k', ls=':', alpha=0.7)
    plt.xlabel(r"$H_0\;[\mathrm{km\ s^{-1}\ Mpc^{-1}}]$ ")
    plt.ylabel("Posterior Density")
    title_str = f"Statistical Standard Siren – {event_name} & GLADE v2.4"
    if num_candidate_hosts is not None:
        title_str += f"\n({num_candidate_hosts:,} Candidate Galaxies)"
    plt.title(title_str)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_plot_file)
    print(f"Saved posterior plot: {output_plot_file}")
    # plt.show() # Optional: uncomment to display plot directly

# -------------------------------------------------------------------
# Main script execution
# -------------------------------------------------------------------
def main(event_name=DEFAULT_EVENT_NAME):
    """Main function to run the H0 analysis for a given GW event."""
    
    effective_cache_dir = configure_astropy_cache(DEFAULT_CACHE_DIR_NAME)
    if not effective_cache_dir:
        print("❌ CRITICAL: Failed to configure cache. Exiting.")
        sys.exit(1)

    print(f"--- Starting H0 Analysis for Event: {event_name} ---")

    # 0. Fetch GW data and extract parameters
    print(f"Fetching GW data for {event_name}...")
    success, gw_data_obj = fetch_candidate_data(event_name, effective_cache_dir)
    if not success:
        print(f"❌ Failed to fetch GW data for {event_name}: {gw_data_obj}. Exiting.")
        sys.exit(1)
    
    dL_gw_samples, ra_gw_samples, dec_gw_samples = extract_gw_event_parameters(gw_data_obj, event_name)
    if dL_gw_samples is None or ra_gw_samples is None or dec_gw_samples is None:
        print(f"❌ Failed to extract essential GW parameters (dL, RA, Dec) for {event_name}. Exiting.")
        sys.exit(1)
    print(f"Successfully fetched and extracted {len(dL_gw_samples)} GW samples.")

    # 1. Load and clean galaxy catalog
    glade_raw_df = download_and_load_galaxy_catalog(
        url=GLADE_URL, 
        filename=GLADE_FILE, 
        use_cols=GLADE_USE_COLS, 
        col_names=GLADE_COL_NAMES, 
        na_vals=GLADE_NA_VALUES
    )
    if glade_raw_df.empty:
        print("❌ GLADE catalog is empty after loading. Exiting.")
        sys.exit(1)
        
    glade_cleaned_df = clean_galaxy_catalog(
        glade_raw_df,
        numeric_cols=GLADE_COL_NAMES,
        cols_to_dropna=['ra', 'dec', 'z'], 
        range_filters=GLADE_RANGE_CHECKS
    )
    if glade_cleaned_df.empty:
        print("❌ GLADE catalog is empty after cleaning. Exiting.")
        sys.exit(1)

    # 2. Generate sky map, credible region, and select candidate hosts
    # 2a. Generate GW probability map and credible region sky mask
    prob_map_gw, sky_mask_boolean, _ = generate_sky_map_and_credible_region(
        ra_gw_samples, dec_gw_samples, 
        nside=CURRENT_NSIDE_SKYMAP, 
        cdf_threshold=CURRENT_PROB_THRESHOLD_CDF
    )
    if not sky_mask_boolean.any(): # If mask is all False
        print(f"⚠️ No pixels in the {CURRENT_PROB_THRESHOLD_CDF*100:.0f}% credible region for {event_name}. MCMC might be problematic or yield no hosts.")
        # Depending on policy, one might exit or proceed hoping some galaxies are found anyway (unlikely)

    # 2b. Select galaxies within this sky_mask_boolean (spatially selected)
    spatially_selected_hosts_df = select_galaxies_in_sky_region(
        glade_cleaned_df, sky_mask_boolean, nside=CURRENT_NSIDE_SKYMAP
    )
    if spatially_selected_hosts_df.empty:
        print(f"⚠️ No galaxies found within the {CURRENT_PROB_THRESHOLD_CDF*100:.0f}% credible region for {event_name}. Cannot proceed with H0 estimation.")
        sys.exit(f"Exiting: No spatially selected candidate galaxies for {event_name}.")

    # 2c. Apply redshift cut to spatially selected galaxies
    candidate_hosts_intermediate_df = filter_galaxies_by_redshift(
        spatially_selected_hosts_df, z_max_filter=CURRENT_HOST_Z_MAX
    )
    if candidate_hosts_intermediate_df.empty:
        print(f"⚠️ No candidate host galaxies remain after redshift cut (z < {CURRENT_HOST_Z_MAX}) for {event_name}. Cannot proceed.")
        sys.exit(f"Exiting: No candidate galaxies after redshift cut for {event_name}.")
    
    # 2d. Apply specific galaxy corrections (e.g., NGC 4993 for GW170817)
    final_candidate_hosts_df = apply_specific_galaxy_corrections(
        candidate_hosts_intermediate_df, event_name, corrections_dict=CURRENT_GALAXY_CORRECTIONS
    )
    if final_candidate_hosts_df.empty:
        print(f"⚠️ No candidate host galaxies remain after specific corrections for {event_name}. Cannot proceed.")
        sys.exit(f"Exiting: No candidate galaxies after specific corrections for {event_name}.")

    print(f"Identified {len(final_candidate_hosts_df)} final candidate host galaxies for {event_name}.")

    # 3. Define Likelihood function
    try:
        log_likelihood_h0_func = get_log_likelihood_h0(
            dL_gw_samples,
            final_candidate_hosts_df['z'].values,
            sigma_v=DEFAULT_SIGMA_V_PEC, 
            c_val=DEFAULT_C_LIGHT, 
            omega_m_val=DEFAULT_OMEGA_M,
            h0_min=DEFAULT_H0_PRIOR_MIN,
            h0_max=DEFAULT_H0_PRIOR_MAX
        )
    except ValueError as ve:
        print(f"❌ Error creating log-likelihood function for {event_name}: {ve}")
        sys.exit(1)

    # 4. Run MCMC
    mcmc_sampler = run_mcmc_h0(
        log_likelihood_h0_func,
        event_name,
        n_walkers=DEFAULT_MCMC_N_WALKERS, 
        n_dim=DEFAULT_MCMC_N_DIM, 
        initial_h0_mean=DEFAULT_MCMC_INITIAL_H0_MEAN, 
        initial_h0_std=DEFAULT_MCMC_INITIAL_H0_STD, 
        n_steps=DEFAULT_MCMC_N_STEPS
    )
    if mcmc_sampler is None:
        print(f"❌ MCMC run failed for {event_name}. Exiting.")
        sys.exit(1)
    
    flat_h0_samples = process_mcmc_samples(
        mcmc_sampler, 
        event_name,
        burnin=DEFAULT_MCMC_BURNIN, 
        thin_by=DEFAULT_MCMC_THIN_BY,
        n_dim=DEFAULT_MCMC_N_DIM
        )
    
    if flat_h0_samples is None or len(flat_h0_samples) == 0:
        print(f"❌ MCMC processing yielded no valid samples for {event_name}. Exiting.")
        sys.exit(1)

    # 5. Save and Plot results
    save_and_plot_h0_posterior(flat_h0_samples, event_name, len(final_candidate_hosts_df))

    print(f"\n--- H0 Analysis Script Finished for {event_name} ---")


if __name__ == "__main__":
    current_event_name = DEFAULT_EVENT_NAME
    if len(sys.argv) > 1:
        current_event_name = sys.argv[1]
        print(f"Running analysis for event specified from command line: {current_event_name}")
    else:
        print(f"Running analysis for default event: {current_event_name}")
        
    main(event_name=current_event_name)
