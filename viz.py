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
from astropy.cosmology import FlatLambdaCDM # For H0 likelihood
from scipy.stats import norm # For H0 likelihood
import emcee # For MCMC

# Import from our new module
from gw_data_fetcher import fetch_candidate_data, configure_astropy_cache, DEFAULT_CACHE_DIR_NAME
from event_data_extractor import extract_gw_event_parameters

# Galaxy Catalog Handling
from galaxy_catalog_handler import (
    download_and_load_galaxy_catalog,
    clean_galaxy_catalog,
    apply_specific_galaxy_corrections,
    DEFAULT_GALAXY_CORRECTIONS,
    GLADE_URL, GLADE_FILE, GLADE_USE_COLS, GLADE_COL_NAMES, GLADE_NA_VALUES, GLADE_RANGE_CHECKS
)

# Sky Analysis and Candidate Selection
from sky_analyzer import (
    generate_sky_map_and_credible_region,
    select_galaxies_in_sky_region,
    filter_galaxies_by_redshift,
    DEFAULT_NSIDE_SKYMAP as MODULE_DEFAULT_NSIDE_SKYMAP, # Avoid name clash if viz.py has its own NSIDE_SKYMAP
    DEFAULT_PROB_THRESHOLD_CDF as MODULE_DEFAULT_PROB_THRESHOLD_CDF
)

# H0 MCMC Analysis
from h0_mcmc_analyzer import (
    get_log_likelihood_h0,
    run_mcmc_h0,
    process_mcmc_samples,
    DEFAULT_SIGMA_V_PEC, DEFAULT_C_LIGHT, DEFAULT_OMEGA_M,
    DEFAULT_MCMC_N_WALKERS, DEFAULT_MCMC_N_DIM, DEFAULT_MCMC_INITIAL_H0_MEAN,
    DEFAULT_MCMC_INITIAL_H0_STD, DEFAULT_MCMC_N_STEPS, DEFAULT_MCMC_BURNIN,
    DEFAULT_MCMC_THIN_BY, DEFAULT_H0_PRIOR_MIN, DEFAULT_H0_PRIOR_MAX
)

# -------------------------------------------------------------------
# Configuration specific to this visualization script
# -------------------------------------------------------------------
DEFAULT_EVENT_NAME_VIZ = "GW170817"

# HEALPix Sky Map parameters for this script
VIZ_NSIDE_SKYMAP = MODULE_DEFAULT_NSIDE_SKYMAP # Use default from sky_analyzer, can be overridden

# Analysis specific for host selection for this script
VIZ_PROB_THRESHOLD_CDF = MODULE_DEFAULT_PROB_THRESHOLD_CDF # Use default from sky_analyzer
VIZ_HOST_Z_MAX = 0.15 # Final redshift cut for candidate hosts in visualizations

# Galaxy corrections for this script
VIZ_GALAXY_CORRECTIONS = DEFAULT_GALAXY_CORRECTIONS

# MCMC parameters for this script (if running MCMC within viz.py)
# These can be set to the defaults from h0_mcmc_analyzer or customized here
VIZ_MCMC_N_STEPS = DEFAULT_MCMC_N_STEPS
VIZ_MCMC_BURNIN = DEFAULT_MCMC_BURNIN
VIZ_MCMC_THIN_BY = DEFAULT_MCMC_THIN_BY
VIZ_MCMC_N_WALKERS = DEFAULT_MCMC_N_WALKERS

# -------------------------------------------------------------------
# Plotting Functions (specific to or enhanced for viz.py)
# -------------------------------------------------------------------

def plot_redshift_distribution(
    galaxies_df,
    event_name,
    plot_suffix_label, # e.g., "Spatially Selected" or "Final Candidates"
    host_z_max_cutoff=None, # Made optional, as it might not always be relevant for all galaxy sets
    output_filename=None
):
    """
    Plots a histogram of redshift 'z' for the given DataFrame of galaxies.
    Includes an optional vertical line for a redshift cutoff.
    """
    if galaxies_df.empty or 'z' not in galaxies_df.columns:
        print(f"⚠️ Galaxies DataFrame is empty or 'z' column is missing. Cannot plot redshift distribution for '{plot_suffix_label}'.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(galaxies_df['z'], bins=50, density=False, histtype="stepfilled", alpha=0.7, label=f"Galaxies ({len(galaxies_df)})")
    
    if host_z_max_cutoff is not None:
        plt.axvline(host_z_max_cutoff, color='r', linestyle='--', 
                    label=f'$\mathit{{z}}_{{\mathrm{{max}}}}$ Cutoff = {host_z_max_cutoff:.3f}')

    plt.xlabel("Redshift (z)")
    plt.ylabel("Number of Galaxies")
    plt.title(f"Redshift Distribution of {plot_suffix_label} Galaxies - {event_name}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if output_filename is None:
        safe_suffix = "".join(c if c.isalnum() else "_" for c in plot_suffix_label).lower()
        output_filename = f"redshift_dist_{event_name}_{safe_suffix}.pdf"
    
    try:
        plt.savefig(output_filename)
        print(f"Redshift distribution plot saved to {output_filename}")
    except Exception as e:
        print(f"❌ Error saving redshift distribution plot '{output_filename}': {e}")
    plt.show(block=False)
    plt.pause(1) # Give time for plot to render if not blocking

def plot_basic_sky_probability_map(prob_map_gw, nside, event_name, output_filename=None):
    """
    Generates and plots a HEALPix sky map from a pre-computed probability map.

    Args:
        prob_map_gw (np.array): Pre-computed HEALPix probability map (normalized).
        nside (int): The Nside parameter for the HEALPix map.
        event_name (str): Name of the GW event, used for title and filename.
        output_filename (str, optional): Path to save the plot.
    """
    if prob_map_gw is None or prob_map_gw.size != hp.nside2npix(nside):
        print("⚠️ Probability map is None or Nside mismatch. Cannot generate basic sky map.")
        return
    if prob_map_gw.sum() == 0:
        print("⚠️ Probability map sum is zero. Plotting empty map.")

    print(f"\nGenerating basic sky probability map for {event_name} (Nside={nside})...")
    plt.figure(figsize=(10, 7))
    cmap = plt.cm.viridis 
    cmap.set_under("w")

    hp.mollview(
        prob_map_gw,
        title=f"GW Sky Localization Probability Map - {event_name}",
        unit="Probability Density",
        norm="hist", 
        cmap=cmap,
        min=1e-9 # Set a very small min to avoid issues with all-zero maps if cmap.set_under doesn't catch it
    )
    hp.graticule()

    if output_filename is None:
        output_filename = f"skymap_basic_{event_name}_nside{nside}.pdf"

    try:
        plt.savefig(output_filename)
        print(f"Sky map saved to {output_filename}")
    except Exception as e:
        print(f"❌ Error saving sky map: {e}")
    plt.show(block=False)
    plt.pause(1)

def plot_skymap_with_galaxies(
    prob_map_gw,       
    sky_mask_boolean,  
    all_galaxies_df,   
    selected_hosts_df, 
    nside,             
    event_name,        
    cred_level_percent,
    host_z_max,        
    output_filename=None
):
    """
    Plots the GW probability skymap, highlighting the credible region,
    and overlays cataloged galaxies, distinguishing selected candidates.
    """
    plt.figure(figsize=(12, 9))

    prob_map_display = np.copy(prob_map_gw)
    if sky_mask_boolean is not None and sky_mask_boolean.size == prob_map_gw.size:
        prob_map_display[~sky_mask_boolean] = hp.UNSEEN 
    else:
        print("⚠️ sky_mask_boolean is invalid or None. Displaying full probability map.")
        # prob_map_display remains a copy of prob_map_gw

    title = (
        f"GW Skymap ({event_name}) with Galaxies\n"
        f"{cred_level_percent:.0f}% Credible Region & z < {host_z_max:.2f} Selected Hosts"
    )

    # Use a small minimum for mollview if map can be all UNSEEN or zero
    min_val_for_plot = 1e-9 if np.all(prob_map_display == hp.UNSEEN) or np.sum(prob_map_display[prob_map_display != hp.UNSEEN]) == 0 else 0

    hp.mollview(
        prob_map_display,
        title=title,
        unit="Probability Density (in C.R.)",
        norm="hist",
        cmap=plt.cm.Blues,
        min=min_val_for_plot, 
        cbar=True,
        sub=None,
        nest=False
    )
    hp.graticule()

    galaxies_to_plot_far = pd.DataFrame()
    galaxies_to_plot_other_in_z_cut = pd.DataFrame()

    if not all_galaxies_df.empty and all(col in all_galaxies_df.columns for col in ['ra', 'dec', 'z']):
        galaxies_to_plot_far = all_galaxies_df[all_galaxies_df['z'] >= host_z_max]
        if not galaxies_to_plot_far.empty:
            hp.projplot(
                galaxies_to_plot_far['ra'].values, galaxies_to_plot_far['dec'].values,
                '.', lonlat=True, color='gray', alpha=0.5, markersize=2
            )

        galaxies_within_z_cut = all_galaxies_df[all_galaxies_df['z'] < host_z_max]
        if not galaxies_within_z_cut.empty:
            if not selected_hosts_df.empty and 'PGC' in galaxies_within_z_cut.columns and 'PGC' in selected_hosts_df.columns:
                # Ensure PGC columns are of compatible types for isin (e.g., string or float)
                pgc_selected_ids = selected_hosts_df['PGC'].dropna().astype(str).unique()
                galaxies_within_z_cut_pgc_str = galaxies_within_z_cut['PGC'].dropna().astype(str)
                is_not_selected = ~galaxies_within_z_cut_pgc_str.isin(pgc_selected_ids)
                galaxies_to_plot_other_in_z_cut = galaxies_within_z_cut[is_not_selected]
            else:
                galaxies_to_plot_other_in_z_cut = galaxies_within_z_cut
            
            if not galaxies_to_plot_other_in_z_cut.empty:
                hp.projplot(
                    galaxies_to_plot_other_in_z_cut['ra'].values, galaxies_to_plot_other_in_z_cut['dec'].values,
                    '.', lonlat=True, color='darkorange', alpha=0.7, markersize=3
                )

    if not selected_hosts_df.empty and all(col in selected_hosts_df.columns for col in ['ra', 'dec']):
        hp.projplot(
            selected_hosts_df['ra'].values, selected_hosts_df['dec'].values,
            '*', lonlat=True, color='red', markersize=10, markeredgecolor='black'
        )

    legend_elements = []
    if not galaxies_to_plot_far.empty: legend_elements.append(Line2D([0], [0], marker='.', color='w', label=f'Galaxies (z ≥ {host_z_max:.2f})', markerfacecolor='gray', alpha=0.5, markersize=5))
    if not galaxies_to_plot_other_in_z_cut.empty: legend_elements.append(Line2D([0], [0], marker='.', color='w', label=f'Other Galaxies (z < {host_z_max:.2f})', markerfacecolor='darkorange', alpha=0.7, markersize=6))
    if not selected_hosts_df.empty: legend_elements.append(Line2D([0], [0], marker='*', color='w', label=f'Selected Hosts (in C.R. & z < {host_z_max:.2f})', markerfacecolor='red', markeredgecolor='black', markersize=9))

    if legend_elements: plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize='medium', frameon=True)

    if output_filename is None: output_filename = f"skymap_galaxies_{event_name}_cr{int(cred_level_percent)}.pdf"
    
    try:
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Sky map with galaxies saved to {output_filename}")
    except Exception as e: print(f"❌ Error saving sky map with galaxies: {e}")
    plt.show(block=False)
    plt.pause(1)

def plot_mcmc_trace(sampler, event_name, burnin_steps, n_walkers_total, mcmc_n_dim_expected=1):
    """Plots the MCMC trace (walker paths) for H0."""
    if sampler is None: print(f"⚠️ Sampler object is None for {event_name}. Cannot plot MCMC trace."); return

    print(f"\nGenerating MCMC trace plot for {event_name}...")
    try: full_chain = sampler.get_chain() # Shape: (n_steps, n_walkers, n_dim)
    except Exception as e: print(f"❌ Error getting chain from sampler for {event_name}: {e}"); return

    if full_chain.ndim < 3 or full_chain.shape[2] != mcmc_n_dim_expected: print(f"❌ Chain dimension mismatch. Expected {mcmc_n_dim_expected}, got {full_chain.shape}"); return
    # if full_chain.shape[1] != n_walkers_total: print(f"⚠️ Walker number mismatch. Expected {n_walkers_total}, chain has {full_chain.shape[1]}. Plotting available walkers.")

    plt.figure(figsize=(12, 6))
    num_walkers_to_plot = min(full_chain.shape[1], 16) # Plot a subset if too many
    for i in range(num_walkers_to_plot):
        plt.plot(full_chain[:, i, 0], alpha=0.7, lw=0.5, label=f'Walker {i}' if num_walkers_to_plot <= 16 and full_chain.shape[1] >1 else None)
    if full_chain.shape[1] == 1 and num_walkers_to_plot ==1 : plt.plot(full_chain[:,0,0], alpha=0.7, lw=0.5, label='Walker 0') # Single walker case

    if burnin_steps > 0 and burnin_steps < full_chain.shape[0]: plt.axvline(burnin_steps, color='k', linestyle=':', linewidth=2, label=f'Burn-in Cutoff ({burnin_steps} steps)')
    
    plt.xlabel("MCMC Step Number"); plt.ylabel("$H_0$ Value (km s⁻¹ Mpc⁻¹)"); plt.title(f"Trace Plot for MCMC Walkers ($H_0$) - {event_name}")
    if num_walkers_to_plot <= 16 and num_walkers_to_plot > 0 and full_chain.shape[1] > 1: plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    elif full_chain.shape[1] == 1: plt.legend()
    plt.grid(True, alpha=0.3);
    plt.tight_layout(rect=[0, 0, 0.85 if num_walkers_to_plot <=16 and full_chain.shape[1] > 1 else 1, 1])
    
    output_filename = f"mcmc_trace_{event_name}.pdf"
    try: plt.savefig(output_filename); print(f"MCMC trace plot saved to {output_filename}")
    except Exception as e: print(f"❌ Error saving MCMC trace plot: {e}")
    plt.show(block=False); plt.pause(1)

def save_and_plot_h0_posterior_viz(h0_samples, event_name, num_candidate_hosts=None):
    """Saves H0 samples and plots the posterior distribution (viz.py version)."""
    if h0_samples is None or len(h0_samples) == 0: print(f"⚠️ No H0 samples provided for {event_name}. Cannot save or plot posterior."); return

    output_samples_file = f"H0_samples_{event_name}_viz.npy"
    output_plot_file = f"H0_posterior_{event_name}_viz.pdf"

    np.save(output_samples_file, h0_samples)
    print(f"MCMC H0 samples saved to {output_samples_file}")

    q16, q50, q84 = np.percentile(h0_samples, [16, 50, 84]); err_minus = q50 - q16; err_plus = q84 - q50
    print(f"\n{event_name} H0 (from viz) = {q50:.1f} +{err_plus:.1f} / -{err_minus:.1f} km s⁻¹ Mpc⁻¹ (68% C.I.)")

    plt.figure(figsize=(8, 6)); plt.hist(h0_samples, bins=50, density=True, histtype="stepfilled", alpha=0.6, label="Posterior Samples")
    plt.axvline(q50, color='k', ls='--', label=f'Median: {q50:.1f} km s⁻¹ Mpc⁻¹'); plt.axvline(q16, color='k', ls=':', alpha=0.7); plt.axvline(q84, color='k', ls=':', alpha=0.7)
    plt.xlabel(r"$H_0\;[\mathrm{km\ s^{-1}\ Mpc^{-1}}]$ "); plt.ylabel("Posterior Density")
    title_str = f"$H_0$ Posterior - {event_name}"
    if num_candidate_hosts is not None: title_str += f"\n({num_candidate_hosts:,} Candidate Galaxies)"
    plt.title(title_str); plt.legend(); plt.grid(True, linestyle=':', alpha=0.5); plt.tight_layout()
    plt.savefig(output_plot_file); print(f"Saved H0 posterior plot: {output_plot_file}")
    plt.show(block=False); plt.pause(1)

# -------------------------------------------------------------------
# Main script execution for viz.py
# -------------------------------------------------------------------
def main():
    """Main function for visualization: fetches samples, processes catalog, plots, and optionally runs H0 MCMC."""
    
    effective_cache_dir = configure_astropy_cache(DEFAULT_CACHE_DIR_NAME)
    if not effective_cache_dir: print("❌ CRITICAL: Failed to configure cache. Exiting."); sys.exit(1)

    current_event_name = DEFAULT_EVENT_NAME_VIZ
    if len(sys.argv) > 1: current_event_name = sys.argv[1]
    print(f"--- Starting Full Visualization Analysis for Event: {current_event_name} ---")
    
    # 1. Fetch GW data and extract parameters
    print(f"Fetching GW data for {current_event_name}...")
    success, gw_data_obj = fetch_candidate_data(current_event_name, effective_cache_dir)
    if not success: print(f"❌ Failed to fetch GW data: {gw_data_obj}. Some visualizations might be skipped."); gw_data_obj = None
    
    dL_gw_samples, ra_gw_samples, dec_gw_samples = (None, None, None)
    if gw_data_obj:
        dL_gw_samples, ra_gw_samples, dec_gw_samples = extract_gw_event_parameters(gw_data_obj, current_event_name)

    if ra_gw_samples is None or dec_gw_samples is None:
        print(f"❌ Failed to get essential RA/Dec samples for {current_event_name}. Sky maps will be skipped. MCMC may also be affected.")
    else:
        print(f"Successfully extracted {len(ra_gw_samples)} RA/Dec GW samples.")
        if dL_gw_samples is not None: print(f"  and {len(dL_gw_samples)} dL samples.")

    # 2. Generate basic sky probability map (if RA/Dec available)
    prob_map_gw_for_plots, sky_mask_for_plots = (None, None)
    if ra_gw_samples is not None and dec_gw_samples is not None:
        prob_map_gw_for_plots, sky_mask_for_plots, _ = generate_sky_map_and_credible_region(
            ra_gw_samples, dec_gw_samples, 
            nside=VIZ_NSIDE_SKYMAP, 
            cdf_threshold=VIZ_PROB_THRESHOLD_CDF
        )
        if prob_map_gw_for_plots is not None and prob_map_gw_for_plots.sum() > 0:
            plot_basic_sky_probability_map(
                prob_map_gw_for_plots,
                VIZ_NSIDE_SKYMAP,
                current_event_name
            )
        else:
            print("Skipping basic sky probability map due to issues generating it.")
    else:
        print("Skipping basic sky probability map due to missing RA/Dec samples.")

    print("\n--- Preparing data for skymap with galaxies and H0 MCMC ---")
    # 3. Load and clean galaxy catalog
    glade_raw_df = download_and_load_galaxy_catalog(GLADE_URL, GLADE_FILE, GLADE_USE_COLS, GLADE_COL_NAMES, GLADE_NA_VALUES)
    if glade_raw_df.empty: print("❌ GLADE catalog empty after loading. Galaxy-dependent plots and MCMC will be skipped."); return
        
    glade_cleaned_df = clean_galaxy_catalog(glade_raw_df, GLADE_COL_NAMES, ['ra', 'dec', 'z'], GLADE_RANGE_CHECKS)
    if glade_cleaned_df.empty: print("❌ GLADE catalog empty after cleaning. Galaxy-dependent plots and MCMC will be skipped."); return

    # 4. Select candidate hosts for visualization and potential MCMC
    final_candidate_hosts_df = pd.DataFrame() 
    if sky_mask_for_plots is not None and sky_mask_for_plots.any():
        spatially_selected_galaxies_df = select_galaxies_in_sky_region(
            glade_cleaned_df, sky_mask_for_plots, nside=VIZ_NSIDE_SKYMAP
        )
        if not spatially_selected_galaxies_df.empty:
            plot_redshift_distribution(
                spatially_selected_galaxies_df, current_event_name, 
                "Spatially Selected (in C.R.)", host_z_max_cutoff=VIZ_HOST_Z_MAX
            )
            intermediate_hosts_df = filter_galaxies_by_redshift(spatially_selected_galaxies_df, VIZ_HOST_Z_MAX)
            final_candidate_hosts_df = apply_specific_galaxy_corrections(
                intermediate_hosts_df, current_event_name, VIZ_GALAXY_CORRECTIONS
            )
    else:
        print("Sky mask for CR not available or empty. Attempting selection from full catalog with z-cut for MCMC if needed.")
        # Fallback: if no sky_mask, consider all cleaned galaxies up to z_max for MCMC (less optimal)
        # This path is more for ensuring MCMC can run if dL samples exist but skymap failed badly.
        intermediate_hosts_df = filter_galaxies_by_redshift(glade_cleaned_df, VIZ_HOST_Z_MAX)
        final_candidate_hosts_df = apply_specific_galaxy_corrections(
            intermediate_hosts_df, current_event_name, VIZ_GALAXY_CORRECTIONS
        )

    if final_candidate_hosts_df.empty:
        print(f"⚠️ No candidate host galaxies identified for {current_event_name} after selection process. Overlay plot might be sparse. MCMC for H0 might be skipped or use broader galaxy set.")
    else:
        print(f"Final number of candidate hosts for {current_event_name} for overlay & H0: {len(final_candidate_hosts_df)}")
        plot_redshift_distribution(
            final_candidate_hosts_df, current_event_name, 
            "Final Candidate Hosts (for MCMC/Overlay)", host_z_max_cutoff=VIZ_HOST_Z_MAX
        )

    # Plot skymap with galaxies (if prob_map_gw and sky_mask_for_plots were generated)
    if prob_map_gw_for_plots is not None and sky_mask_for_plots is not None:
        plot_skymap_with_galaxies(
            prob_map_gw=prob_map_gw_for_plots,
            sky_mask_boolean=sky_mask_for_plots,
            all_galaxies_df=glade_cleaned_df,
            selected_hosts_df=final_candidate_hosts_df, # Use the most refined set for highlighting
            nside=VIZ_NSIDE_SKYMAP,
            event_name=current_event_name,
            cred_level_percent=VIZ_PROB_THRESHOLD_CDF * 100,
            host_z_max=VIZ_HOST_Z_MAX
        )
    else:
        print("Skipping skymap with galaxies due to missing probability map or sky mask.")

    # 5. Perform MCMC for H0 if dL samples and some candidate hosts are available
    can_run_mcmc = dL_gw_samples is not None and len(dL_gw_samples) > 0
    if can_run_mcmc and not final_candidate_hosts_df.empty:
        print(f"\n--- Proceeding with MCMC H0 estimation for {current_event_name} (within viz.py) ---")
        try:
            log_likelihood_h0_func = get_log_likelihood_h0(
                dL_gw_samples, final_candidate_hosts_df['z'].values,
                DEFAULT_SIGMA_V_PEC, DEFAULT_C_LIGHT, DEFAULT_OMEGA_M,
                DEFAULT_H0_PRIOR_MIN, DEFAULT_H0_PRIOR_MAX
            )
        except ValueError as ve:
            print(f"❌ Error creating H0 likelihood for viz: {ve}. Skipping MCMC."); log_likelihood_h0_func = None

        if log_likelihood_h0_func:
            mcmc_sampler = run_mcmc_h0(
                log_likelihood_h0_func, current_event_name,
                n_walkers=VIZ_MCMC_N_WALKERS, n_steps=VIZ_MCMC_N_STEPS
            )
            if mcmc_sampler:
                plot_mcmc_trace(mcmc_sampler, current_event_name, VIZ_MCMC_BURNIN, VIZ_MCMC_N_WALKERS, DEFAULT_MCMC_N_DIM)
                flat_h0_samples = process_mcmc_samples(mcmc_sampler, current_event_name, VIZ_MCMC_BURNIN, VIZ_MCMC_THIN_BY, DEFAULT_MCMC_N_DIM)
                if flat_h0_samples is not None and len(flat_h0_samples) > 0:
                    save_and_plot_h0_posterior_viz(flat_h0_samples, current_event_name, len(final_candidate_hosts_df))
                else: print(f"Skipping H0 posterior plot for {current_event_name} due to no valid MCMC samples after processing.")
            else: print(f"Skipping MCMC post-processing for {current_event_name} because MCMC run failed or returned no sampler.")
    elif not can_run_mcmc:
        print(f"\n--- Skipping MCMC H0 estimation for {current_event_name} due to missing dL samples. ---")
    elif final_candidate_hosts_df.empty:
        print(f"\n--- Skipping MCMC H0 estimation for {current_event_name} because no candidate host galaxies were identified. ---")

    print(f"\n--- Visualization Script Finished for {current_event_name} ---")
    print("Close all plot windows to exit script fully if plots are blocking.")
    plt.show() # Final show to ensure all non-blocking plots are displayed until closed

if __name__ == "__main__":
    main()
