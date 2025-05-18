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
from matplotlib.animation import FuncAnimation # Added for GIF animation
import urllib.request # For downloading GLADE
from astropy.coordinates import SkyCoord # For select_candidate_hosts
from astropy import units as u # For select_candidate_hosts
from astropy.cosmology import FlatLambdaCDM # For H0 likelihood
from scipy.stats import norm # For H0 likelihood
import emcee # For MCMC
from emcee.interruptible_pool import InterruptiblePool # For parallel MCMC
import logging # Added logging
import argparse # Added argparse
from gwsiren import CONFIG

# Get a logger for this module (configuration will be done in main)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "output" # Define the output directory

# Configure basic logging as early as possible, ideally before other module imports
# if those modules also use logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Default level, change to logging.DEBUG for MCMC details etc.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
)

# Import from our new module
from gw_data_fetcher import fetch_candidate_data, configure_astropy_cache, DEFAULT_CACHE_DIR_NAME
from event_data_extractor import extract_gw_event_parameters

# Galaxy Catalog Handling
from gwsiren.data.catalogs import (
    download_and_load_galaxy_catalog,
    clean_galaxy_catalog,
    apply_specific_galaxy_corrections,
    DEFAULT_GALAXY_CORRECTIONS,
    DEFAULT_RANGE_CHECKS
)

# Sky Analysis and Candidate Selection
from sky_analyzer import (
    generate_sky_map_and_credible_region,
    select_galaxies_in_sky_region,
    filter_galaxies_by_redshift,
    estimate_event_specific_z_max
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

# Import the new 3D plotting function
from plot_utils import plot_3d_localization_with_galaxies

# -------------------------------------------------------------------
# Configuration specific to this visualization script
# -------------------------------------------------------------------
DEFAULT_EVENT_NAME_VIZ = "GW170817"
VIZ_CATALOG_TYPE = 'glade+' # Specify catalog type: 'glade+' or 'glade24'

# HEALPix Sky Map parameters for this script
VIZ_NSIDE_SKYMAP = CONFIG.skymap["default_nside"]

# Analysis specific for host selection for this script
VIZ_PROB_THRESHOLD_CDF = CONFIG.skymap["credible_level"]
VIZ_HOST_Z_MAX_FALLBACK = 0.05 

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
        logger.warning(f"⚠️ Galaxies DataFrame is empty or 'z' column is missing. Cannot plot redshift distribution for '{plot_suffix_label}'.")
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
    
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        plt.savefig(full_output_path)
        logger.info(f"Redshift distribution plot saved to {full_output_path}")
    except Exception as e:
        logger.error(f"❌ Error saving redshift distribution plot '{full_output_path}': {e}")
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
        logger.warning("⚠️ Probability map is None or Nside mismatch. Cannot generate basic sky map.")
        return
    if prob_map_gw.sum() == 0:
        logger.warning("⚠️ Probability map sum is zero. Plotting empty map.")

    logger.info(f"\nGenerating basic sky probability map for {event_name} (Nside={nside})...")
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

    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(full_output_path)
        logger.info(f"Sky map saved to {full_output_path}")
    except Exception as e:
        logger.error(f"❌ Error saving sky map: {e}")
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
        logger.warning("⚠️ sky_mask_boolean is invalid or None. Displaying full probability map.")
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
                pgc_selected_ids_set = set(selected_hosts_df['PGC'].dropna().astype(str))
                
                # Use .apply() to ensure the boolean mask 'is_not_selected' has an index aligned with 'galaxies_within_z_cut'
                is_not_selected = galaxies_within_z_cut['PGC'].apply(lambda pgc: str(pgc) not in pgc_selected_ids_set)
                
                galaxies_to_plot_other_in_z_cut = galaxies_within_z_cut[is_not_selected]
            else:
                # If PGC info is missing for comparison or no selected hosts, all within z-cut are "other"
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
    
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(full_output_path, bbox_inches='tight')
        logger.info(f"Sky map with galaxies saved to {full_output_path}")
    except Exception as e: logger.error(f"❌ Error saving sky map with galaxies: {e}")
    plt.show(block=False)
    plt.pause(1)

def plot_mcmc_trace(sampler, event_name, burnin_steps, n_walkers_total, mcmc_n_dim_expected=1):
    """Plots the MCMC trace (walker paths) for H0."""
    if sampler is None: logger.warning(f"⚠️ Sampler object is None for {event_name}. Cannot plot MCMC trace."); return

    logger.info(f"\nGenerating MCMC trace plot for {event_name}...")
    try: full_chain = sampler.get_chain() # Shape: (n_steps, n_walkers, n_dim)
    except Exception as e: logger.error(f"❌ Error getting chain from sampler for {event_name}: {e}"); return

    if full_chain.ndim < 3 or full_chain.shape[2] != mcmc_n_dim_expected: logger.error(f"❌ Chain dimension mismatch. Expected {mcmc_n_dim_expected}, got {full_chain.shape}"); return
    # if full_chain.shape[1] != n_walkers_total: logger.warning(f"⚠️ Walker number mismatch. Expected {n_walkers_total}, chain has {full_chain.shape[1]}. Plotting available walkers.")

    plt.figure(figsize=(12, 6))
    num_walkers_to_plot = min(full_chain.shape[1], 5) # Plot a subset of up to 5 walkers
    for i in range(num_walkers_to_plot):
        # Label walkers if we are plotting 5 or fewer and there is more than one walker
        label = None
        if num_walkers_to_plot <= 5:
            if full_chain.shape[1] > 1:
                label = f'Walker {i}'
            elif full_chain.shape[1] == 1: # Single walker case
                label = 'Walker 0'
        plt.plot(full_chain[:, i, 0], alpha=0.7, lw=0.5, label=label)

    if burnin_steps > 0 and burnin_steps < full_chain.shape[0]: plt.axvline(burnin_steps, color='k', linestyle=':', linewidth=2, label=f'Burn-in Cutoff ({burnin_steps} steps)')
    
    plt.xlabel("MCMC Step Number"); plt.ylabel("$H_0$ Value (km s⁻¹ Mpc⁻¹)"); plt.title(f"Trace Plot for MCMC Walkers ($H_0$) - {event_name}")
    # Show legend if labels were assigned
    if num_walkers_to_plot > 0 and any(plt.gca().get_legend_handles_labels()[1]):
        if full_chain.shape[1] > 1 and num_walkers_to_plot <=5: # Legend outside for multiple walkers if few are plotted
             plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        elif full_chain.shape[1] == 1: # Standard legend for single walker
            plt.legend()
        # No legend if many walkers are plotted without individual labels (though current logic plots at most 5)
            
    plt.grid(True, alpha=0.3);
    # Adjust layout for external legend if present
    legend_is_external = full_chain.shape[1] > 1 and num_walkers_to_plot <= 5 and any(plt.gca().get_legend_handles_labels()[1])
    plt.tight_layout(rect=[0, 0, 0.85 if legend_is_external else 1, 1])
    
    output_filename = f"mcmc_trace_{event_name}.pdf"
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    try: plt.savefig(full_output_path); logger.info(f"MCMC trace plot saved to {full_output_path}")
    except Exception as e: logger.error(f"❌ Error saving MCMC trace plot: {e}")
    plt.show(block=False); plt.pause(1)

def save_and_plot_h0_posterior_viz(h0_samples, event_name, num_candidate_hosts=None):
    """Save MCMC samples and plot 2D posterior for ``H0`` and ``alpha``."""
    if h0_samples is None or len(h0_samples) == 0:
        logger.warning("⚠ No samples provided to plotting function.")
        return

    base_filename_samples = f"H0_samples_{event_name}_viz.npy"
    base_filename_plot = f"H0_alpha_posterior_{event_name}_viz.pdf"

    full_output_path_samples = os.path.join(OUTPUT_DIR, base_filename_samples)
    full_output_path_plot = os.path.join(OUTPUT_DIR, base_filename_plot)

    np.save(full_output_path_samples, h0_samples)
    logger.info(f"MCMC samples saved to {full_output_path_samples}")

    samples = np.asarray(h0_samples)
    h0_vals = samples[:, 0]
    alpha_vals = samples[:, 1]
    q16_h0, q50_h0, q84_h0 = np.percentile(h0_vals, [16, 50, 84])
    q16_a, q50_a, q84_a = np.percentile(alpha_vals, [16, 50, 84])
    logger.info(
        f"\n{event_name} H0 (from viz) = {q50_h0:.1f} +{q84_h0 - q50_h0:.1f} / -{q50_h0 - q16_h0:.1f} km s⁻¹ Mpc⁻¹ (68% C.I.)"
    )
    logger.info(
        f"{event_name} alpha (from viz) = {q50_a:.2f} +{q84_a - q50_a:.2f} / -{q50_a - q16_a:.2f} (68% C.I.)"
    )

    fig = corner.corner(
        samples,
        labels=[r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)", r"$\alpha$"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
    )
    title_str = f"Posterior - {event_name}"
    if num_candidate_hosts is not None:
        fig.suptitle(f"{title_str}\n({num_candidate_hosts:,} Candidate Galaxies)")
    else:
        fig.suptitle(title_str)

    fig.tight_layout()
    try:
        fig.savefig(full_output_path_plot)
        logger.info(f"Saved posterior corner plot: {full_output_path_plot}")
    except Exception as e:
        logger.error(f"❌ Error saving posterior plot: {e}")
    plt.show(block=False); plt.pause(1)

def create_walker_animation_gif(
    sampler,
    n_steps,
    burnin_steps,
    event_id,
    walker_idx=0,
    output_filename_template="{event_id}_walker_{walker_idx}_animation.gif",
    plot_interval=10,
    fps=15,
    dim_to_plot=0,
):
    """Create a GIF animation of a single MCMC walker's path."""
    try:
        full_chain = sampler.get_chain() # Shape: (n_steps, n_walkers, n_dim)
        if walker_idx >= full_chain.shape[1]:
            logger.error(f"❌ Error: Walker index {walker_idx} is out of bounds for {full_chain.shape[1]} walkers for event {event_id}.")
            return
        # Assuming H0 is the 0-th dimension, and chain has n_dim dimensions
        if full_chain.ndim < 3 or full_chain.shape[2] == 0 : # n_dim must be at least 1
             logger.error(f"❌ Error: Chain has unexpected dimensions {full_chain.shape} for event {event_id}.")
             return
        single_walker_h0_history = full_chain[:, walker_idx, dim_to_plot]
    except Exception as e:
        logger.error(f"❌ Could not get chain for animation for event {event_id}: {e}")
        return

    if len(single_walker_h0_history) == 0:
        logger.warning(f"No MCMC history found for walker {walker_idx} for event {event_id}.")
        return

    logger.info(f"Generating MCMC walker animation for event {event_id}, walker {walker_idx}...")

    min_h0 = np.min(single_walker_h0_history)
    max_h0 = np.max(single_walker_h0_history)
    padding_h0 = (max_h0 - min_h0) * 0.1
    y_min_limit = min_h0 - padding_h0
    y_max_limit = max_h0 + padding_h0

    total_steps_in_chain = len(single_walker_h0_history)
    effective_n_steps = total_steps_in_chain # Use actual chain length for x-limit

    # Ensure n_steps (from config) is used if it's the intended x-axis visual limit,
    # but animation frames should not exceed actual chain length.
    # The original code uses effective_n_steps (total_steps_in_chain) for xlim, which is good.
    # The `n_steps` parameter to this function is more of an expectation or for context.

    steps_to_plot_indices = np.arange(0, total_steps_in_chain, plot_interval)
    if not steps_to_plot_indices.size: # If plot_interval is too large for the chain length
        if total_steps_in_chain > 0: # Ensure at least one frame if there's data
            steps_to_plot_indices = np.array([total_steps_in_chain - 1])
        else: # No data, no frames
            logger.warning(f"⚠️ No frames to animate for walker {walker_idx} (event {event_id}) with plot_interval={plot_interval} and chain length {total_steps_in_chain}.")
            plt.close(fig)
            return


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, effective_n_steps)
    ax.set_ylim(y_min_limit, y_max_limit)
    ax.set_xlabel("MCMC Step Number")
    ylabel = "$H_0$ Value (km s⁻¹ Mpc⁻¹)" if dim_to_plot == 0 else "$\\alpha$"
    title_param = "$H_0$" if dim_to_plot == 0 else "$\\alpha$"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Path of MCMC Walker {walker_idx} for {title_param} ({event_id})")
    ax.grid(True, alpha=0.3)

    # Plot burn-in line only if burnin_steps is within the chain's range
    if 0 < burnin_steps < effective_n_steps:
        ax.axvline(burnin_steps, color='red', linestyle=':', linewidth=2, label=f'Burn-in Cutoff ({burnin_steps} steps)')
        ax.legend(loc='upper right') # Show legend if burn-in line is plotted
    elif burnin_steps >= effective_n_steps:
        logger.info(f"ℹ️ Burn-in ({burnin_steps}) is beyond or at chain length ({effective_n_steps}). Not shown in animation for walker {walker_idx}, event {event_id}.")


    line, = ax.plot([], [], lw=1.5, color='dodgerblue', label=f'Walker {walker_idx} Path')
    point, = ax.plot([], [], 'o', color='red', markersize=8, label='Current Position')
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    
    # Update legend to include line and point if not showing burn-in legend
    if not (0 < burnin_steps < effective_n_steps):
        handles, labels = ax.get_legend_handles_labels()
        if not handles: # Add default legend if no burn-in line
             ax.legend([line, point], [f'Walker {walker_idx} Path', 'Current Position'], loc='upper right')


    def init_animation_func():
        line.set_data([], [])
        point.set_data([], [])
        step_text.set_text('')
        return line, point, step_text

    def update_animation_func(frame_idx):
        # frame_idx is the actual index from steps_to_plot_indices
        current_mcmc_step_to_show = steps_to_plot_indices[frame_idx]
        
        # Ensure x_data and y_data are indexed correctly up to current_mcmc_step_to_show
        x_data_path = np.arange(0, current_mcmc_step_to_show + 1)
        y_data_path = single_walker_h0_history[: current_mcmc_step_to_show + 1]
        
        line.set_data(x_data_path, y_data_path)
        
        current_value = single_walker_h0_history[current_mcmc_step_to_show]
        point.set_data([current_mcmc_step_to_show], [current_value])

        label_param = "$H_0$" if dim_to_plot == 0 else "$\\alpha$"
        step_text.set_text(f'Step: {current_mcmc_step_to_show}\n{label_param}: {current_value:.2f}')
        return line, point, step_text

    num_animation_frames = len(steps_to_plot_indices)
    if num_animation_frames == 0: # Should be caught earlier, but as a safeguard
        logger.warning(f"⚠️ No frames to animate for walker {walker_idx} (event {event_id}) with plot_interval={plot_interval}. Chain length: {total_steps_in_chain}.")
        plt.close(fig)
        return

    # Interval is in milliseconds. max(20, ...) ensures it's not too fast for display.
    anim_interval = max(20, 1000 // fps) 
    ani = FuncAnimation(fig, update_animation_func, frames=num_animation_frames,
                        init_func=init_animation_func, blit=True, interval=anim_interval)

    base_output_filename = output_filename_template.format(walker_idx=walker_idx, event_id=event_id)
    full_output_path = os.path.join(OUTPUT_DIR, base_output_filename)
    try:
        logger.info(f"Attempting to save animation to {full_output_path} (this may take a while)...")
        ani.save(full_output_path, writer='pillow', fps=fps)
        logger.info(f"✅ Animation saved successfully: {full_output_path}")
    except Exception as e:
        logger.error(f"❌ Error saving animation for event {event_id}, walker {walker_idx}: {e}")
        logger.error("  Please ensure you have 'Pillow' installed (e.g., pip install Pillow).")
        logger.error("  Alternatively, you might need to install ImageMagick and specify writer='imagemagick'.")
    finally:
        plt.close(fig) # Ensure figure is closed after saving or error

def animate_mean_log_prob(
    sampler,
    event_name: str,
    n_total_steps: int,
    burnin_steps: int | None = None,
    output_dir: str = OUTPUT_DIR,
    output_filename_template: str = "log_prob_animation_{event_name}.gif",
    plot_interval: int = 10,
    fps: int = 20,
):
    """Create an animated GIF showing evolution of mean log posterior probability."""
    logger.info(f"Generating mean log probability animation for {event_name}...")

    try:
        log_prob = sampler.get_log_prob()
        logger.debug(f"Log probability shape: {log_prob.shape if log_prob is not None else 'None'}")
        if log_prob is None or log_prob.size == 0:
            logger.error(f"❌ No log probabilities available for {event_name}")
            return None
    except Exception as e:
        logger.error(f"❌ Error getting log probabilities for {event_name}: {e}")
        return None

    # Validate log_prob shape and adjust n_steps if needed
    if log_prob.shape[0] != n_total_steps:
        n_steps = log_prob.shape[0]
        logger.info(
            f"Adjusting n_steps from {n_total_steps} to {n_steps} based on actual chain length"
        )
    else:
        n_steps = n_total_steps

    # Calculate mean log probability
    try:
        mean_log_prob = np.mean(log_prob, axis=1)
        logger.debug(f"Mean log probability shape: {mean_log_prob.shape}")
        if not np.any(np.isfinite(mean_log_prob)):
            logger.error(f"❌ No finite values in mean log probability for {event_name}")
            return None
    except Exception as e:
        logger.error(f"❌ Error calculating mean log probability: {e}")
        return None

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Step Number")
    ax.set_ylabel("Mean Log Posterior Probability")
    ax.set_title(f"Evolution of Mean Log Probability - {event_name}")

    # Set plot limits with padding
    y_min = np.min(mean_log_prob[np.isfinite(mean_log_prob)])
    y_max = np.max(mean_log_prob[np.isfinite(mean_log_prob)])
    y_padding = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    ax.set_xlim(0, n_steps)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Add burn-in line if specified
    if burnin_steps is not None and 0 < burnin_steps < n_steps:
        ax.axvline(burnin_steps, ls="--", c="red", label=f"Burn-in ({burnin_steps})")
        ax.legend()

    # Initialize plot elements
    line, = ax.plot([], [], lw=1.5, color="tab:blue")
    point, = ax.plot([], [], "o", color="tab:orange")
    step_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7),
    )

    # Calculate steps to plot
    steps = np.arange(0, n_steps, max(1, plot_interval))
    logger.debug(f"Number of steps to plot: {len(steps)}")
    
    if len(steps) == 0:
        logger.error(f"❌ No steps to plot for {event_name} with plot_interval={plot_interval}")
        return None
    
    # Ensure we include the final step
    if steps[-1] != n_steps - 1:
        steps = np.append(steps, n_steps - 1)
        logger.debug(f"Added final step. Total steps to plot: {len(steps)}")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        step_text.set_text("")
        return line, point, step_text

    def update(frame_idx):
        try:
            if frame_idx >= len(steps):
                logger.error(f"❌ Frame index {frame_idx} out of range for {len(steps)} steps")
                return line, point, step_text
                
            step = steps[frame_idx]
            if step >= len(mean_log_prob):
                logger.error(f"❌ Step {step} out of range for mean_log_prob length {len(mean_log_prob)}")
                return line, point, step_text
                
            # Create sequences for plotting
            x_data = np.arange(0, step + 1)
            y_data = mean_log_prob[: step + 1]
            
            # Ensure data is in the correct format for set_data
            line.set_data(x_data.tolist(), y_data.tolist())
            point.set_data([step], [mean_log_prob[step]])  # Point data must be lists/arrays
            step_text.set_text(f"Step: {step}\nMean LogProb: {mean_log_prob[step]:.2f}")
            return line, point, step_text
        except Exception as e:
            logger.error(f"❌ Error in update function for frame {frame_idx}: {e}")
            return line, point, step_text

    # Create animation
    try:
        logger.debug(f"Creating animation with {len(steps)} frames")
        anim = FuncAnimation(
            fig,
            update,
            frames=len(steps),
            init_func=init,
            blit=True,
            interval=max(20, 1000 // fps),
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename_template.format(event_name=event_name))
        
        # Save animation
        logger.debug(f"Saving animation to {output_path}")
        anim.save(output_path, writer="pillow", fps=fps)
        logger.info(f"✅ Mean log probability animation saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Error creating/saving animation for {event_name}: {e}")
        logger.error(f"  Steps array length: {len(steps)}")
        logger.error(f"  Mean log probability length: {len(mean_log_prob)}")
        logger.error(f"  n_steps: {n_steps}")
        return None
    finally:
        plt.close(fig)

# -------------------------------------------------------------------
# Main script execution for viz.py
# -------------------------------------------------------------------
def main():
    """Main function for visualization: fetches samples, processes catalog, plots, and optionally runs H0 MCMC."""
    
    parser = argparse.ArgumentParser(description="Visualize GW event data, skymaps, and run H0 MCMC.")
    parser.add_argument(
        "event_name", 
        nargs='?', 
        default=DEFAULT_EVENT_NAME_VIZ, 
        help=f"Name of the GW event to process (default: {DEFAULT_EVENT_NAME_VIZ})"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging level."
    )
    args = parser.parse_args()

    # Configure basic logging based on CLI arguments
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensuring output directory exists: {os.path.abspath(OUTPUT_DIR)}")

    effective_cache_dir = configure_astropy_cache(DEFAULT_CACHE_DIR_NAME)
    if not effective_cache_dir: logger.critical("❌ CRITICAL: Failed to configure cache. Exiting."); sys.exit(1)

    current_event_name = args.event_name # Use event_name from argparse
    logger.info(f"--- Starting Full Visualization Analysis for Event: {current_event_name} (Log Level: {logging.getLevelName(logger.getEffectiveLevel())}) ---")
    
    # Initialize H0 to be used for the 3D plot; default or from MCMC
    H0_for_3d_dist_calc = 70.0  # Default H0 km/s/Mpc
    logger.info(f"ℹ️ Default H0 for 3D plot distance calculations initialized to: {H0_for_3d_dist_calc} km/s/Mpc")

    # 1. Fetch GW data and extract parameters
    logger.info(f"Fetching GW data for {current_event_name}...")
    success, gw_data_obj = fetch_candidate_data(current_event_name, effective_cache_dir)
    if not success: logger.error(f"❌ Failed to fetch GW data: {gw_data_obj}. Some visualizations might be skipped."); gw_data_obj = None
    
    dL_gw_samples, ra_gw_samples, dec_gw_samples = (None, None, None)
    if gw_data_obj:
        dL_gw_samples, ra_gw_samples, dec_gw_samples = extract_gw_event_parameters(gw_data_obj, current_event_name)

    if ra_gw_samples is None or dec_gw_samples is None:
        logger.error(f"❌ Failed to get essential RA/Dec samples for {current_event_name}. Sky maps will be skipped. MCMC may also be affected.")
    else:
        logger.info(f"Successfully extracted {len(ra_gw_samples)} RA/Dec GW samples.")
        if dL_gw_samples is not None: logger.info(f"  and {len(dL_gw_samples)} dL samples.")

    # Estimate event-specific host_z_max
    if dL_gw_samples is not None and len(dL_gw_samples) > 0:
        logger.info("Estimating event-specific host_z_max based on dL_gw_samples...")
        current_host_z_max_dynamic = estimate_event_specific_z_max(
            dL_gw_samples,
            percentile_dL=95.0,     
            z_margin_factor=1.2,    
            min_z_max_val=0.01,     
            max_z_max_val=0.3       # Max z for this adaptive cut; GLADE cleaning has a broader one
        )
        logger.info(f"Using dynamically estimated host_z_max: {current_host_z_max_dynamic:.4f} for event {current_event_name}")
    else:
        # Fallback if dL_gw_samples are not available for some reason
        current_host_z_max_dynamic = VIZ_HOST_Z_MAX_FALLBACK # Use the script's static default fallback
        logger.warning(f"dL_gw_samples not available. Using static VIZ_HOST_Z_MAX_FALLBACK: {current_host_z_max_dynamic:.4f}")

    # 2. Generate basic sky probability map (if RA/Dec available)
    prob_map_gw_for_plots, sky_mask_for_plots = (None, None)
    if ra_gw_samples is not None and dec_gw_samples is not None:
        prob_map_gw_for_plots, sky_mask_for_plots, _ = generate_sky_map_and_credible_region(
            ra_gw_samples, dec_gw_samples, 
            nside=VIZ_NSIDE_SKYMAP, 
            cdf_threshold=VIZ_PROB_THRESHOLD_CDF
        )
        if prob_map_gw_for_plots is not None and prob_map_gw_for_plots.sum() > 0:
            # plot_basic_sky_probability_map(
            #     prob_map_gw_for_plots,
            #     VIZ_NSIDE_SKYMAP,
            #     current_event_name
            # )
            pass
        else:
            logger.warning("Skipping basic sky probability map due to issues generating it.")
    else:
        logger.warning("Skipping basic sky probability map due to missing RA/Dec samples.")

    logger.info("--- Preparing data for skymap with galaxies and H0 MCMC ---")
    # 3. Load and clean galaxy catalog
    glade_raw_df = download_and_load_galaxy_catalog(catalog_type=VIZ_CATALOG_TYPE)
    if glade_raw_df.empty: logger.error(f"❌ {VIZ_CATALOG_TYPE.upper()} catalog empty after loading. Galaxy-dependent plots and MCMC will be skipped."); return
    
    glade_cleaned_df = clean_galaxy_catalog(glade_raw_df, range_filters=DEFAULT_RANGE_CHECKS)
    if glade_cleaned_df.empty: logger.error(f"❌ {VIZ_CATALOG_TYPE.upper()} catalog empty after cleaning. Galaxy-dependent plots and MCMC will be skipped."); return

    # 4. Select candidate hosts for visualization and potential MCMC
    final_candidate_hosts_df = pd.DataFrame() 
    if sky_mask_for_plots is not None and sky_mask_for_plots.any():
        spatially_selected_galaxies_df = select_galaxies_in_sky_region(
            glade_cleaned_df, sky_mask_for_plots, nside=VIZ_NSIDE_SKYMAP
        )
        if not spatially_selected_galaxies_df.empty:
            # plot_redshift_distribution(
            #     spatially_selected_galaxies_df, current_event_name, 
            #     "Spatially Selected (in C.R.)", host_z_max_cutoff=current_host_z_max_dynamic
            # )
            intermediate_hosts_df = filter_galaxies_by_redshift(spatially_selected_galaxies_df, current_host_z_max_dynamic)
            final_candidate_hosts_df = apply_specific_galaxy_corrections(
                intermediate_hosts_df, current_event_name, VIZ_GALAXY_CORRECTIONS
            )
    else:
        logger.warning("Sky mask for CR not available or empty. Attempting selection from full catalog with z-cut for MCMC if needed.")
        # Fallback: if no sky_mask, consider all cleaned galaxies up to current_host_z_max_dynamic for MCMC
        intermediate_hosts_df = filter_galaxies_by_redshift(glade_cleaned_df, current_host_z_max_dynamic)
        final_candidate_hosts_df = apply_specific_galaxy_corrections(
            intermediate_hosts_df, current_event_name, VIZ_GALAXY_CORRECTIONS
        )

    if final_candidate_hosts_df.empty:
        logger.warning(f"⚠️ No candidate host galaxies identified for {current_event_name} after selection process. Overlay plot might be sparse. MCMC for H0 might be skipped or use broader galaxy set.")
    else:
        logger.info(f"Final number of candidate hosts for {current_event_name} for overlay & H0: {len(final_candidate_hosts_df)}")
        # plot_redshift_distribution(
        #     final_candidate_hosts_df, current_event_name, 
        #     "Final Candidate Hosts (for MCMC/Overlay)", host_z_max_cutoff=current_host_z_max_dynamic
        # )
        pass

    # Plot skymap with galaxies (if prob_map_gw and sky_mask_for_plots were generated)
    if prob_map_gw_for_plots is not None and sky_mask_for_plots is not None:
        # plot_skymap_with_galaxies(
        #     prob_map_gw=prob_map_gw_for_plots,
        #     sky_mask_boolean=sky_mask_for_plots,
        #     all_galaxies_df=glade_cleaned_df,
        #     selected_hosts_df=final_candidate_hosts_df, # Use the most refined set for highlighting
        #     nside=VIZ_NSIDE_SKYMAP,
        #     event_name=current_event_name,
        #     cred_level_percent=VIZ_PROB_THRESHOLD_CDF * 100,
        #     host_z_max=current_host_z_max_dynamic
        # )
        pass
    else:
        logger.warning("Skipping skymap with galaxies due to missing probability map or sky mask.")

    # 5. Perform MCMC for H0 if dL samples and some candidate hosts are available
    can_run_mcmc = dL_gw_samples is not None and len(dL_gw_samples) > 0
    if can_run_mcmc and not final_candidate_hosts_df.empty:
        logger.info(f"\n--- Proceeding with MCMC H0 estimation for {current_event_name} (within viz.py) ---")
        try:
            logger.debug(f"DEBUG VIZ: dL_gw_samples (first 5): {dL_gw_samples[:5]}, len: {len(dL_gw_samples)}, any non-finite: {np.any(~np.isfinite(dL_gw_samples))}")
            zs_for_mcmc = final_candidate_hosts_df['z'].values
            logger.debug(f"DEBUG VIZ: host_galaxies_z (first 5): {zs_for_mcmc[:5]}, len: {len(zs_for_mcmc)}, any non-finite: {np.any(~np.isfinite(zs_for_mcmc))}")
            
            log_likelihood_h0_func = get_log_likelihood_h0(
                dL_gw_samples, zs_for_mcmc, # Use the printed variable
                DEFAULT_SIGMA_V_PEC, DEFAULT_C_LIGHT, DEFAULT_OMEGA_M,
                DEFAULT_H0_PRIOR_MIN, DEFAULT_H0_PRIOR_MAX
            )
        except ValueError as ve:
            logger.error(f"❌ Error creating H0 likelihood for viz: {ve}. Skipping MCMC."); log_likelihood_h0_func = None

        if log_likelihood_h0_func:
            # Determine number of cores for parallelization
            n_cores = os.cpu_count()
            logger.info(f"Utilizing {n_cores} cores for MCMC parallelization.")

            with InterruptiblePool(n_cores) as pool:
                mcmc_sampler = run_mcmc_h0(
                    log_likelihood_h0_func, current_event_name,
                    n_walkers=VIZ_MCMC_N_WALKERS, n_steps=VIZ_MCMC_N_STEPS,
                    pool=pool # Pass the pool object
                )
                if mcmc_sampler:
                    plot_mcmc_trace(mcmc_sampler, current_event_name, VIZ_MCMC_BURNIN, VIZ_MCMC_N_WALKERS, DEFAULT_MCMC_N_DIM)
                    flat_h0_samples = process_mcmc_samples(mcmc_sampler, current_event_name, VIZ_MCMC_BURNIN, VIZ_MCMC_THIN_BY, DEFAULT_MCMC_N_DIM)
                    if flat_h0_samples is not None and len(flat_h0_samples) > 0:
                        save_and_plot_h0_posterior_viz(flat_h0_samples, current_event_name, len(final_candidate_hosts_df))
                        # Update H0_for_3d_dist_calc with MCMC median result
                        _, q50_mcmc, _ = np.percentile(flat_h0_samples, [16, 50, 84])
                        H0_for_3d_dist_calc = q50_mcmc
                        logger.info(f"ℹ️ H0 for 3D plot distance calculations updated to MCMC median: {H0_for_3d_dist_calc:.1f} km/s/Mpc")
                    else:
                        logger.warning(f"Skipping H0 posterior plot for {current_event_name} due to no valid MCMC samples after processing.")
                        logger.warning(f"⚠️ MCMC samples not available after processing; 3D plot will use H0: {H0_for_3d_dist_calc:.1f} km/s/Mpc.")

                    # Create and save walker animation GIF
                    create_walker_animation_gif(
                        sampler=mcmc_sampler,
                        n_steps=VIZ_MCMC_N_STEPS,       # Total steps configured for MCMC
                        burnin_steps=VIZ_MCMC_BURNIN,   # Burn-in steps configured
                        event_id=current_event_name,
                        walker_idx=0,                   # Animate the first walker by default
                        plot_interval=max(1, VIZ_MCMC_N_STEPS // 100), # Aim for ~100 frames, ensure at least 1
                        fps=15
                    )

                    # Create and save mean log probability animation GIF
                    animate_mean_log_prob(
                        sampler=mcmc_sampler,
                        event_name=current_event_name,
                        n_total_steps=VIZ_MCMC_N_STEPS,
                        burnin_steps=VIZ_MCMC_BURNIN,
                        output_dir=OUTPUT_DIR,
                        plot_interval=max(1, VIZ_MCMC_N_STEPS // 100),
                        fps=15,
                    )
                else:
                    logger.error(f"Skipping MCMC post-processing for {current_event_name} because MCMC run failed or returned no sampler.")
                    logger.warning(f"⚠️ MCMC run failed; 3D plot will use H0: {H0_for_3d_dist_calc:.1f} km/s/Mpc.")
    elif not can_run_mcmc:
        logger.info(f"\n--- Skipping MCMC H0 estimation for {current_event_name} due to missing dL samples. 3D plot will use H0: {H0_for_3d_dist_calc:.1f} km/s/Mpc. ---")
    elif final_candidate_hosts_df.empty: # This implies dL_gw_samples were present, but no hosts found
        logger.info(f"\n--- Skipping MCMC H0 estimation for {current_event_name} because no candidate host galaxies were identified. 3D plot will use H0: {H0_for_3d_dist_calc:.1f} km/s/Mpc. ---")

    # 6. Generate 3D Localization Plot
    logger.info(f"--- Preparing for 3D Localization Plot for {current_event_name} ---")
    if ra_gw_samples is not None and dec_gw_samples is not None and dL_gw_samples is not None:
        
        # Ensure final_candidate_hosts_df is a DataFrame, even if empty.
        # The plot_3d_localization_with_galaxies function handles an empty DataFrame gracefully.
        effective_hosts_df_for_3d = final_candidate_hosts_df
        if not isinstance(final_candidate_hosts_df, pd.DataFrame):
            logger.warning(f"⚠️ final_candidate_hosts_df was not a DataFrame (type: {type(final_candidate_hosts_df)}). Using empty DataFrame for 3D plot.")
            effective_hosts_df_for_3d = pd.DataFrame()
            if final_candidate_hosts_df is not None and len(final_candidate_hosts_df) > 0 : # if it was non-empty but not a df
                 logger.warning("  Original final_candidate_hosts_df was not empty, this might indicate an issue.")


        plot_3d_localization_with_galaxies(
            event_name=current_event_name,
            ra_gw_samples=ra_gw_samples,
            dec_gw_samples=dec_gw_samples,
            dL_gw_samples=dL_gw_samples,
            candidate_hosts_df=effective_hosts_df_for_3d,
            H0_for_galaxy_dist=H0_for_3d_dist_calc,
            omega_m_for_galaxy_dist=DEFAULT_OMEGA_M, # Imported from h0_mcmc_analyzer
            num_gw_samples_to_plot=1000, # Default, can be made a VIZ_ constant
            output_dir=OUTPUT_DIR # Pass the output directory to the 3D plot function
        )
    else:
        logger.warning(f"⚠️ Skipping 3D localization plot for {current_event_name} due to missing essential GW sample data (RA, Dec, or dL).")

    logger.info(f"\n--- Visualization Script Finished for {current_event_name} ---")
    logger.info("Close all plot windows to exit script fully if plots are blocking.")
    plt.show() # Final show to ensure all non-blocking plots are displayed until closed

if __name__ == "__main__":
    main()
