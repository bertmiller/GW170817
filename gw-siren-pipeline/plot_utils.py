import numpy as np
import pandas as pd # Used for type hinting candidate_hosts_df
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d', do not remove
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import logging
import os # Added os module
from matplotlib.animation import FuncAnimation # Added for GIF animation

logger = logging.getLogger(__name__)

def plot_3d_localization_with_galaxies(
    event_name: str,
    ra_gw_samples: np.ndarray,
    dec_gw_samples: np.ndarray,
    dL_gw_samples: np.ndarray,
    candidate_hosts_df: pd.DataFrame,
    H0_for_galaxy_dist: float,
    omega_m_for_galaxy_dist: float,
    output_filename_template: str = "3d_localization_{event_name}.pdf",
    num_gw_samples_to_plot: int = 1000,
    output_dir: str = "output"  # Added output_dir argument with default
):
    """
    Generates a 3D scatter plot visualizing GW event localization and candidate host galaxies.

    Args:
        event_name (str): Name of the GW event.
        ra_gw_samples (np.ndarray): Right Ascension posterior samples for GW event (degrees).
        dec_gw_samples (np.ndarray): Declination posterior samples for GW event (degrees).
        dL_gw_samples (np.ndarray): Luminosity distance posterior samples for GW event (Mpc).
        candidate_hosts_df (pd.DataFrame): DataFrame with candidate host galaxies.
                                           Must contain 'ra', 'dec', 'z' columns.
                                           Optional 'PGC' column for labels.
        H0_for_galaxy_dist (float): Hubble constant (km/s/Mpc) for galaxy distance calculation.
        omega_m_for_galaxy_dist (float): Matter density parameter (Omega_M) for cosmology.
        output_filename_template (str, optional): Template for the output plot filename.
                                                  Defaults to "3d_localization_{event_name}.pdf".
        num_gw_samples_to_plot (int, optional): Number of GW samples to plot. Defaults to 1000.
        output_dir (str, optional): Directory to save the output plot. Defaults to "output".
    """
    logger.info(f"\nGenerating 3D localization plot for {event_name} (will save to '{output_dir}')...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plot_gw_data = True
    if ra_gw_samples is None or dec_gw_samples is None or dL_gw_samples is None or \
       len(ra_gw_samples) == 0 or len(dec_gw_samples) == 0 or len(dL_gw_samples) == 0:
        logger.warning(f"⚠️ GW samples (RA, Dec, or dL) are missing or empty for {event_name}. GW data will not be plotted.")
        plot_gw_data = False
    elif not (len(ra_gw_samples) == len(dec_gw_samples) == len(dL_gw_samples)):
        min_len = min(len(ra_gw_samples), len(dec_gw_samples), len(dL_gw_samples))
        logger.warning(f"⚠️ Mismatch in lengths of GW sample arrays for {event_name} ({len(ra_gw_samples)}, {len(dec_gw_samples)}, {len(dL_gw_samples)}). Truncating to shortest length: {min_len}.")
        ra_gw_samples = ra_gw_samples[:min_len]
        dec_gw_samples = dec_gw_samples[:min_len]
        dL_gw_samples = dL_gw_samples[:min_len]
        if min_len == 0:
            plot_gw_data = False


    # 1. Prepare GW Sample Coordinates for Plotting
    x_gw_plot, y_gw_plot, z_gw_plot = np.array([]), np.array([]), np.array([])
    num_plotted_gw_samples = 0
    if plot_gw_data:
        if len(ra_gw_samples) > num_gw_samples_to_plot:
            rng = np.random.default_rng()
            indices = rng.choice(len(ra_gw_samples), size=num_gw_samples_to_plot, replace=False)
        else:
            indices = np.arange(len(ra_gw_samples)) # Use all samples
        
        num_plotted_gw_samples = len(indices)

        ra_rad_gw = np.deg2rad(ra_gw_samples[indices])
        dec_rad_gw = np.deg2rad(dec_gw_samples[indices])
        dist_gw = dL_gw_samples[indices]

        x_gw_plot = dist_gw * np.cos(dec_rad_gw) * np.cos(ra_rad_gw)
        y_gw_plot = dist_gw * np.cos(dec_rad_gw) * np.sin(ra_rad_gw)
        z_gw_plot = dist_gw * np.sin(dec_rad_gw)
        logger.info(f"Prepared {num_plotted_gw_samples} GW samples for 3D plot.")

    # 2. Prepare Galaxy Coordinates for Plotting
    x_gal_plot, y_gal_plot, z_gal_plot = [], [], []
    # galaxy_pgc_labels = [] # For optional PGC labels, can be very cluttered

    if not candidate_hosts_df.empty and \
       all(col in candidate_hosts_df.columns for col in ['ra', 'dec', 'z']):
        cosmo = FlatLambdaCDM(H0=H0_for_galaxy_dist * u.km / u.s / u.Mpc, Om0=omega_m_for_galaxy_dist)
        
        processed_galaxies = 0
        for _, galaxy_row in candidate_hosts_df.iterrows():
            try:
                # Ensure RA, Dec, z are valid numbers
                gal_z = float(galaxy_row['z'])
                gal_ra_deg = float(galaxy_row['ra'])
                gal_dec_deg = float(galaxy_row['dec'])

                if gal_z < 0: # Redshift should be non-negative for dL calculation
                    logger.info(f"ℹ️ Skipping galaxy with negative redshift z={gal_z:.3f} (PGC: {galaxy_row.get('PGC', 'N/A')}).")
                    continue
                
                # Luminosity distance calculation
                # astropy's luminosity_distance(0) correctly returns 0 Mpc.
                dL_galaxy_mpc = cosmo.luminosity_distance(gal_z).to(u.Mpc).value

                # Convert to Cartesian
                gal_ra_rad = np.deg2rad(gal_ra_deg)
                gal_dec_rad = np.deg2rad(gal_dec_deg)

                x_gal_plot.append(dL_galaxy_mpc * np.cos(gal_dec_rad) * np.cos(gal_ra_rad))
                y_gal_plot.append(dL_galaxy_mpc * np.cos(gal_dec_rad) * np.sin(gal_ra_rad))
                z_gal_plot.append(dL_galaxy_mpc * np.sin(gal_dec_rad))
                processed_galaxies += 1
                # if 'PGC' in galaxy_row and pd.notna(galaxy_row['PGC']):
                #     galaxy_pgc_labels.append(f"PGC {galaxy_row['PGC']}")
                # else:
                #     galaxy_pgc_labels.append(None)
            except ValueError as ve:
                logger.warning(f"⚠️ Skipping galaxy due to invalid data (e.g., non-numeric RA/Dec/z): {ve}. Row: PGC {galaxy_row.get('PGC', 'N/A')}, data {galaxy_row.to_dict()}")
            except Exception as e: # Catch other astropy or calculation errors
                logger.warning(f"⚠️ Error processing galaxy PGC {galaxy_row.get('PGC', 'N/A')}: {e}")
        
        if processed_galaxies > 0:
            logger.info(f"Prepared {processed_galaxies} candidate host galaxies for 3D plot.")
        else:
            logger.info("ℹ️ No valid candidate host galaxies processed for 3D plot.")
            
    elif candidate_hosts_df.empty:
        logger.info(f"ℹ️ Candidate hosts DataFrame is empty for {event_name}. No galaxies will be plotted.")
    else: # DataFrame not empty, but missing required columns
        logger.warning(f"⚠️ Candidate hosts DataFrame for {event_name} is missing 'ra', 'dec', or 'z' columns. Galaxies will not be plotted.")

    # 3. Create the 3D Plot
    if not plot_gw_data and not x_gal_plot:
        logger.warning(f"⚠️ No data available to plot for {event_name} (neither GW samples nor galaxies). Skipping 3D plot generation.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if num_plotted_gw_samples > 0:
        ax.scatter(x_gw_plot, y_gw_plot, z_gw_plot, color='cornflowerblue', alpha=0.1, s=15, label=f'GW Samples ({num_plotted_gw_samples} pts)')

    if x_gal_plot: # Check if list is not empty
        ax.scatter(x_gal_plot, y_gal_plot, z_gal_plot, color='orangered', marker='o', s=70, edgecolor='black', depthshade=True, label=f'Candidate Hosts ({len(x_gal_plot)})')
        # Optional: Add labels for galaxies. This can be very cluttered.
        # for i in range(len(x_gal_plot)):
        #     if galaxy_pgc_labels[i]:
        #         ax.text(x_gal_plot[i], y_gal_plot[i], z_gal_plot[i], galaxy_pgc_labels[i], size=7, zorder=10, color='k')

    ax.set_xlabel("X [Mpc]")
    ax.set_ylabel("Y [Mpc]")
    ax.set_zlabel("Z [Mpc]")
    title_str = (f"3D GW Localization & Candidate Host Galaxies - {event_name}\n"
                 f"($H_0={H0_for_galaxy_dist:.1f}$ km/s/Mpc, $\Omega_M={omega_m_for_galaxy_dist:.3f}$)")
    ax.set_title(title_str)

    if num_plotted_gw_samples > 0 or x_gal_plot:
        ax.legend(loc='best')

    # Set axis limits to be somewhat cubic and encompass all data
    all_coords_x = np.concatenate((x_gw_plot, np.array(x_gal_plot)))
    all_coords_y = np.concatenate((y_gw_plot, np.array(y_gal_plot)))
    all_coords_z = np.concatenate((z_gw_plot, np.array(z_gal_plot)))

    if len(all_coords_x) > 0: # Check if there's any data at all
        min_x, max_x = all_coords_x.min(), all_coords_x.max()
        min_y, max_y = all_coords_y.min(), all_coords_y.max()
        min_z, max_z = all_coords_z.min(), all_coords_z.max()

        mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
        
        half_range = max(max_x - min_x, max_y - min_y, max_z - min_z, 10) / 2 # Ensure some minimal range
        # Add a small buffer to the range
        buffer_factor = 1.1 
        half_range *= buffer_factor

        ax.set_xlim(mid_x - half_range, mid_x + half_range)
        ax.set_ylim(mid_y - half_range, mid_y + half_range)
        ax.set_zlim(mid_z - half_range, mid_z + half_range)
    else: # No data, use default limits or a small box around origin
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([-50, 50])


    ax.view_init(elev=25., azim=45) # Adjust viewing angle (elevation, azimuth) as preferred

    base_filename = output_filename_template.format(event_name=event_name)
    full_output_path = os.path.join(output_dir, base_filename)
    try:
        plt.savefig(full_output_path, bbox_inches='tight', dpi=150)
        logger.info(f"✅ 3D localization plot saved to {full_output_path}")
    except Exception as e:
        logger.error(f"❌ Error saving 3D localization plot '{full_output_path}': {e}")
    
    plt.show(block=False)
    plt.pause(1) # Allow plot to render in non-blocking mode
    # Consider plt.close(fig) if generating many plots in a loop to save memory

def create_mean_log_prob_animation_gif(
    sampler,  # emcee.EnsembleSampler object
    event_name: str,
    n_total_steps: int, # Total steps the sampler ran for
    output_dir: str = "output",
    burnin_steps: int | None = None,
    output_filename_template: str = "log_prob_animation_{event_name}.gif",
    plot_interval: int = 10, # Animate every N-th step
    fps: int = 15
):
    """
    Creates a GIF animation of the mean log posterior probability over MCMC steps.

    Args:
        sampler: The emcee.EnsembleSampler object after run_mcmc.
        event_name (str): Name of the GW event for titles and filenames.
        n_total_steps (int): The total number of steps the sampler ran for.
        output_dir (str, optional): Directory to save the output GIF. Defaults to "output".
        burnin_steps (int, optional): If provided, a vertical line indicating burn-in.
        output_filename_template (str, optional): Template for the output GIF filename.
        plot_interval (int, optional): Interval for plotting frames. Defaults to 10.
        fps (int, optional): Frames per second for the GIF. Defaults to 15.
    """
    logger.info(f"Generating mean log probability animation for {event_name}...")

    try:
        log_probs = sampler.get_log_prob()  # Shape: (n_steps, n_walkers)
        if log_probs is None or log_probs.size == 0:
            logger.error(f"❌ No log probability data found in sampler for event {event_name}.")
            return
        if log_probs.shape[0] != n_total_steps:
            logger.warning(
                f"⚠️ Sampler log_probs steps ({log_probs.shape[0]}) mismatch "
                f"n_total_steps ({n_total_steps}). Using sampler's step count."
            )
            actual_n_steps = log_probs.shape[0]
        else:
            actual_n_steps = n_total_steps
            
        mean_log_prob = np.mean(log_probs, axis=1)
    except Exception as e:
        logger.error(f"❌ Error extracting log probability from sampler for {event_name}: {e}")
        return

    if mean_log_prob.size == 0:
        logger.warning(f"⚠️ Mean log probability array is empty for {event_name}. Cannot create animation.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine y-axis limits with padding
    min_lp = np.min(mean_log_prob)
    max_lp = np.max(mean_log_prob)
    padding_lp = (max_lp - min_lp) * 0.05 if (max_lp - min_lp) > 0 else 1.0 # Add padding, handle flat case
    y_min_limit = min_lp - padding_lp
    y_max_limit = max_lp + padding_lp

    ax.set_xlim(0, actual_n_steps)
    ax.set_ylim(y_min_limit, y_max_limit)
    ax.set_xlabel("MCMC Step Number")
    ax.set_ylabel("Mean Log Posterior Probability")
    ax.set_title(f"Evolution of Mean Log Probability - {event_name}")
    ax.grid(True, alpha=0.4)

    if burnin_steps is not None and 0 < burnin_steps < actual_n_steps:
        ax.axvline(burnin_steps, color='red', linestyle=':', linewidth=1.5, label=f'Burn-in Cutoff ({burnin_steps})')
        ax.legend(loc='lower right')

    line, = ax.plot([], [], lw=1.5, color='teal')
    point, = ax.plot([], [], 'o', color='orangered', markersize=5)
    step_text = ax.text(0.02, 0.05, '', transform=ax.transAxes, fontsize=9,
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

    steps_to_animate_indices = np.arange(0, actual_n_steps, plot_interval)
    if not steps_to_animate_indices.size or steps_to_animate_indices[-1] != actual_n_steps -1:
        # Ensure the last step is always included if not perfectly divisible
        if actual_n_steps > 0 :
             steps_to_animate_indices = np.append(steps_to_animate_indices, actual_n_steps - 1)
             steps_to_animate_indices = np.unique(steps_to_animate_indices) # Remove duplicates if any

    if not steps_to_animate_indices.size:
        logger.warning(f"⚠️ No frames to animate for mean log prob (event {event_name}) with plot_interval={plot_interval} and steps {actual_n_steps}.")
        plt.close(fig)
        return

    def init_animation():
        line.set_data([], [])
        point.set_data([], [])
        step_text.set_text('')
        return line, point, step_text

    def update_animation(frame_num_idx):
        current_mcmc_step_to_show = steps_to_animate_indices[frame_num_idx]

        # Data for the line trace up to the current step
        x_data_trace = np.arange(0, current_mcmc_step_to_show + 1)
        y_data_trace = mean_log_prob[:current_mcmc_step_to_show + 1]
        line.set_data(x_data_trace, y_data_trace)

        # Current point
        current_lp_value = mean_log_prob[current_mcmc_step_to_show]
        point.set_data([current_mcmc_step_to_show], [current_lp_value])

        step_text.set_text(f'Step: {current_mcmc_step_to_show}\nMean LogProb: {current_lp_value:.2f}')
        return line, point, step_text

    num_animation_frames = len(steps_to_animate_indices)
    anim_interval_ms = max(20, 1000 // fps)

    ani = FuncAnimation(fig, update_animation, frames=num_animation_frames,
                        init_func=init_animation, blit=True, interval=anim_interval_ms)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_output_filename = output_filename_template.format(event_name=event_name)
    full_output_path = os.path.join(output_dir, base_output_filename)

    try:
        logger.info(f"Attempting to save mean log probability animation to {full_output_path}...")
        ani.save(full_output_path, writer='pillow', fps=fps)
        logger.info(f"✅ Mean log probability animation saved: {full_output_path}")
    except Exception as e:
        logger.error(f"❌ Error saving mean log probability animation for {event_name}: {e}")
        logger.error("  Ensure 'Pillow' is installed (pip install Pillow).")
    finally:
        plt.close(fig)