import numpy as np
import pandas as pd # Used for type hinting candidate_hosts_df
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d', do not remove
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import logging
import os # Added os module

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