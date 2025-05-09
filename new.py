#!/usr/bin/env python
"""
Reproduce Fishbach et al. (2019) statistical-standard-siren analysis
for GW170817 using the GLADE v2.4 galaxy catalogue.

Refactored for modularity and reusability.

Outputs:
  - 'H0_samples_<event_name>.npy'
  - 'H0_posterior_<event_name>.pdf'
"""
import os, sys
import urllib.request
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
import emcee

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
# GW Event specific
DEFAULT_EVENT_NAME = "GW170817"
DEFAULT_SAMPLES_PATH_PREFIX = "IMRPhenomPv2NRT_highSpin_posterior" # Path or prefix for pesummary

# GLADE Catalog specific
GLADE_URL = "https://glade.elte.hu/GLADE-2.4.txt"
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
NSIDE_SKYMAP = 128
PROB_THRESHOLD_CDF = 0.95
HOST_Z_MAX = 0.15 # Final redshift cut for candidate hosts

# Cosmological parameters for likelihood
SIGMA_V_PEC = 250.0  # km/s, peculiar velocity uncertainty
C_LIGHT = 299792.458 # km/s
OMEGA_M = 0.31

# MCMC parameters
MCMC_N_DIM = 1
MCMC_N_WALKERS = 32
MCMC_N_STEPS = 6000
MCMC_BURNIN = 1000
MCMC_THIN_BY = 10
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
# 0.  Fetch posterior samples
# -------------------------------------------------------------------
def fetch_gw_posterior_samples(event_name, path_to_samples_prefix):
    """Fetches GW posterior samples using pesummary."""
    try:
        from pesummary.gw.fetch import fetch_open_samples
    except ImportError:
        sys.exit("Install dependencies first: pip install pesummary healpy emcee astropy pandas matplotlib scipy")

    print(f"Downloading {event_name} posterior samples...")
    try:
        post = fetch_open_samples(event_name, path_to_samples=path_to_samples_prefix)
        samples_dict = post.samples_dict
        dL_samples = samples_dict["luminosity_distance"]
        ra_samples = np.rad2deg(samples_dict["ra"])
        dec_samples = np.rad2deg(samples_dict["dec"])
        return dL_samples, ra_samples, dec_samples
    except Exception as e:
        print(f"❌ Error fetching posterior samples for {event_name}: {e}")
        sys.exit(1)

# -------------------------------------------------------------------
# 1.  Download (if needed) and load GLADE v2.4
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
            glade_df[c] = pd.to_numeric(glade_df[c], errors='coerce')
        else:
            print(f"⚠️ Warning: Column {c} not found for numeric conversion.")

    initial_count = len(glade_df)
    glade_df = glade_df.dropna(subset=cols_to_dropna)
    print(f"  {len(glade_df):,} galaxies kept after dropping NaNs in {cols_to_dropna} (from {initial_count}).")

    if glade_df.empty:
        print("  No galaxies remaining after dropping NaNs. Cannot proceed.")
        return glade_df

    count_before_range_checks = len(glade_df)
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
# 2.  Build probability sky-map and select candidate hosts
# -------------------------------------------------------------------
def select_candidate_hosts(ra_gw_samples, dec_gw_samples, galaxy_df, nside, cdf_threshold, z_max_filter):
    """Selects candidate host galaxies based on GW sky localization and redshift."""
    if galaxy_df.empty:
        print("⚠️ Galaxy catalog is empty. Cannot select candidate hosts.")
        return pd.DataFrame()

    print("\nBuilding sky-map and selecting candidate hosts...")
    ipix = hp.ang2pix(nside, ra_gw_samples, dec_gw_samples, lonlat=True)
    prob_map = np.bincount(ipix, minlength=hp.nside2npix(nside))
    prob_map = prob_map / prob_map.sum()

    sorted_probs = np.sort(prob_map)[::-1]
    cdf = np.cumsum(sorted_probs)
    try:
        sky_map_threshold_val = sorted_probs[np.searchsorted(cdf, cdf_threshold)]
    except IndexError: # Happens if cdf_threshold is 1.0 and all probs are not exactly 0
        if cdf_threshold == 1.0 and len(sorted_probs) > 0:
            sky_map_threshold_val = sorted_probs[-1] # Smallest non-zero probability
        else: # Or if prob_map was empty/all zero
            print("⚠️ Could not determine sky map threshold. Using 0.")
            sky_map_threshold_val = 0


    sky_mask = prob_map >= sky_map_threshold_val

    coords = SkyCoord(galaxy_df.ra.values * u.deg, galaxy_df.dec.values * u.deg, frame='icrs')
    gpix = hp.ang2pix(nside, coords.ra.deg, coords.dec.deg, lonlat=True)

    selected_hosts = galaxy_df[sky_mask[gpix]].copy()
    print(f"  Selected {len(selected_hosts):,} galaxies within the initial {cdf_threshold*100:.0f}% sky area.")

    # Apply redshift cut
    selected_hosts = selected_hosts[selected_hosts['z'] < z_max_filter]
    print(f"  → {len(selected_hosts):,} candidate host galaxies after z < {z_max_filter} cut.")
    
    return selected_hosts

def apply_specific_galaxy_corrections(hosts_df, event_name, corrections_dict):
    """Applies specific redshift corrections for known galaxies for a given event."""
    if event_name not in corrections_dict or hosts_df.empty:
        return hosts_df

    correction_info = corrections_dict[event_name]
    pgc_id_to_correct = correction_info["PGC_ID"]
    literature_z = correction_info["LITERATURE_Z"]

    galaxy_mask = hosts_df['PGC'] == pgc_id_to_correct
    is_galaxy_present = galaxy_mask.any()

    if is_galaxy_present:
        current_z = hosts_df.loc[galaxy_mask, 'z'].iloc[0]
        print(f"\nFound galaxy PGC {pgc_id_to_correct} in candidate hosts for {event_name}.")
        print(f"  Its current redshift from GLADE is: {current_z:.5f}")
        # Apply correction if significantly different or if a policy is to always update
        if abs(current_z - literature_z) > 0.0001: # Tolerance for floating point
            print(f"  Correcting its redshift to the literature value: {literature_z:.5f}")
            hosts_df.loc[galaxy_mask, 'z'] = literature_z
        else:
            print(f"  Its current redshift {current_z:.5f} is close enough to the literature value. No correction applied.")
    else:
        print(f"\nNote: Galaxy PGC {pgc_id_to_correct} (for {event_name} correction) not found in candidate hosts.")
    return hosts_df

# -------------------------------------------------------------------
# 3.  Likelihood function  ℒ(H0)
# -------------------------------------------------------------------
def get_log_likelihood_h0(dL_gw_samples, host_galaxies_z, sigma_v, c_val, omega_m_val):
    """Returns the log likelihood function for H0."""
    
    def lum_dist(z, H0):
        cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=omega_m_val)
        return cosmo.luminosity_distance(z).value

    def log_likelihood(theta):
        H0 = theta[0]
        if H0 <= 10 or H0 >= 200: # Prior range for H0
            return -np.inf

        # Ensure host_galaxies_z is an array for broadcasting
        z_values = np.asarray(host_galaxies_z)
        if z_values.ndim == 0: # if it was a single scalar
            z_values = np.array([z_values])

        model_d = lum_dist(z_values, H0) # model_d will be array of same size as z_values
        
        # Ensure dL_gw_samples is a column vector for broadcasting with model_d
        # dL_gw_samples shape: (N_samples,) -> (N_samples, 1)
        # model_d shape: (N_hosts,)
        # pdf result shape: (N_samples, N_hosts)
        
        sigma_d_val = (model_d / c_val) * sigma_v
        sigma_d_val = np.maximum(sigma_d_val, 1e-9) # Avoid division by zero or tiny sigma

        # norm.logpdf(value, loc, scale)
        # value: dL_gw_samples[:, None] (N_samples, 1)
        # loc: model_d (N_hosts,) -> broadcasts to (1, N_hosts)
        # scale: sigma_d_val (N_hosts,) -> broadcasts to (1, N_hosts)
        pdf = norm.logpdf(dL_gw_samples[:, None], loc=model_d[None, :], scale=sigma_d_val[None, :])
        
        # Marginalize over GW samples: sum_s P(dL_s|H0, z_i) for each galaxy i
        log_sum_over_gw_samples = np.logaddexp.reduce(pdf, axis=0) # Result shape (N_hosts,)
        
        # Marginalize over galaxies: sum_i P(data | H0, z_i) by summing probabilities (log(sum(exp(L_i))))
        # This correctly implements the marginalization over discrete host galaxy possibilities.
        total_log_likelihood = np.logaddexp.reduce(log_sum_over_gw_samples) # Sum log-likelihoods for each host correctly
        
        if not np.isfinite(total_log_likelihood):
            return -np.inf # Catch any numerical issues early
            
        return total_log_likelihood

    return log_likelihood

# -------------------------------------------------------------------
# 4.  MCMC over a *single* parameter (H0)
# -------------------------------------------------------------------
def run_mcmc_h0(log_likelihood_func, n_walkers, n_dim, initial_h0_mean, initial_h0_std, n_steps, event_name):
    """Runs the MCMC sampler for H0."""
    print(f"\nRunning MCMC for {event_name} (this might take a few minutes)...")
    # Initial positions for walkers, centered around a plausible H0
    walkers0 = initial_h0_mean + initial_h0_std * np.random.randn(n_walkers, n_dim)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood_func, moves=emcee.moves.StretchMove())
    
    try:
        sampler.run_mcmc(walkers0, n_steps, progress=True)
    except ValueError as ve:
        print(f"❌ ValueError during MCMC for {event_name}: {ve}")
        sys.exit("MCMC failed.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during MCMC for {event_name}: {e}")
        sys.exit("MCMC failed.")
        
    return sampler

def process_mcmc_samples(sampler, burnin, thin_by, event_name):
    """Processes MCMC samples: applies burn-in and thinning."""
    print(f"Processing MCMC samples for {event_name} (burn-in: {burnin}, thin: {thin_by})...")
    try:
        flat_samples = sampler.get_chain(discard=burnin, thin=thin_by, flat=True)
        if flat_samples.ndim > 1 : # If n_dim > 1, it's (N_samples_after_thin, N_dim)
             flat_samples = flat_samples[:,0] # Assuming H0 is the first (and only) parameter

        if len(flat_samples) == 0:
            print(f"⚠️ MCMC for {event_name} resulted in no valid samples after burn-in and thinning.")
            sys.exit("MCMC post-processing failed: no samples.")
        return flat_samples
    except Exception as e:
        print(f"❌ Error processing MCMC chain for {event_name}: {e}")
        sys.exit("MCMC post-processing failed.")

# -------------------------------------------------------------------
# 5.  Plot and save posterior
# -------------------------------------------------------------------
def save_and_plot_h0_posterior(h0_samples, event_name, num_candidate_hosts):
    """Saves H0 samples and plots the posterior distribution."""
    output_samples_file = f"H0_samples_{event_name}.npy"
    output_plot_file = f"H0_posterior_{event_name}.pdf"

    np.save(output_samples_file, h0_samples)
    print(f"MCMC samples saved to {output_samples_file}")

    q16, q50, q84 = np.percentile(h0_samples, [16, 50, 84])
    err_minus = q50 - q16
    err_plus = q84 - q50
    print(f"\n{event_name} H0 = {q50:.1f} +{err_plus:.1f} / -{err_minus:.1f} km s⁻¹ Mpc⁻¹ (68% C.I.)")

    plt.figure(figsize=(8, 6))
    plt.hist(h0_samples, bins=50, density=True, histtype="stepfilled", alpha=0.6, label="Posterior Samples")
    plt.axvline(q50, color='k', ls='--', label=f'Median: {q50:.1f} km s⁻¹ Mpc⁻¹')
    plt.axvline(q16, color='k', ls=':', alpha=0.7)
    plt.axvline(q84, color='k', ls=':', alpha=0.7)
    plt.xlabel(r"$H_0\;[\mathrm{km\ s^{-1}\ Mpc^{-1}}]$")
    plt.ylabel("Posterior Density")
    plt.title(f"Statistical Standard Siren – {event_name} & GLADE v2.4\n({num_candidate_hosts:,} Candidate Galaxies)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_plot_file)
    print(f"Saved posterior plot: {output_plot_file}")

# -------------------------------------------------------------------
# Main script execution
# -------------------------------------------------------------------
def main(event_name=DEFAULT_EVENT_NAME, samples_path_prefix=DEFAULT_SAMPLES_PATH_PREFIX):
    """Main function to run the H0 analysis for a given GW event."""
    print(f"Starting H0 analysis for event: {event_name}")

    # 0. Fetch GW posterior samples
    dL_gw_samples, ra_gw_samples, dec_gw_samples = fetch_gw_posterior_samples(event_name, samples_path_prefix)
    if dL_gw_samples is None or ra_gw_samples is None or dec_gw_samples is None :
        print(f"Failed to get posterior samples for {event_name}. Exiting.")
        return

    # 1. Load and clean galaxy catalog
    glade_raw_df = download_and_load_galaxy_catalog(
        GLADE_URL, GLADE_FILE, GLADE_USE_COLS, GLADE_COL_NAMES, GLADE_NA_VALUES
    )
    if glade_raw_df.empty:
        print("GLADE catalog is empty after loading. Exiting.")
        return
        
    glade_cleaned_df = clean_galaxy_catalog(
        glade_raw_df,
        numeric_cols=GLADE_COL_NAMES, # Attempt to convert all loaded columns
        cols_to_dropna=['ra', 'dec', 'z'], # Essential for sky position and redshift
        range_filters=GLADE_RANGE_CHECKS
    )
    if glade_cleaned_df.empty:
        print("GLADE catalog is empty after cleaning. Exiting.")
        return

    # 2. Select candidate hosts
    candidate_hosts_df = select_candidate_hosts(
        ra_gw_samples, dec_gw_samples, glade_cleaned_df,
        NSIDE_SKYMAP, PROB_THRESHOLD_CDF, HOST_Z_MAX
    )
    
    # Apply specific galaxy corrections (e.g., NGC 4993 for GW170817)
    candidate_hosts_df = apply_specific_galaxy_corrections(
        candidate_hosts_df, event_name, GALAXY_CORRECTIONS
    )

    if candidate_hosts_df.empty:
        print(f"No candidate host galaxies found for {event_name} after selection and/or correction. MCMC cannot proceed.")
        sys.exit(f"Exiting: No candidate galaxies for {event_name}.")

    print(f"Final number of candidate hosts for {event_name}: {len(candidate_hosts_df)}")

    # 3. Define Likelihood function
    log_likelihood_h0_func = get_log_likelihood_h0(
        dL_gw_samples,
        candidate_hosts_df['z'].values, # Pass only the redshift series/array
        SIGMA_V_PEC, C_LIGHT, OMEGA_M
    )

    # 4. Run MCMC
    mcmc_sampler = run_mcmc_h0(
        log_likelihood_h0_func,
        MCMC_N_WALKERS, MCMC_N_DIM,
        MCMC_INITIAL_H0_MEAN, MCMC_INITIAL_H0_STD,
        MCMC_N_STEPS,
        event_name
    )
    
    flat_h0_samples = process_mcmc_samples(mcmc_sampler, MCMC_BURNIN, MCMC_THIN_BY, event_name)

    # 5. Save and Plot results
    save_and_plot_h0_posterior(flat_h0_samples, event_name, len(candidate_hosts_df))

    print(f"\nScript finished for {event_name}.")


if __name__ == "__main__":
    # This allows running for a specific event from command line, e.g.
    # python main.py GW190814 path_to_GW190814_samples
    # For now, it defaults to GW170817
    current_event_name = "GW200105_162426"
    current_samples_prefix = "IMRPhenomPv2NRT_highSpin_posterior"
    
    if len(sys.argv) > 1:
        current_event_name = sys.argv[1]
    if len(sys.argv) > 2:
        current_samples_prefix = sys.argv[2]
        
    main(event_name=current_event_name, samples_path_prefix=current_samples_prefix)
