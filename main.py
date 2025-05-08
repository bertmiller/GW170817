#!/usr/bin/env python
"""
Reproduce Fishbach et al. (2019) statistical-standard-siren analysis
for GW170817 using the GLADE v2.4 galaxy catalogue.

Outputs:
  - 'H0_samples.npy'      -> flat MCMC samples
  - 'H0_posterior.pdf'    -> histogram plot
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
# 0.  Fetch posterior samples with pesummary
# -------------------------------------------------------------------
try:
    from pesummary.gw.fetch import fetch_open_samples
except ImportError:
    sys.exit("❌  Install dependencies first:  pip install pesummary healpy emcee astropy pandas matplotlib scipy")

print("Downloading GW170817 posterior …")
post = fetch_open_samples("GW170817", path_to_samples="IMRPhenomPv2NRT_highSpin_posterior")
samples = post.samples_dict
dL_samples = samples["luminosity_distance"]
ra_samples = np.rad2deg(samples["ra"])
dec_samples= np.rad2deg(samples["dec"])

# -------------------------------------------------------------------
# 1.  Download (if needed) and load GLADE v2.4
# -------------------------------------------------------------------
GLADE_URL = "https://glade.elte.hu/GLADE-2.4.txt"
GLADE_FILE = "GLADE_2.4.txt"

if not os.path.exists(GLADE_FILE):
    print(f"Downloading GLADE catalogue ({GLADE_URL}, ~450 MB) …")
    urllib.request.urlretrieve(GLADE_URL, GLADE_FILE)

print("Reading GLADE …")

# Corrected column indices based on diagnostic output:
# PGC: pandas field 0
# RA:  pandas field 6
# Dec: pandas field 7
# z:   pandas field 15
use_glade_cols = [0, 6, 7, 15] # These are the 0-indexed fields pandas will parse
my_col_names = ['PGC', 'ra', 'dec', 'z'] # Assign names to these extracted fields

custom_na_values = ['-99.0', '-999.0', '-9999.0', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'N/A', 'n/a', 'None', '...', 'no_value']

try:
    glade = pd.read_csv(
        GLADE_FILE,
        sep=r"\s+",
        usecols=use_glade_cols,
        names=my_col_names,
        comment='#',
        low_memory=False,
        na_values=custom_na_values,
    )
except Exception as e:
    print(f"Error reading GLADE catalog: {e}")
    sys.exit(1)

print(f"\n  {len(glade):,} total rows read from GLADE specified columns.")
print("Data types as read by pandas (before any to_numeric):")
print(glade.dtypes) # Expect PGC, ra, dec, z to be potentially objects or floats

# --- Inspect raw values (optional, keep for debugging if needed) ---
# print("\nInspecting raw values from specified columns (first 10 unique problematic strings):")
# for col_name in my_col_names:
#     if col_name in glade:
#         print(f"\n--- Examining column: '{col_name}' (dtype as read: {glade[col_name].dtype}) ---")
#         original_not_nan = glade[col_name].notna()
#         coerce_would_produce_nan = pd.to_numeric(glade[col_name].astype(str), errors='coerce').isna()
#         problematic_mask = original_not_nan & coerce_would_produce_nan
#         if problematic_mask.any():
#             problematic_values = glade[col_name][problematic_mask].astype(str).unique()
#             print(f"  Unique problematic (non-convertible to numeric) strings found in '{col_name}':")
#             print(f"    {problematic_values[:10].tolist()} ...")
#         else:
#             print(f"  No problematic (non-convertible to numeric) strings identified in '{col_name}'.")
#         print(f"  First 10 unique raw string values seen in '{col_name}':")
#         print(f"    {glade[col_name].astype(str).unique()[:10].tolist()} ...")

# --- Clean GLADE numeric columns ---------------------------------
numeric_cols_to_convert = ['PGC','ra', 'dec', 'z'] # PGC is often numeric, can coerce
for c in numeric_cols_to_convert:
    if c in glade:
        glade[c] = pd.to_numeric(glade[c], errors='coerce')
    else:
        print(f"Warning: Column {c} not found for numeric conversion.")

initial_galaxy_count_after_load = len(glade)
glade = glade.dropna(subset=['ra', 'dec', 'z']) # PGC can be NaN if not essential for selection
print(f"\n  {len(glade):,} galaxies kept after dropping rows with NaNs in RA, Dec, or z.")
print(f"    (Started with {initial_galaxy_count_after_load} rows before this dropna step)")

if not glade.empty:
    glade_before_range_checks = len(glade)
    glade = glade[
        (glade['dec'] >= -90) & (glade['dec'] <= 90) &
        (glade['ra'] >= 0) & (glade['ra'] < 360) &
        (glade['z'] > 0) & (glade['z'] < 2.0)
    ]
    print(f"  {len(glade):,} clean galaxies kept after range checks (RA, Dec, 0 < z < 2.0).")
    print(f"    (Started with {glade_before_range_checks} before this range check)")
    if not glade.empty:
        print("  Sample of cleaned galaxy data (head):")
        print(glade.head())
    else:
        print("  No galaxies remaining after range checks.")
else:
    print("  No galaxies remaining after attempting to drop NaNs (RA, Dec, z).")

if glade.empty:
    print("\nNo valid galaxies found after cleaning. Further analysis will likely fail.")
    sys.exit("Exiting due to no valid galaxies found.")


# -------------------------------------------------------------------
# 2.  Build a probability sky-map and 3-D mask
# -------------------------------------------------------------------
print("\nBuilding sky-map and selecting candidate hosts...")
nside = 128
ipix  = hp.ang2pix(nside, ra_samples, dec_samples, lonlat=True)
prob  = np.bincount(ipix, minlength=hp.nside2npix(nside))
prob  = prob / prob.sum()

cdf   = np.cumsum(np.sort(prob)[::-1])
thr   = np.sort(prob)[::-1][np.searchsorted(cdf, 0.99)]
sky_mask = prob >= thr

coords = SkyCoord(glade.ra.values*u.deg, glade.dec.values*u.deg, frame='icrs')
gpix   = hp.ang2pix(nside, coords.ra.deg, coords.dec.deg, lonlat=True)

hosts  = glade[sky_mask[gpix]].copy()
print(f"  Selected {len(hosts):,} galaxies within the initial 99% sky area.")

# --- Correction block for NGC 4993's redshift ---
if not hosts.empty:
    NGC4993_GLADE_PGC = 45657.0  # The PGC ID for NGC 4993 as it appears in your GLADE output
    NGC4993_literature_z = 0.009783  # Accurate heliocentric redshift for NGC 4993

    # Find NGC 4993 in the hosts DataFrame
    ngc4993_mask = hosts['PGC'] == NGC4993_GLADE_PGC
    is_ngc4993_present = ngc4993_mask.any()

    if is_ngc4993_present:
        current_z = hosts.loc[ngc4993_mask, 'z'].iloc[0]
        print(f"\nFound NGC 4993 (PGC {NGC4993_GLADE_PGC}) in candidate hosts.")
        print(f"  Its current redshift from GLADE is: {current_z:.5f}")
        if abs(current_z - NGC4993_literature_z) > 0.001: # If significantly different
            print(f"  Correcting its redshift to the literature value: {NGC4993_literature_z:.5f}")
            hosts.loc[ngc4993_mask, 'z'] = NGC4993_literature_z
        else:
            print(f"  Its current redshift {current_z:.5f} is close to the literature value. No correction applied.")
    else:
        # This case should not happen given your output, but good for robustness
        print(f"\nWARNING: NGC 4993 (PGC {NGC4993_GLADE_PGC}) was NOT found in the final candidate list.")
# --- End of correction block ---
hosts  = hosts[hosts['z'] < 0.15]
print(f"  → {len(hosts):,} candidate host galaxies inside 99% sky area and z < 0.15.")

if hosts.empty:
    print("\nNo candidate host galaxies found. MCMC cannot proceed.")
    sys.exit("Exiting: No candidate galaxies found in the GW localization or redshift range.")

# -------------------------------------------------------------------
# 3.  Likelihood function  ℒ(H0)
# -------------------------------------------------------------------
sigma_v = 250.0
c       = 299792.458
Omega_m = 0.31

def lum_dist(z, H0):
    cosmo = FlatLambdaCDM(H0=H0*u.km/u.s/u.Mpc, Om0=Omega_m)
    return cosmo.luminosity_distance(z).value

def log_likelihood(theta):
    H0 = theta[0]
    if H0 <= 10 or H0 >= 200:
        return -np.inf

    model_d = lum_dist(hosts.z.values, H0)
    sigma_d = (model_d / c) * sigma_v
    sigma_d = np.maximum(sigma_d, 1e-9)

    pdf = norm.logpdf(dL_samples[:,None], loc=model_d, scale=sigma_d)
    log_sum_samples = np.logaddexp.reduce(pdf, axis=0)
    total_log_likelihood = np.logaddexp.reduce(log_sum_samples)
    return total_log_likelihood

# -------------------------------------------------------------------
# 4.  MCMC over a *single* parameter (H0)
# -------------------------------------------------------------------
ndim, nwalk = 1, 32
walkers0 = 70 + 10*np.random.randn(nwalk, ndim)

sampler  = emcee.EnsembleSampler(nwalk, ndim, log_likelihood, moves=emcee.moves.StretchMove())
print("\nRunning MCMC … (this might take a few minutes on a laptop)")
nsteps = 6000
try:
    sampler.run_mcmc(walkers0, nsteps, progress=True)
except ValueError as ve:
    print(f"ValueError during MCMC: {ve}")
    sys.exit("MCMC failed.")

burnin = 1000
thin_by = 10
try:
    flat_samples = sampler.get_chain(discard=burnin, thin=thin_by, flat=True)[:,0]
    if len(flat_samples) == 0:
        print("MCMC resulted in no valid samples after burn-in and thinning.")
        sys.exit("MCMC post-processing failed.")
except Exception as e:
    print(f"Error processing MCMC chain: {e}")
    sys.exit("MCMC post-processing failed.")

np.save("H0_samples.npy", flat_samples)
print(f"MCMC complete. Samples saved to H0_samples.npy (after {burnin} burn-in, {thin_by} thinning).")

q16, q50, q84 = np.percentile(flat_samples,[16,50,84])
err_minus = q50 - q16
err_plus = q84 - q50
print(f"\nH0 = {q50:.1f} +{err_plus:.1f} / -{err_minus:.1f}  km s⁻¹ Mpc⁻¹ (68% credible interval)")

# -------------------------------------------------------------------
# 5.  Plot and save posterior
# -------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.hist(flat_samples, bins=50, density=True, histtype="stepfilled", alpha=0.6, label="Posterior Samples")
plt.axvline(q50, color='k', ls='--', label=f'Median: {q50:.1f} km s⁻¹ Mpc⁻¹')
plt.axvline(q16, color='k', ls=':', alpha=0.7)
plt.axvline(q84, color='k', ls=':', alpha=0.7)
plt.xlabel(r"$H_0\;[\mathrm{km\ s^{-1}\ Mpc^{-1}}]$")
plt.ylabel("Posterior Density")
plt.title(f"Statistical Standard Siren – GW170817 & GLADE v2.4\n({len(hosts):,} Candidate Galaxies)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("H0_posterior.pdf")
print("Saved posterior plot: H0_posterior.pdf")

print("\nScript finished.")
