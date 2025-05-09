import os
import sys
import re
try:
    from pesummary.gw.fetch import fetch_open_samples
    from astropy.utils.data import conf as astropy_conf
except ImportError:
    sys.exit("❌  Install dependencies first:  pip install pesummary healpy emcee astropy pandas matplotlib scipy")

# Define a cache directory
CACHE_DIR = "./pesummary_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    print(f"Created cache directory: {CACHE_DIR}")

# Configure Astropy to use this custom cache directory
# Make sure the path is absolute for Astropy's configuration
ABS_CACHE_DIR = os.path.abspath(CACHE_DIR)
astropy_conf.cache_dir = ABS_CACHE_DIR
print(f"Configured Astropy cache directory: {astropy_conf.cache_dir}")

# Ensure Astropy is allowed to download files (it should be by default, but good to be explicit)
# For older Astropy versions:
if hasattr(astropy_conf, 'remote_data_strict') and astropy_conf.remote_data_strict:
    print("Setting astropy_conf.remote_data_strict = False")
    astropy_conf.remote_data_strict = False
# For newer Astropy versions (Astropy 3.0+ changed this):
if hasattr(astropy_conf, 'allow_remote_data') and not astropy_conf.allow_remote_data:
    print("Setting astropy_conf.allow_remote_data = True")
    astropy_conf.allow_remote_data = True

candidate_list = [
    "GW170817",  # The original BNS with EM counterpart (your reference)
    "GW190814",  # Merger with a mass-gap object, excellent dark siren localization
    "GW190425",  # Second BNS detected, no EM counterpart, good for dark BNS siren
    "GW170814",  # BBH, first 3-detector observation, well-localized for its time
    "GW170608",  # Relatively nearby and low-mass BBH, good dark siren candidate
    "GW200105_162426", # NSBH merger from O3
    "GW200115_042309", # Another NSBH merger from O3
    "GW190412",  # BBH with asymmetric masses, well-localized in O3
    "GW190924_021846", # BBH from O3b used in H0 dark siren analyses
    "GW200202_154313"  # Another BBH from O3b used in H0 dark siren analyses
]

successful_candidates = {}
failed_candidates = {}

# Define common download keyword arguments for pesummary
# This tells pesummary to use Astropy's caching mechanism
common_download_kwargs = {'cache': True}

for candidate_name in candidate_list:
    print(f"\n--- Processing {candidate_name} ---")
    print(f"Attempting to fetch {candidate_name} posterior (astropy cache: {ABS_CACHE_DIR}) …")
    post = None
    try:
        post = fetch_open_samples(candidate_name, outdir=ABS_CACHE_DIR, download_kwargs=common_download_kwargs)
        print(f"Successfully fetched {candidate_name} on first attempt.")
        if hasattr(post, 'samples_dict') and post.samples_dict:
            successful_candidates[candidate_name] = list(post.samples_dict.keys())
            print(f"Available sample keys: {successful_candidates[candidate_name]}")
        elif isinstance(post, dict):
            successful_candidates[candidate_name] = list(post.keys())
            print(f"Available sample keys (direct dict): {successful_candidates[candidate_name]}")
        else:
            print(f"Fetched {candidate_name}, but no sample_dict found or it is empty. Type: {type(post)}")
            failed_candidates[candidate_name] = f"No sample_dict or empty (Type: {type(post)})"

    except Exception as e:  # Catch generic Exception first
        error_message = str(e)
        handled_specific_error = False

        if "Found multiple posterior sample tables" in error_message:
            handled_specific_error = True
            print(f"Caught error for {candidate_name}: Multiple posterior tables found within the data file.")
            match = re.search(r"Found multiple posterior sample tables in .*?: (.*)\. Not sure which to load\.", error_message)
            if match:
                table_names_str = match.group(1)
                available_tables = [name.strip() for name in table_names_str.split(',')]
                if available_tables:
                    chosen_table = available_tables[0]
                    print(f"Retrying with the first available table: '{chosen_table}'")
                    try:
                        post_retry = fetch_open_samples(
                            candidate_name,
                            outdir=ABS_CACHE_DIR,
                            path_to_samples=chosen_table,
                            cache=True # Ensure cache is also used on retry
                        )
                        print(f"Successfully fetched {candidate_name} on retry with table '{chosen_table}'.")
                        if hasattr(post_retry, 'samples_dict') and post_retry.samples_dict:
                            successful_candidates[candidate_name] = list(post_retry.samples_dict.keys())
                            print(f"Available sample keys: {successful_candidates[candidate_name]}")
                        elif isinstance(post_retry, dict):
                            successful_candidates[candidate_name] = list(post_retry.keys())
                            print(f"Available sample keys (direct dict): {successful_candidates[candidate_name]}")
                        else:
                            print(f"Fetched {candidate_name} on retry, but no sample_dict. Type: {type(post_retry)}")
                            failed_candidates[candidate_name] = f"Retry success, but no sample_dict (Type: {type(post_retry)})"
                    except Exception as retry_e:
                        print(f"ERROR for {candidate_name} on retry with table '{chosen_table}': {retry_e}")
                        failed_candidates[candidate_name] = f"Retry failed for table '{chosen_table}': {retry_e}"
                else:
                    print(f"Could not parse table names from error message for {candidate_name}.")
                    failed_candidates[candidate_name] = "Failed to parse multiple tables error (no tables found in string)"
            else:
                print(f"Could not parse table names from error message format for {candidate_name}.")
                failed_candidates[candidate_name] = "Regex failed for multiple tables error"
        
        elif "Unknown URL" in error_message: # Check this specific known error string
            handled_specific_error = True
            print(f"ERROR for {candidate_name}: Unknown URL. Data cannot be fetched automatically.")
            failed_candidates[candidate_name] = "Unknown URL"
        
        if not handled_specific_error: # If the error was not one of the above string matches
            if isinstance(e, ValueError): # Check if it was originally a ValueError
                 print(f"A ValueError (not matching known patterns) occurred for {candidate_name}: {e}")
                 failed_candidates[candidate_name] = f"ValueError: {e}"
            else: # Otherwise, it's some other unexpected exception
                 print(f"An UNEXPECTED error (not matching known patterns) occurred for {candidate_name}: {e}")
                 failed_candidates[candidate_name] = f"Unexpected error: {e}"

print("\n--- Processing Summary ---")
print(f"Successfully processed {len(successful_candidates)} candidates:")
for name, keys in successful_candidates.items():
    print(f"  {name}: {keys}")

if failed_candidates:
    print(f"\nFailed to process {len(failed_candidates)} candidates:")
    for name, reason in failed_candidates.items():
        print(f"  {name}: {reason}")
else:
    print("\nAll candidates processed successfully!")

print("\n--- All candidates processed ---")