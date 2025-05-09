import os
import sys
import re

try:
    from pesummary.gw.fetch import fetch_open_samples
    from astropy.utils.data import conf as astropy_conf
except ImportError:
    print("ERROR: pesummary or astropy not installed. Please install dependencies.", file=sys.stderr)
    # Depending on how you want to handle this, you could raise an ImportError
    # for the importing script to catch, or exit here.
    sys.exit("❌ Install dependencies first: pip install pesummary healpy emcee astropy pandas matplotlib scipy")

DEFAULT_CACHE_DIR_NAME = "pesummary_cache" # Relative to where the script using this module is run

def configure_astropy_cache(cache_dir_base=DEFAULT_CACHE_DIR_NAME):
    """
    Configures Astropy's cache settings and creates the directory if it doesn't exist.

    Args:
        cache_dir_base (str): The base name or path for the cache directory.
                              If relative, it's relative to the current working directory.

    Returns:
        str: The absolute path to the configured cache directory, or None if configuration failed.
    """
    abs_cache_dir = os.path.abspath(cache_dir_base)
    if not os.path.exists(abs_cache_dir):
        try:
            os.makedirs(abs_cache_dir)
            print(f"Created cache directory: {abs_cache_dir}")
        except OSError as e:
            print(f"Error creating cache directory {abs_cache_dir}: {e}", file=sys.stderr)
            return None # Indicate failure
            
    astropy_conf.cache_dir = abs_cache_dir
    print(f"Configured Astropy cache directory: {astropy_conf.cache_dir}")

    # Ensure Astropy is allowed to download files
    if hasattr(astropy_conf, 'allow_remote_data') and not astropy_conf.allow_remote_data:
        print("Setting astropy_conf.allow_remote_data = True")
        astropy_conf.allow_remote_data = True
    elif hasattr(astropy_conf, 'remote_data_strict') and astropy_conf.remote_data_strict: # Older Astropy
        print("Setting astropy_conf.remote_data_strict = False")
        astropy_conf.remote_data_strict = False
    return abs_cache_dir

def fetch_candidate_data(candidate_name, configured_cache_dir):
    """
    Fetches posterior samples for a given gravitational wave candidate.

    Handles caching and common errors like multiple posterior tables or unknown URLs.
    It's expected that configure_astropy_cache() has been called appropriately before this.

    Args:
        candidate_name (str): The name of the GW candidate (e.g., "GW170817").
        configured_cache_dir (str): The absolute path to the directory pesummary should use for 'outdir',
                                    which Astropy is also configured to use as its cache_dir.

    Returns:
        tuple: (success_flag, data_or_error_message)
               - If successful: (True, pesummary_object)
               - If failed: (False, error_message_string)
    """
    print(f"Attempting to fetch {candidate_name} posterior (using astropy cache: {configured_cache_dir}) …")
    
    # Using the download_kwargs structure as per your last working version
    common_download_kwargs = {'cache': True}
    post = None

    try:
        # pesummary's 'outdir' tells it where to place the downloaded file.
        # Astropy (used by pesummary for download) respects its own global astropy_conf.cache_dir
        # for checking if the file already exists there.
        post = fetch_open_samples(candidate_name, outdir=configured_cache_dir, download_kwargs=common_download_kwargs)
        
        # Validate post object
        if hasattr(post, 'samples_dict') and post.samples_dict:
            return True, post
        elif isinstance(post, dict) and post: # Check if post itself is a non-empty samples dict
             return True, post
        else:
            return False, f"Fetched {candidate_name}, but no sample_dict found or it is empty. Type: {type(post)}"

    except Exception as e:
        error_message = str(e)
        
        if "Found multiple posterior sample tables" in error_message:
            print(f"Caught error for {candidate_name}: Multiple posterior tables found. Attempting retry.")
            match = re.search(r"Found multiple posterior sample tables in .*?: (.*)\. Not sure which to load\.", error_message)
            if match:
                table_names_str = match.group(1)
                available_tables = [name.strip() for name in table_names_str.split(',')]
                if available_tables:
                    chosen_table = available_tables[0]
                    print(f"Retrying {candidate_name} with the first available table: '{chosen_table}'")
                    try:
                        post_retry = fetch_open_samples(
                            candidate_name,
                            outdir=configured_cache_dir,
                            path_to_samples=chosen_table,
                            download_kwargs=common_download_kwargs # Ensure cache is also used on retry
                        )
                        if hasattr(post_retry, 'samples_dict') and post_retry.samples_dict:
                            return True, post_retry
                        elif isinstance(post_retry, dict) and post_retry:
                             return True, post_retry
                        else:
                            return False, f"Retry for {candidate_name} successful, but no sample_dict. Type: {type(post_retry)}"
                    except Exception as retry_e:
                        return False, f"Retry for {candidate_name} with table '{chosen_table}' FAILED: {retry_e}"
                else:
                    return False, f"Could not parse table names from error for {candidate_name}: No tables found in string."
            else:
                return False, f"Could not parse table names from error for {candidate_name}: Regex failed."
        
        elif "Unknown URL" in error_message:
            return False, f"Unknown URL for {candidate_name}."
        
        else: # Other errors
            return False, f"Unexpected error for {candidate_name}: {error_message}"

if __name__ == "__main__":
    # This block runs if the script is executed directly (e.g., python gw_data_fetcher.py)
    # Useful for testing the module itself.
    
    print("--- Testing gw_data_fetcher module ---")
    
    # Configure cache for the test
    # Uses the DEFAULT_CACHE_DIR_NAME defined in this module
    test_cache_dir = configure_astropy_cache() 

    if not test_cache_dir:
        print("CRITICAL: Failed to configure cache directory for module test. Exiting.", file=sys.stderr)
        sys.exit(1)

    test_candidates = [
        "GW170817",  # Should trigger retry
        "GW190814",  # Should succeed directly
        "GW190412",  # Should fail (Unknown URL)
        "GW170608"   # Should succeed directly
    ]
    
    all_results = {}

    for candidate in test_candidates:
        print(f"\n--- Testing candidate: {candidate} ---")
        success_status, result = fetch_candidate_data(candidate, test_cache_dir)
        all_results[candidate] = {"success": success_status, "details": result}
        if success_status:
            print(f"Successfully fetched test candidate: {candidate}")
            # For display, show keys if available
            if hasattr(result, 'samples_dict') and result.samples_dict:
                 print(f"  Sample keys: {list(result.samples_dict.keys())}")
            elif isinstance(result, dict):
                 print(f"  Sample keys (direct dict): {list(result.keys())}")
        else:
            print(f"Failed to fetch test candidate {candidate}: {result}")
            
    print("\n--- Module Test Summary ---")
    for candidate, res_info in all_results.items():
        status_str = "SUCCESS" if res_info["success"] else "FAILED"
        details_str = ""
        if res_info["success"]:
            if hasattr(res_info["details"], 'samples_dict') and res_info["details"].samples_dict:
                details_str = f"Keys: {list(res_info['details'].samples_dict.keys())}"
            elif isinstance(res_info["details"], dict):
                details_str = f"Keys: {list(res_info['details'].keys())}"
            else:
                details_str = "Data object present"
        else:
            details_str = res_info["details"] # Error message
        print(f"  {candidate}: {status_str} - {details_str}") 