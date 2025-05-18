import sys
from gw_data_fetcher import fetch_candidate_data, configure_astropy_cache, DEFAULT_CACHE_DIR_NAME

# Candidate list can be defined here or passed around
candidate_list = [
    "GW170817", "GW190814", "GW190425", "GW170814", "GW170608",
    "GW200105_162426", "GW200115_042309", "GW190412",
    "GW190924_021846", "GW200202_154313"
]

successful_candidates_data = {}
failed_candidates_reasons = {}

def main():
    # Configure cache once at the beginning of your script/application
    # The `DEFAULT_CACHE_DIR_NAME` is imported from your new module.
    # You can also specify a custom path here, e.g., configure_astropy_cache("./my_custom_cache")
    current_cache_dir = configure_astropy_cache(DEFAULT_CACHE_DIR_NAME) 
    
    if not current_cache_dir:
        print("CRITICAL: Failed to configure cache directory. Exiting.", file=sys.stderr)
        sys.exit(1)

    for candidate_name in candidate_list:
        print(f"\n--- Processing {candidate_name} (using gw_data_fetcher) ---")
        
        # Use the function from the module
        success, data_or_error = fetch_candidate_data(candidate_name, current_cache_dir)
        
        if success:
            print(f"Successfully processed {candidate_name}.")
            # Store the actual data object (post)
            successful_candidates_data[candidate_name] = data_or_error 
            
            # You can now work with data_or_error as the pesummary object
            # For example, to print available sample keys:
            if hasattr(data_or_error, 'samples_dict') and data_or_error.samples_dict:
                 print(f"  Available sample keys: {list(data_or_error.samples_dict.keys())}")
            elif isinstance(data_or_error, dict): # Handles cases where data_or_error might be a SamplesDict itself
                 print(f"  Available sample keys (direct dict): {list(data_or_error.keys())}")
        else:
            print(f"Failed to process {candidate_name}: {data_or_error}")
            failed_candidates_reasons[candidate_name] = data_or_error

    print("\n--- Main Script Processing Summary ---")
    print(f"Successfully processed {len(successful_candidates_data)} candidates.")
    # Example: Accessing data for a successfully processed candidate
    # for name, data_obj in successful_candidates_data.items():
    #     print(f"Data for {name} can be accessed. Type: {type(data_obj)}")

    if failed_candidates_reasons:
        print(f"\nFailed to process {len(failed_candidates_reasons)} candidates:")
        for name, reason in failed_candidates_reasons.items():
            print(f"  {name}: {reason}")
    else:
        print("\nAll candidates processed successfully by the main script!")

    print("\n--- All candidates processed by main_script.py ---")

if __name__ == "__main__":
    main()