import numpy as np

def extract_gw_event_parameters(data_object, event_name):
    """
    Extracts luminosity_distance, RA, and Dec samples from a PESummary data object or dictionary.

    Args:
        data_object (object or dict): The data object returned by gw_data_fetcher.fetch_candidate_data,
                                      which could be a PESummary result object or a dictionary of samples.
        event_name (str): Name of the GW event (for logging purposes).

    Returns:
        tuple: (dL_samples, ra_samples, dec_samples) 
               Returns (None, None, None) if extraction fails or essential keys are missing.
    """
    print(f"Extracting dL, RA, Dec for {event_name} from provided data object...")
    try:
        samples_dict_top_level = None
        if hasattr(data_object, 'samples_dict') and data_object.samples_dict:
            # Standard PESummary result object case
            samples_dict_top_level = data_object.samples_dict
            print(f"  Extracted samples_dict from PESummary object attributes for {event_name}.")
        elif isinstance(data_object, dict):
            # Case where data_object is already a dictionary (e.g., from direct cache load or simple dict structure)
            samples_dict_top_level = data_object
            print(f"  Using provided data object directly as samples_dict for {event_name}.")
        else:
            print(f"❌ Data for {event_name} is not a recognized PESummary object or samples dictionary. Type: {type(data_object)}")
            return None, None, None

        actual_samples = None
        # Check if the top-level dictionary directly contains the target keys
        if all(k in samples_dict_top_level for k in ["luminosity_distance", "ra", "dec"]):
            actual_samples = samples_dict_top_level
            print(f"  Found target keys directly in top-level samples_dict for {event_name}.")
        elif samples_dict_top_level: 
            # Common case: samples_dict_top_level is a dict of dicts (multiple analyses)
            # We need to pick one. Let's try to be robust.
            # Prefer analyses like 'C01:IMRPhenomXPHM' or common ones.
            preferred_keys = [key for key in samples_dict_top_level if isinstance(samples_dict_top_level[key], dict) and all(k in samples_dict_top_level[key] for k in ["luminosity_distance", "ra", "dec"])]
            
            if preferred_keys:
                # Simple heuristic: choose the first one that looks complete.
                # More sophisticated selection might be needed if many such keys exist.
                chosen_key = preferred_keys[0]
                actual_samples = samples_dict_top_level[chosen_key]
                print(f"  Multiple analyses found in samples_dict for {event_name}. Using: '{chosen_key}'.")
            elif samples_dict_top_level and isinstance(next(iter(samples_dict_top_level.values()), None), dict):
                # Fallback: if no preferred keys, take the first entry if it's a dictionary
                first_key = next(iter(samples_dict_top_level))
                actual_samples = samples_dict_top_level[first_key]
                print(f"  Multiple analyses found in samples_dict for {event_name}. Using first one (fallback): '{first_key}'.")
            else:
                print(f"  Could not identify a suitable sub-dictionary with all target keys in samples_dict for {event_name}.")
                print(f"  Available top-level keys: {list(samples_dict_top_level.keys())}") # Log available keys for diagnosis
                return None, None, None
        else:
            print(f"❌ Samples dictionary for {event_name} is empty or None.")
            return None, None, None
        
        if actual_samples and all(k in actual_samples for k in ["luminosity_distance", "ra", "dec"]):
            dL_samples = actual_samples["luminosity_distance"]
            # Samples are usually in radians, convert to degrees for consistency with sky maps
            ra_samples = np.rad2deg(actual_samples["ra"])
            dec_samples = np.rad2deg(actual_samples["dec"])
            print(f"  Successfully extracted and processed {len(dL_samples)} dL, RA (deg), Dec (deg) samples for {event_name}.")
            return dL_samples, ra_samples, dec_samples
        else:
            print(f"❌ Error: 'luminosity_distance', 'ra', or 'dec' keys not found in the determined actual_samples for {event_name}.")
            if actual_samples:
                print(f"  Available keys in actual_samples: {list(actual_samples.keys())}")
            else:
                print("  actual_samples was None.")
            return None, None, None
            
    except KeyError as e:
        print(f"❌ Key Error during extraction for {event_name}: Key {e} not found.")
        # Consider logging the structure of actual_samples if it exists
        try:
            if 'actual_samples' in locals() and actual_samples is not None:
                 print(f"    Available keys in actual_samples where error occurred: {list(actual_samples.keys())}")
            elif 'samples_dict_top_level' in locals() and samples_dict_top_level is not None:
                 print(f"    Available keys in samples_dict_top_level where error occurred: {list(samples_dict_top_level.keys())}")
        except Exception as log_e:
            print(f"    (Additionally, error logging keys: {log_e})")
        return None, None, None
    except Exception as e:
        print(f"❌ An unexpected error occurred extracting samples for {event_name}: {e}")
        return None, None, None

if __name__ == '__main__':
    print("--- Testing event_data_extractor.py ---")

    # Mock data for testing
    mock_event_name = "GW_TEST"

    # Test Case 1: Ideal PESummary object structure
    print("\nTest Case 1: Ideal PESummary object")
    class MockPESummaryObject:
        def __init__(self):
            self.samples_dict = {
                "C01:IMRPhenomXPHM": {
                    "luminosity_distance": np.array([100., 110., 120.]),
                    "ra": np.array([np.pi/4, np.pi/3, np.pi/2]), # radians
                    "dec": np.array([-np.pi/6, -np.pi/4, -np.pi/3]), # radians
                    "other_param": np.array([1, 2, 3])
                }
            }
    mock_obj = MockPESummaryObject()
    dL, ra, dec = extract_gw_event_parameters(mock_obj, mock_event_name + "_Case1")
    if dL is not None and ra is not None and dec is not None:
        print(f"  Extracted Case 1: dL_mean={np.mean(dL):.1f}, RA_mean={np.mean(ra):.1f} deg, Dec_mean={np.mean(dec):.1f} deg")
        assert len(dL) == 3
        assert abs(np.mean(ra) - (np.degrees(np.pi/4) + np.degrees(np.pi/3) + np.degrees(np.pi/2))/3) < 1e-6
    else:
        print("  Case 1 FAILED")

    # Test Case 2: Direct dictionary input (already selected analysis)
    print("\nTest Case 2: Direct dictionary input")
    mock_dict_direct = {
        "luminosity_distance": np.array([200., 210.]),
        "ra": np.array([np.pi, 1.5*np.pi]), # radians
        "dec": np.array([0, -np.pi/4]),      # radians
        "mass_1": np.array([30,32])
    }
    dL, ra, dec = extract_gw_event_parameters(mock_dict_direct, mock_event_name + "_Case2")
    if dL is not None and ra is not None and dec is not None:
        print(f"  Extracted Case 2: dL_mean={np.mean(dL):.1f}, RA_mean={np.mean(ra):.1f} deg, Dec_mean={np.mean(dec):.1f} deg")
        assert len(dL) == 2
        assert abs(np.mean(ra) - (np.degrees(np.pi) + np.degrees(1.5*np.pi))/2) < 1e-6
    else:
        print("  Case 2 FAILED")

    # Test Case 3: Dictionary of dictionaries (multiple analyses)
    print("\nTest Case 3: Dictionary of dictionaries")
    mock_dict_multiple = {
        "analysis_A": {"param_x": np.array([1,2])},
        "analysis_B_complete": {
            "luminosity_distance": np.array([300., 310., 320., 330.]),
            "ra": np.array([0.1, 0.2, 0.3, 0.4]), # radians
            "dec": np.array([-0.1, -0.2, -0.3, -0.4]), # radians
        }
    }
    dL, ra, dec = extract_gw_event_parameters(mock_dict_multiple, mock_event_name + "_Case3")
    if dL is not None and ra is not None and dec is not None:
        print(f"  Extracted Case 3: dL_mean={np.mean(dL):.1f}, RA_mean={np.mean(ra):.1f} deg, Dec_mean={np.mean(dec):.1f} deg")
        assert len(dL) == 4
    else:
        print("  Case 3 FAILED")

    # Test Case 4: Missing essential key
    print("\nTest Case 4: Missing essential key")
    mock_dict_missing_key = {
        "C01:MixedAnalysis": {
            "luminosity_distance": np.array([400., 410.]),
            # "ra" is missing
            "dec": np.array([np.pi/8, np.pi/7])
        }
    }
    dL, ra, dec = extract_gw_event_parameters(mock_dict_missing_key, mock_event_name + "_Case4")
    if dL is None and ra is None and dec is None:
        print("  Case 4 correctly returned None for missing 'ra'.")
    else:
        print("  Case 4 FAILED - should have returned None.")

    # Test Case 5: Empty data object
    print("\nTest Case 5: Empty dictionary")
    empty_dict = {}
    dL, ra, dec = extract_gw_event_parameters(empty_dict, mock_event_name + "_Case5")
    if dL is None and ra is None and dec is None:
        print("  Case 5 correctly returned None for empty data.")
    else:
        print("  Case 5 FAILED - should have returned None.")

    # Test Case 6: Invalid data type
    print("\nTest Case 6: Invalid data type")
    invalid_data = "this_is_a_string"
    dL, ra, dec = extract_gw_event_parameters(invalid_data, mock_event_name + "_Case6")
    if dL is None and ra is None and dec is None:
        print("  Case 6 correctly returned None for invalid data type.")
    else:
        print("  Case 6 FAILED - should have returned None.")
        
    # Test Case 7: samples_dict is present but empty
    print("\nTest Case 7: PESummary object with empty samples_dict")
    class MockPESummaryObjectEmptyDict:
        def __init__(self):
            self.samples_dict = {}
    mock_obj_empty_dict = MockPESummaryObjectEmptyDict()
    dL, ra, dec = extract_gw_event_parameters(mock_obj_empty_dict, mock_event_name + "_Case7")
    if dL is None and ra is None and dec is None:
        print(f"  Case 7 correctly returned None due to empty samples_dict.")
    else:
        print("  Case 7 FAILED")

    # Test Case 8: samples_dict has sub-dict, but sub-dict is missing keys
    print("\nTest Case 8: PESummary object with sub-dict missing keys")
    class MockPESummaryObjectSubDictMissing:
        def __init__(self):
            self.samples_dict = {"analysis_X": {"param_a": np.array([1,2])}}
    mock_obj_sub_missing = MockPESummaryObjectSubDictMissing()
    dL, ra, dec = extract_gw_event_parameters(mock_obj_sub_missing, mock_event_name + "_Case8")
    if dL is None and ra is None and dec is None:
        print(f"  Case 8 correctly returned None due to sub-dict missing essential keys.")
    else:
        print("  Case 8 FAILED")

    print("\n--- Finished testing event_data_extractor.py ---") 