import gwcosmo
from gwcosmo.utilities.cosmology import standard_cosmology # Changed import
from astropy import units as u

def main():
    """
    Tests the gwcosmo installation by importing the package,
    printing its version, and performing a simple cosmological calculation.
    """
    print("Attempting to test gwcosmo installation...")

    try:
        print(f"Successfully imported gwcosmo version: {gwcosmo.__version__}")

        # Instantiate the standard_cosmology class
        # We can use default parameters for H0 and Omega_m for this test
        cosmo = standard_cosmology(H0=70, Omega_m=0.3)
        print(f"Successfully instantiated standard_cosmology: {cosmo}")

        # Perform a simple calculation (luminosity distance)
        z = 0.1  # Redshift
        
        # Call the dl_zH0 method
        # This method directly returns the luminosity distance in Mpc
        dL_mpc = cosmo.dl_zH0(z)
        print(f"Luminosity distance at z={z}: {dL_mpc:.2f} Mpc")

        print("\nTest PASSED: gwcosmo seems to be installed and working correctly.")

    except ImportError as e:
        print(f"Test FAILED: Could not import gwcosmo or its components. Error: {e}")
        print("Please ensure gwcosmo and its dependencies are installed correctly.")
    except AttributeError as e:
        print(f"Test FAILED: Attribute error. This might indicate an issue with the gwcosmo installation or an API change. Error: {e}")
    except Exception as e:
        print(f"Test FAILED: An unexpected error occurred. Error: {e}")

if __name__ == "__main__":
    main()
