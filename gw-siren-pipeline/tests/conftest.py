import pytest
import tempfile
import shutil
import os
import yaml  # Ensure PyYAML is used for creating the mock config content
from gwsiren.config import load_config  # Assuming gwsiren.config.load_config exists


@pytest.fixture(scope="session")
def project_root_dir():
    """Returns the absolute path to the project root directory."""
    # Assumes conftest.py is in tests/ relative to the project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def temp_data_dir():
    """Creates a temporary directory for test data and yields its path.
    The directory is removed after the test using this fixture concludes.
    """
    td = tempfile.mkdtemp(prefix="gw_test_data_")
    yield td
    shutil.rmtree(td)


@pytest.fixture
def mock_config(monkeypatch, tmp_path, project_root_dir):
    """
    Mocks the gwsiren.CONFIG object with a temporary config file.
    The temporary config's catalog.data_dir points to a temporary directory.
    It also patches CONFIG where it's imported directly in key modules.
    """
    # Define the content for the dummy/mock configuration YAML
    # Ensure catalog.data_dir uses a unique temporary path for this test session
    mock_catalog_data_dir = tmp_path / "test_data_catalogs"
    os.makedirs(mock_catalog_data_dir, exist_ok=True)  # Ensure this directory exists

    dummy_config_content = {
        "catalog": {
            "glade_plus_url": "http://fake.gladeplus.url/GLADE+.txt",
            "glade24_url": "http://fake.glade24.url/GLADE_2.4.txt",
            "data_dir": str(mock_catalog_data_dir),  # Use the temporary path
        },
        "skymap": {"default_nside": 16, "credible_level": 0.9},
        "mcmc": {
            "walkers": 8,
            "steps": 50,
            "burnin": 10,
            "thin_by": 1,
            "prior_h0_min": 20.0,
            "prior_h0_max": 150.0,
        },
        "cosmology": {
            "sigma_v_pec": 200.0,
            "c_light": 299792.458,
            "omega_m": 0.3,
        },
        "fetcher": {
            "cache_dir_name": "test_cache",
            "timeout": 42,
            "max_retries": 2,
        },
    }
    config_file_path = tmp_path / "test_mock_config.yaml"
    with open(config_file_path, 'w') as f:
        yaml.dump(dummy_config_content, f)

    # Load the mock configuration using the project's load_config function
    # This assumes gwsiren.config.load_config can take a path argument
    new_mocked_config = load_config(config_file_path)

    # Patch the CONFIG object in gwsiren.config (where it's defined)
    monkeypatch.setattr("gwsiren.config.CONFIG", new_mocked_config)

    # Also patch CONFIG where it might be imported directly by modules at their top level.
    # Add other modules here if they also do `from gwsiren import CONFIG`
    modules_to_patch_config = [
        "gwsiren.gw_data_fetcher",
        "gwsiren.data.catalogs",
        "sky_analyzer",
        "h0_mcmc_analyzer",
        "new",
        "viz",
        "main",
        "plot_utils",
        "analyze_candidates",
    ]
    for module_name in modules_to_patch_config:
        try:
            module = __import__(module_name)
            if hasattr(module, "CONFIG"):
                monkeypatch.setattr(f"{module_name}.CONFIG", new_mocked_config)
        except ImportError:
            # Module might not exist or doesn't import CONFIG, safe to ignore
            pass
        except AttributeError:
            # Module exists but doesn't have a CONFIG attribute at its top level.
            pass

    return new_mocked_config
