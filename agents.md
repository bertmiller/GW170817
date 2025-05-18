# Agent Instructions for the GW170817 Analysis Codebase

Welcome, AI Assistant! This document provides guidelines for working on the Gravitational Wave (GW) event analysis codebase. Your adherence to these guidelines will help maintain code quality, consistency, and testability.

## 1. Project Overview

This project focuses on analyzing gravitational wave event data, primarily for cosmological parameter estimation (like H0) using statistical standard siren methods. Key functionalities include:
- Fetching GW event posterior samples.
- Handling and processing galaxy catalogs (e.g., GLADE+).
- Performing sky localization and selecting candidate host galaxies.
- Running MCMC simulations for parameter estimation.
- Visualizing data and results.

The codebase is modular, with core logic in the `gwsiren/` package and various scripts for analysis and visualization.

The goal of this project is to create scalable pipelines for analyzing GW events, and to get a good estimate of H0.

## 2. Codebase Structure

- **`gwsiren/`**: Core package containing all reusable modules.
  - `__init__.py`: Package initializer.
  - `config.py`: Handles loading project-wide configurations from `config.yaml`.
  - `gw_data_fetcher.py`: Fetches GW event data.
  - `event_data_extractor.py`: Extracts data from GW event objects.
  - `sky_analyzer.py`: Handles sky localization and galaxy selection logic.
  - `h0_mcmc_analyzer.py`: Implements H0 likelihood and MCMC logic.
  - `plot_utils.py`: Utility functions for plotting.
  - `data/`: Subpackage for data handling.
    - `catalogs.py`: Manages galaxy catalog operations.
    - `__init__.py`: Subpackage initializer.

- **`scripts/`**: Contains executable scripts for analysis.
  - `h0_e2e_pipeline.py`: Main end-to-end pipeline for H0 analysis.
  - `viz.py`: Comprehensive visualization script.
  - `analyze_candidates.py`: Script for analyzing multiple GW candidates (no MCMC).

- **`tests/`**: Contains all tests for the project.
  - `unit/`: Unit tests for individual modules and functions.
  - `integration/`: Tests for interactions between modules.
  - `e2e/`: End-to-end tests for main analysis scripts.
  - `conftest.py`: Common `pytest` fixtures.

- **`config.yaml`**: (In project root) Central configuration file for URLs, paths, and parameters. **Always prefer using values from this config over hardcoding.**

## 3. Configuration Management

- All global configurations (URLs, default parameters, file paths, MCMC settings) are managed via `config.yaml` in the project root.
- The `gwsiren.CONFIG` object (loaded by `gwsiren.config.load_config()`) provides access to these configurations.
- When writing code, **access configuration values via `gwsiren.CONFIG`** (e.g., `from gwsiren import CONFIG; data_dir = CONFIG.catalog["data_dir"]`).
- When writing tests, use the `mock_config` fixture (from `tests/conftest.py`) to provide a controlled, temporary configuration environment. This fixture ensures that `gwsiren.CONFIG` and direct `CONFIG` imports in other modules are patched to use the mock settings.

## 4. Coding Style and Conventions

- **PEP 8**: Follow PEP 8 guidelines for Python code.
- **Type Hinting**: Use type hints for all function signatures (arguments and return types) and important variables.
- **Docstrings**:
  - Use Google-style docstrings for modules, classes, and functions.
  - Docstrings should explain the purpose, arguments (`Args:`), return values (`Returns:`), and any exceptions raised (`Raises:`).
  - Example:
    ```python
    def my_function(param1: str, param2: int) -> bool:
        """Does something interesting.

        Args:
            param1: The first parameter, a string.
            param2: The second parameter, an integer.

        Returns:
            True if successful, False otherwise.
        """
        # ... code ...
    ```
- **Imports**:
  - Group imports: standard library, then third-party, then local application/library specific.
  - Use absolute imports from the `gwsiren` package (e.g., `from gwsiren.sky_analyzer import ...`).
- **Modularity**:
  - Strive for small, focused functions and classes.
  - Utilize existing functions from modules like `gwsiren.data.catalogs`, `gwsiren.sky_analyzer`, etc., instead of re-implementing logic.
  - If adding substantial new, reusable functionality, consider placing it in an appropriate existing module or discuss creating a new one.

## 5. Logging

- Use the standard `logging` module.
- Get a logger instance at the top of each module: `logger = logging.getLogger(__name__)`.
- Use appropriate log levels:
  - `logger.debug()`: For detailed diagnostic information.
  - `logger.info()`: For general operational information.
  - `logger.warning()`: For potential issues or unexpected situations that don't prevent execution.
  - `logger.error()`: For errors that prevent a specific operation from completing.
  - `logger.critical()`: For severe errors that might terminate the application.
- Provide informative log messages. Configuration for log levels and handlers is typically done in main script entry points (like `scripts/viz.py` or `if __name__ == "__main__":` blocks).

## 6. Error Handling

- Use specific exception types where appropriate (e.g., `ValueError`, `FileNotFoundError`).
- Catch exceptions reasonably and log errors.
- Avoid broad `except Exception:` clauses without re-raising or specific handling.
- For functions that might fail due to external factors (e.g., network, file I/O), consider returning a success flag and data/error message, or raising a custom, informative exception.

## 7. Testing (`pytest`)

- **Framework**: We use `pytest`.
- **Location**: All tests are in the `tests/` directory, mirroring the main codebase structure (e.g., tests for `gwsiren/module.py` go into `tests/unit/test_module.py`).
- **Running Tests**:
  - Run all tests: `pytest` or `pytest tests/`
  - Run tests for a specific file: `pytest tests/unit/test_galaxy_catalog_handler.py`
  - Run tests with a specific marker: `pytest -m unit`
- **Writing Tests**:
  - **New Code**: Any new function or significant logic change **must** be accompanied by corresponding unit tests.
  - **Bug Fixes**: When fixing a bug, first write a test that reproduces the bug, then fix the code so the test passes.
  - **Fixtures**: Utilize fixtures from `tests/conftest.py` (especially `mock_config`, `temp_data_dir`). Create new local fixtures if needed for specific test files.
  - **Mocking**: Use `pytest-mock` (the `mocker` fixture) to mock external dependencies (e.g., `pesummary.fetch_open_samples`, `urllib.request.urlretrieve`, `pd.read_csv` for large files, `emcee.EnsembleSampler`) in unit tests. This makes tests fast and reliable.
  - **Parametrization**: Use `@pytest.mark.parametrize` to test functions with multiple different inputs efficiently.
  - **Assertions**: Use descriptive assertions. For DataFrames/Series, use `pandas.testing.assert_frame_equal()` or `pandas.testing.assert_series_equal()`.
  - **Test Naming**: `test_function_name_when_condition_then_expected_behavior()`.
  - **Atomicity**: Each test should verify one specific behavior or outcome.
- **CI**: Tests are run automatically via GitHub Actions. Ensure your changes pass all tests.

## 8. Key External Dependencies

- **`pesummary`**: For fetching GW posterior samples. Data is cached (see `gwsiren.gw_data_fetcher` and `DEFAULT_CACHE_DIR_NAME`).
- **`healpy`**: For HEALPix sky map operations.
- **`emcee`**: For MCMC sampling.
- **`astropy`**: For cosmological calculations and coordinates.
- **`pandas`**: For data manipulation (galaxy catalogs, event data).
- **`numpy`**: For numerical operations.
- **`matplotlib`**: For plotting.

## 9. Data Handling Conventions

- **Galaxy Catalogs**: Expect columns like 'PGC', 'ra', 'dec', 'z'. See `gwsiren.data.catalogs` for details on column names and types after cleaning.
- **Sky Coordinates**:
  - Posterior samples (e.g., from `pesummary`) often have RA/Dec in radians. Convert to degrees for sky map generation (`healpy.ang2pix` often expects degrees if `lonlat=True`).
  - Galaxy catalog coordinates are typically in degrees.
- **Distances**: Luminosity distances (`dL`) are generally in Megaparsecs (Mpc).
- **Redshifts (`z`)**: Cosmological redshifts are dimensionless.

## 10. Workflow & Making Changes

1.  **Understand the Task**: Ensure you understand the requirements.
2.  **Locate Relevant Code**: Identify the modules or scripts to be modified.
3.  **Implement Changes**: Follow coding style and conventions.
4.  **Write/Update Tests**:
    * Add unit tests for new/modified logic.
    * Ensure existing tests still pass. If a change in behavior is intentional, update the relevant tests.
5.  **Run All Tests**: Execute `pytest` locally to confirm all tests pass.
6.  **Logging & Docstrings**: Add/update logging statements and docstrings as needed.
7.  **Commit Changes**: Use clear and descriptive commit messages.

## 11. Specific Instructions for AI Agent Tasks

- **Code Generation**:
  - When adding new functionality, prioritize adding it to existing relevant modules in the `gwsiren` package.
  - Ensure new public functions/classes have type hints and comprehensive docstrings.
  - Use the `gwsiren.CONFIG` object for any configurable parameters.
- **Test Generation**:
  - Place unit tests in `tests/unit/` and integration tests in `tests/integration/`.
  - Leverage `pytest` features like fixtures (especially `mock_config`, `temp_data_dir` from `conftest.py`), `mocker`, and `parametrize`.
  - For unit tests, thoroughly mock external I/O and API calls to ensure tests are fast and isolated.
- **Refactoring**:
  - Ensure all existing tests pass after refactoring.
  - If refactoring changes interfaces or exposes new testable units, add or update tests accordingly.
  - Aim to improve readability, maintainability, and adherence to project conventions.
- **Documentation**:
  - Generate or update Google-style docstrings for modules, classes, and functions.
  - Explain parameters, return values, and the overall purpose clearly.

## 12. Things to AVOID

- **Hardcoding**: Do not hardcode file paths, URLs, or magic numbers that should be in `config.yaml`. Always use `gwsiren.CONFIG`.
- **Global State**: Avoid introducing new mutable global state. Pass data explicitly.
- **Duplication**: Do not duplicate code/logic that already exists in a utility module. Import and reuse.
- **Large Files in Repo**: Do not commit large data files directly to the repository. Data should be downloaded by scripts (e.g., galaxy catalogs, GW samples).
- **Breaking Tests**: Do not submit changes that cause existing tests to fail, unless the test itself needs to be updated due to an intentional change in functionality.

Thank you for helping improve this project!


### GLADE+ DATABASE DESCRIPTION
Column no.	Name	Description
1	GLADE no	GLADE+ catalog number
2	PGC no	Principal Galaxies Catalogue number
3	GWGC name	Name in the GWGC catalog
4	HyperLEDA name	Name in the HyperLEDA catalog
5	2MASS name	Name in the 2MASS XSC catalog
6	WISExSCOS name	Name in the WISExSuperCOSMOS catalog (wiseX)
7	SDSS-DR16Q name	Name in the SDSS-DR16Q catalog
8	Object type flag	Q: the source is from the SDSS-DR16Q catalog
G:the source is from another catalog and has not been identified as a quasar
9	RA	Right ascension in degrees
10	Dec	Declination in degrees
11	B	Apparent B magnitude
12	B_err	Absolute error of apparent B magnitude
13	B flag	0: the B magnitude is measured
1: the B magnitude is calculated from the B_J magnitude
14	B_Abs	Absolute B magnitude
15	J	Apparent J magnitude
16	J_err	Absolute error of apparent J magnitude
17	H	Apparent H magnitude
18	H_err	Absolute error of apparent H magnitude
19	K	Apparent K_s magnitude
20	K_err	Absolute error of apparent K_s magnitude
21	W1	Apparent W1 magnitude
22	W1_err	Absolute error of apparent W1 magnitude
23	W2	Apparent W2 magnitude
24	W2_err	Absolute error of apparent W2 magnitude
25	W1 flag	0: the W1 magnitude is measured
1: the W1 magnitude is calculated from the K_s magnitude
26	B_J	Apparent B_J magnitude
27	B_J err	Absolute error of apparent B_J magnitude
28	z_helio	Redshift in the heliocentric frame
29	z_cmb	Redshift converted to the Cosmic Microwave Background (CMB) frame
30	z flag	0: the CMB frame redshift and luminosity distance values given in columns 29 and 33 are not corrected for the peculiar velocity
1: they are corrected values
31	v_err	Error of redshift from the peculiar velocity estimation
32	z_err	Measurement error of heliocentric redshift
33	d_L	Luminosity distance in Mpc units
34	d_L err	Error of luminosity distance in Mpc units
35	dist flag	0: the galaxy has no measured redshift or distance value
1: it has a measured photometric redshift from which we have calculated its luminosity distance
2: it has a measured luminosity distance value from which we have calculated its redshift
3: it has a measured spectroscopic redshift from which we have calculated its luminosity distance
36	M*	Stellar mass in 10^10 M_Sun units
37	M*_err	Absolute error of stellar mass in 10^10 M_Sun units
38	M* flag	0: if the stellar mass was calculated assuming no active star formation
1: if the stellar mass was calculated assuming active star formation
39	Merger rate	Base-10 logarithm of estimated BNS merger rate in the galaxy in Gyr^-1 units
40	Merger rate error	Absolute error of estimated BNS merger rate in the galaxy