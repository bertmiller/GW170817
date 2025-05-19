# GW Siren Pipeline

A Python project for gravitational wave data analysis and H0 estimation.

## Project Structure (Initial)

- `gwsiren/`: Core library package.
- `scripts/`: Executable analysis scripts (to be populated).
- `tests/`: Unit, integration, and end-to-end tests.
- Other `.py` files: Modules and scripts to be refactored.

## Setup (Placeholder)

To be updated as the project is refactored.

## Caching

The pipeline caches intermediate data to avoid repeated downloads and
processing. Two key directories can be configured in `config.yaml` under the
`multi_event_analysis.run_settings` section:

- `candidate_galaxy_cache_dir` – stores per-event candidate galaxy tables.
- `gw_posteriors_cache_dir` – stores extracted GW posterior samples
  (`dL`, `ra`, `dec`) in `.npz` files.
