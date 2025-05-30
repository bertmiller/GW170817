# Gravity Wave Dark Siren Analysis

This repository contains a Python pipeline for analysing gravitational-wave events using the "standard siren" method. The code focuses on estimating cosmological parameters—primarily the Hubble constant (H0) - starting with GW170817 but being designed for other events.

It's mostly out of curiousity for me, although I hope to make some contributons with this code.

## Repository Layout

- **`gw-siren-pipeline/`** – Python package `gwsiren` with reusable modules and analysis scripts.
- **`tests/`** – Project-level test configuration.
- **`agents.md`** – Guidelines for agents to contribute to this repository.

The bulk of the functionality lives in the `gwsiren` package. See `gw-siren-pipeline/README.md` for a brief overview of the package layout.

## Installation

1. Ensure you have Python 3.8+ available.
2. Install the dependencies and package in editable mode:

   ```bash
   cd gw-siren-pipeline
   pip install -r requirements.txt
   pip install -e .
   ```

This provides the `gwsiren` package and command-line scripts.

## Running the Pipeline

The main script for end-to-end analysis is `h0_e2e_pipeline.py`.

```bash
python gw-siren-pipeline/scripts/h0_e2e_pipeline.py --event-name GW170817
```

The script downloads event posteriors, processes the galaxy catalogue defined in `config.yaml`, performs sky localization, and runs an MCMC sampler to produce posterior samples of `H0`. Results are written to the `output/` directory.

## Configuration

Global settings such as catalogue paths, MCMC parameters, and cosmology assumptions are stored in `gw-siren-pipeline/config.yaml`. They are accessed at runtime via `gwsiren.CONFIG`.

To override the default config, set the environment variable `GWSIREN_CONFIG` to point to an alternative YAML file.

### Multi-Event Configuration

The optional `multi_event_analysis` section in `config.yaml` collects settings
for combining multiple gravitational-wave events in a single run. When present,
it defines which events to process, global priors on `H0` and `alpha`, optional
MCMC overrides, and cosmology parameters. Candidate galaxy lists generated for
each event are cached in the directory specified by
`multi_event_analysis.run_settings.candidate_galaxy_cache_dir` to speed up
subsequent runs. A minimal example is provided at the end of `config.yaml`.

To execute a multi-event run, call the orchestrator script:

```bash
python gw-siren-pipeline/scripts/run_multi_event_pipeline.py
```

An example wrapper is provided in `examples/run_multi_event_example.py` which
assumes the configuration above and simply invokes the orchestrator function.

## Running Tests

Project tests are written with `pytest`. Execute them from the repository root:

```bash
pytest
```

## Contributing

If you are an agent, please read `agents.md` for coding conventions and workflow instructions. Contributions should include appropriate tests and adhere to PEP 8 style.