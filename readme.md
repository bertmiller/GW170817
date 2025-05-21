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

### JAX Backend (Optional)

The pipeline can leverage JAX for potentially faster computations, especially on GPU/TPU hardware. To use the JAX backend, you need to install JAX with the appropriate hardware support.

**Basic CPU-only JAX:**
If you only plan to use JAX on CPU, you can install it via:
```bash
pip install "jax[cpu]==0.4.*" 
```
*(Note: The version `0.4.*` is specified in `gw-siren-pipeline/requirements.txt` and `gw-siren-pipeline/setup.py`.)*

**JAX with NVIDIA GPU Support (CUDA):**
If you have an NVIDIA GPU and a compatible CUDA toolkit installed, you can install JAX with CUDA support. For example, for CUDA 12:
```bash
pip install "jax[cuda12_pip]==0.4.*" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Replace `cuda12` with the version appropriate for your CUDA setup (e.g., `cuda11`). Refer to the [official JAX installation guide](https://github.com/google/jax#installation) for the most up-to-date instructions and versions.

**Other Platforms (TPU, ROCm for AMD GPUs):**
Refer to the official JAX installation guide for instructions on other platforms.

*(This project does not currently include a Dockerfile with JAX pre-installed, but it might be considered for future development.)*

## Running the Pipeline

The main script for end-to-end analysis is `h0_e2e_pipeline.py`.

```bash
python gw-siren-pipeline/scripts/h0_e2e_pipeline.py --event-name GW170817
```

You can also specify the numerical backend:
```bash
python gw-siren-pipeline/scripts/h0_e2e_pipeline.py --event-name GW170817 --backend jax
```
Accepted backend choices are "auto", "numpy", "jax". If not specified, it defaults to the value in `config.yaml` (which is "auto" by default).

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

### JAX Backend Specifics

When using the JAX backend, a few points are worth noting:

- **JIT Compilation & Static Shapes:** JAX achieves performance gains by Just-In-Time (JIT) compiling functions. The first call to a JIT-compiled function (like the core likelihood calculation when using the JAX backend) will be slower due to this compilation overhead. Subsequent calls with inputs of the *same shape* will be much faster. The `gwsiren` pipeline is designed to work efficiently with this, but if you were to, for example, frequently change the number of GW samples or host galaxies between likelihood calls in a custom script, you might incur repeated recompilation costs.

- **Numerical Precision (`jax_enable_x64`):** For scientific accuracy, `gwsiren`'s JAX backend is designed to work best with 64-bit floating-point precision (float64). By default, JAX might use 32-bit precision (float32) for some operations or on some devices to save memory and improve speed. To ensure 64-bit precision is used where intended, you should enable JAX's x64 mode at the beginning of your script or interactive session:
  ```python
  import jax
  jax.config.update("jax_enable_x64", True)
  ```
  The provided benchmark script `scripts/bench_jax.py` and the example notebook `examples/jax_vs_numpy_timing.ipynb` demonstrate this.

- **Numerical Drift:** While the JAX backend is tested to produce results very close to the NumPy backend (typically agreeing up to `rtol=1e-8`), minor differences due to the order of floating-point operations or variations in library implementations (e.g., for `logsumexp`) can occur. This is a common characteristic of comparing numerical outputs across different highly optimized libraries.

## Contributing

If you are an agent, please read `agents.md` for coding conventions and workflow instructions. Contributions should include appropriate tests and adhere to PEP 8 style.