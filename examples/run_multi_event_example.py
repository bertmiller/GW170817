#!/usr/bin/env python
"""Example script to run the multi-event analysis pipeline.

The ``config.yaml`` should contain a ``multi_event_analysis`` section similar to:

multi_event_analysis:
  run_settings:
    run_label: "demo_run"
    base_output_directory: "output/multi_event_runs/"
  events_to_combine:
    - event_id: "GW170817"
      # gw_dl_samples_path: "precomputed/GW170817/dL.npy"
      # candidate_galaxies_path: "precomputed/GW170817/hosts.csv"
    - event_id: "GW190814"
      single_event_processing_params:
        catalog_type: "glade+"
  priors:
    H0: {min: 10.0, max: 200.0}
    alpha: {min: -2.0, max: 2.0}
  mcmc:
    n_walkers: 64
    n_steps: 4000
    burnin: 500
    thin_by: 10
    initial_pos_config:
      H0: {mean: 70.0, std: 5.0}
      alpha: {mean: 0.0, std: 0.2}
  cosmology:
    sigma_v_pec: 300.0

Running this example simply calls the orchestrator which reads the global
configuration and executes the pipeline.
"""

from scripts.run_multi_event_pipeline import execute_multi_event_analysis

if __name__ == "__main__":
    execute_multi_event_analysis()
