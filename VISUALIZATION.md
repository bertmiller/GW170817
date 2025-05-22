# Visualization Tools

This document describes the utilities for plotting the results of a multi-event analysis.

## Example Usage

After running the multi-event pipeline you will have a file with combined samples,
e.g. `output/multi_event_runs/demo_run/global_samples.npy`. Individual event
samples are saved as `output/H0_samples_<event_id>.npy`.

Run the example script to generate plots:

```bash
python examples/visualize_multi_event_results.py \
    --combined-samples output/multi_event_runs/demo_run/global_samples.npy
```

The script produces:

- `H0_overlaid.pdf` – overlaid one-dimensional posteriors for $H_0$ from each
  event and the combined analysis.
- `alpha_overlaid.pdf` – analogous plot for $\alpha$.
- `combined_corner.pdf` – a corner plot of the joint $H_0$–$\alpha$ posterior.

All plots are saved under `output/plots/` by default.
