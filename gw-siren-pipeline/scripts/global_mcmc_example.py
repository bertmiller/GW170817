#!/usr/bin/env python
"""Example script demonstrating multi-event global MCMC usage."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
from emcee.interruptible_pool import InterruptiblePool

from gwsiren.global_mcmc import run_global_mcmc, process_global_mcmc_samples, save_global_samples
from gwsiren.event_data import EventDataPackage
# from gwsiren.combined_likelihood import CombinedLogLikelihood  # Placeholder for actual import

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run global multi-event MCMC")
    parser.add_argument("output", help="Output path for samples")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Placeholder for actual event data packages and likelihood construction
    event_data_packages: list[EventDataPackage] = []  # TODO: populate with EventDataPackage instances
    combined_ll = None  # CombinedLogLikelihood(event_data_packages)
    if combined_ll is None:
        logger.error("CombinedLogLikelihood not constructed. This is a placeholder script.")
        return

    initial_cfg = {"H0": {"mean": 70.0, "std": 10.0}, "alpha": {"mean": 0.0, "std": 0.5}}
    n_cores = os.cpu_count() or 1
    with InterruptiblePool(n_cores) as pool:
        sampler = run_global_mcmc(combined_ll, initial_pos_config=initial_cfg, pool=pool)

    samples = process_global_mcmc_samples(sampler)
    if samples is not None:
        save_global_samples(samples, args.output)
        logger.info("Saved %d samples", len(samples))


if __name__ == "__main__":
    main()
