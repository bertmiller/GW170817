#!/usr/bin/env python
"""Command-line entry for running the H0 analysis pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from gwsiren.pipeline import (
    run_full_analysis,
    save_h0_samples_and_print_summary,
    OUTPUT_DIR,
    DEFAULT_EVENT_NAME,
    NSIDE_SKYMAP,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the end-to-end analysis pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Run H0 estimation pipeline")
    parser.add_argument(
        "event_name",
        nargs="?",
        default=DEFAULT_EVENT_NAME,
        help=f"GW event name (default: {DEFAULT_EVENT_NAME})",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "jax"],
        default="auto",
        help="Numerical backend to use (default: auto)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    event_name = args.event_name
    logger.info("Starting analysis for %s", event_name)

    results = run_full_analysis(event_name, perform_mcmc=True, backend_override=args.backend)
    if results.get("error"):
        logger.error("Pipeline failed: %s", results["error"])
        sys.exit(1)

    if results.get("flat_h0_samples") is not None:
        hosts_df = results.get("candidate_hosts_df")
        num_hosts = len(hosts_df) if isinstance(hosts_df, pd.DataFrame) else 0
        save_h0_samples_and_print_summary(
            results["flat_h0_samples"], event_name, num_hosts
        )
    logger.info("Analysis completed for %s", event_name)


if __name__ == "__main__":
    main()
