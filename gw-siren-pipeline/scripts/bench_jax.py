#!/usr/bin/env python
"""Benchmark the H0 likelihood using NumPy and JAX backends."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Tuple

import numpy as np

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0

logger = logging.getLogger(__name__)


def generate_mock_data(num_gw_samples: int, num_hosts: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate mock data for benchmarking.

    Args:
        num_gw_samples: Number of GW luminosity distance samples.
        num_hosts: Number of candidate host galaxies.

    Returns:
        Tuple containing arrays of GW distance samples, host galaxy redshifts,
        host galaxy mass proxies, and host redshift errors.
    """
    dL_gw_samples = np.random.normal(loc=400.0, scale=50.0, size=num_gw_samples).astype(np.float32)
    host_galaxies_z = np.random.uniform(0.01, 0.2, size=num_hosts).astype(np.float32)
    host_galaxies_mass_proxy = np.random.lognormal(mean=np.log(1e10), sigma=0.5, size=num_hosts).astype(np.float32)
    host_galaxies_z_err = (0.015 * (1.0 + host_galaxies_z)).astype(np.float32)
    return dL_gw_samples, host_galaxies_z, host_galaxies_mass_proxy, host_galaxies_z_err


def benchmark_likelihood(
    likelihood_callable,
    theta: Tuple[float, float],
    num_evals: int,
    num_warmup: int,
    backend_name: str,
) -> float:
    """Benchmark a likelihood callable.

    Args:
        likelihood_callable: The likelihood function to benchmark.
        theta: Parameter tuple ``(H0, alpha)`` to evaluate.
        num_evals: Number of timed evaluations.
        num_warmup: Number of warm-up evaluations.
        backend_name: Name of the backend ("numpy" or "jax").

    Returns:
        Average time per likelihood evaluation in seconds.
    """
    for _ in range(num_warmup):
        val = likelihood_callable(theta)
        if backend_name == "jax" and hasattr(val, "block_until_ready"):
            val.block_until_ready()

    start_time = time.perf_counter()
    for _ in range(num_evals):
        val = likelihood_callable(theta)
    if backend_name == "jax" and hasattr(val, "block_until_ready"):
        val.block_until_ready()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    return total_time / float(num_evals)


def main() -> None:
    """Entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs JAX for the H0 likelihood")
    parser.add_argument("--num_gw_samples", type=int, default=1000, help="Number of GW distance samples")
    parser.add_argument("--num_hosts", type=int, default=10000, help="Number of candidate host galaxies")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of likelihood evaluations to time")
    parser.add_argument("--num_warmup", type=int, default=10, help="Number of warm-up evaluations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(
        "Generating mock data: %d GW samples, %d host galaxies",
        args.num_gw_samples,
        args.num_hosts,
    )
    dL_samples, host_z, host_mass, host_z_err = generate_mock_data(args.num_gw_samples, args.num_hosts)
    theta = (70.0, 0.0)

    log_like_numpy = get_log_likelihood_h0(
        dL_samples,
        host_z,
        host_mass,
        host_z_err,
        backend_preference="numpy",
    )
    time_numpy = benchmark_likelihood(log_like_numpy, theta, args.num_evals, args.num_warmup, "numpy")

    log_like_jax = get_log_likelihood_h0(
        dL_samples,
        host_z,
        host_mass,
        host_z_err,
        backend_preference="jax",
    )

    jax_device = "CPU"
    try:
        import jax  # type: ignore

        for d in jax.devices():
            if d.platform.lower() in ["gpu", "metal"] or "gpu" in getattr(d, "device_kind", "").lower():
                jax_device = "GPU"
                break
    except Exception as exc:  # pragma: no cover - optional introspection
        logger.debug("Unable to inspect JAX devices: %s", exc)

    time_jax = benchmark_likelihood(log_like_jax, theta, args.num_evals, args.num_warmup, "jax")

    speedup = time_numpy / time_jax if time_jax > 0 else float("inf")

    table = (
        f"\n{'Backend':<10}|{'Device':<10}|{'Avg. Time/eval (s)':<20}|{'Speed-up vs. NumPy CPU':<22}\n"
        f"{'-'*10}|{'-'*10}|{'-'*20}|{'-'*22}\n"
        f"{'NumPy':<10}|{'CPU':<10}|{time_numpy:<20.6f}|{'1.00x':<22}\n"
        f"{'JAX':<10}|{jax_device:<10}|{time_jax:<20.6f}|{speedup:<22.2f}"
    )
    print(table)


if __name__ == "__main__":
    main()
