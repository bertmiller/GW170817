#!/usr/bin/env python
"""Benchmark script for comparing NumPy and JAX H0LogLikelihood performance."""

import argparse
import time
import numpy as np
import logging
import sys

# Attempt to import JAX
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from astropy.cosmology import FlatLambdaCDM

# Assuming the path based on previous tasks
from gw_siren_pipeline.gwsiren.h0_mcmc_analyzer import (
    get_log_likelihood_h0,
    DEFAULT_SIGMA_V_PEC,
    DEFAULT_C_LIGHT,
    DEFAULT_OMEGA_M,
    DEFAULT_H0_PRIOR_MIN,
    DEFAULT_H0_PRIOR_MAX,
    DEFAULT_ALPHA_PRIOR_MIN,
    DEFAULT_ALPHA_PRIOR_MAX
)

logger = logging.getLogger(__name__)

def benchmark_likelihood(
    likelihood_callable, 
    theta, 
    num_evals: int, 
    num_warmup: int, 
    backend_name: str, 
    xp_module # numpy or jax.numpy
    ):
    """
    Benchmarks the given likelihood callable.

    Args:
        likelihood_callable: The likelihood function to call.
        theta: Parameters for the likelihood function.
        num_evals: Number of evaluations to time.
        num_warmup: Number of warmup evaluations.
        backend_name: Name of the backend (e.g., "numpy", "jax").
        xp_module: The numerical library module (np or jnp).

    Returns:
        float: Average time per evaluation in seconds.
    """
    logger.info(f"Warming up {backend_name} backend for {num_warmup} evaluations...")
    for _ in range(num_warmup):
        res = likelihood_callable(xp_module.asarray(theta))
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()

    logger.info(f"Benchmarking {backend_name} backend for {num_evals} evaluations...")
    start_time = time.perf_counter()
    for _ in range(num_evals):
        res = likelihood_callable(xp_module.asarray(theta))
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_eval = total_time / num_evals
    logger.info(f"{backend_name} average time per evaluation: {avg_time_per_eval:.6f} seconds")
    return avg_time_per_eval

def generate_mock_data(num_gw_samples: int, num_hosts: int, xp_module):
    """Generates mock data for H0LogLikelihood."""
    logger.info(f"Generating mock data: {num_gw_samples} GW samples, {num_hosts} host galaxies.")
    rng = np.random.default_rng(seed=123) # Use numpy for base data generation
    
    # Ensure data is float64 as JAX x64 mode would expect
    dL_gw_samples = rng.normal(loc=700, scale=70, size=num_gw_samples).astype(np.float64)
    # For H0LogLikelihood, z_values, mass_proxy_values, z_err_values are needed
    z_values = rng.uniform(low=0.01, high=0.2, size=num_hosts).astype(np.float64)
    mass_proxy_values = rng.lognormal(mean=10, sigma=1, size=num_hosts).astype(np.float64)
    # Ensure mass_proxy is positive
    mass_proxy_values = np.maximum(mass_proxy_values, 1e-5) 
    z_err_values = rng.uniform(low=0.001, high=0.005, size=num_hosts).astype(np.float64)
    
    # Convert to xp_module arrays if needed (though factory might handle it)
    # For consistency, we'll pass numpy arrays to the factory, which should handle internal conversion.
    return dL_gw_samples, z_values, mass_proxy_values, z_err_values

def main():
    parser = argparse.ArgumentParser(description="Benchmark H0LogLikelihood with NumPy and JAX.")
    parser.add_argument("--num_gw_samples", type=int, default=1000, help="Number of GW posterior samples.")
    parser.add_argument("--num_hosts", type=int, default=100000, help="Number of candidate host galaxies.")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of likelihood evaluations for timing.")
    parser.add_argument("--num_warmup", type=int, default=10, help="Number of warmup likelihood evaluations.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if JAX_AVAILABLE:
        logger.info("JAX is available. Enabling x64 mode.")
        jax.config.update("jax_enable_x64", True)
    else:
        logger.info("JAX is not available. JAX benchmarks will be skipped.")

    # Mock data generation (using NumPy as base)
    dL_gw_samples, z_values, mass_proxy_values, z_err_values = generate_mock_data(
        args.num_gw_samples, args.num_hosts, np
    )
    
    # Mock theta parameters [H0, alpha_g]
    mock_theta = [70.0, 0.0] 

    # --- NumPy Benchmark ---
    logger.info("--- Starting NumPy Benchmark ---")
    likelihood_numpy_instance = get_log_likelihood_h0(
        dL_gw_samples=dL_gw_samples,
        host_galaxies_z=z_values,
        host_galaxies_mass_proxy=mass_proxy_values,
        host_galaxies_z_err=z_err_values,
        sigma_v=DEFAULT_SIGMA_V_PEC,
        c_val=DEFAULT_C_LIGHT,
        omega_m_val=DEFAULT_OMEGA_M,
        h0_min=DEFAULT_H0_PRIOR_MIN,
        h0_max=DEFAULT_H0_PRIOR_MAX,
        alpha_min=DEFAULT_ALPHA_PRIOR_MIN,
        alpha_max=DEFAULT_ALPHA_PRIOR_MAX,
        backend_preference="numpy" 
    )
    
    time_numpy = benchmark_likelihood(
        likelihood_numpy_instance, 
        mock_theta, 
        args.num_evals, 
        args.num_warmup, 
        "NumPy", 
        likelihood_numpy_instance.xp # This is numpy
    )
    
    results_summary = [{
        "Backend": "NumPy",
        "Device Used": "CPU", # NumPy always uses CPU
        "Avg Time/eval (s)": f"{time_numpy:.6f}",
        "Speed-up vs NumPy CPU": "1.0x"
    }]

    # --- JAX Benchmark ---
    if JAX_AVAILABLE:
        logger.info("--- Starting JAX Benchmark ---")
        likelihood_jax_instance = get_log_likelihood_h0(
            dL_gw_samples=dL_gw_samples,
            host_galaxies_z=z_values,
            host_galaxies_mass_proxy=mass_proxy_values,
            host_galaxies_z_err=z_err_values,
            sigma_v=DEFAULT_SIGMA_V_PEC,
            c_val=DEFAULT_C_LIGHT,
            omega_m_val=DEFAULT_OMEGA_M,
            h0_min=DEFAULT_H0_PRIOR_MIN,
            h0_max=DEFAULT_H0_PRIOR_MAX,
            alpha_min=DEFAULT_ALPHA_PRIOR_MIN,
            alpha_max=DEFAULT_ALPHA_PRIOR_MAX,
            backend_preference="jax"
        )

        time_jax = benchmark_likelihood(
            likelihood_jax_instance, 
            mock_theta, 
            args.num_evals, 
            args.num_warmup, 
            "JAX", 
            likelihood_jax_instance.xp # This is jax.numpy
        )

        # Determine JAX device
        try:
            # jax_devices = jax.devices() # This is already a list of devices
            # device_kind = jax_devices[0].device_kind
            # platform = jax_devices[0].platform
            # Simplified: get_xp already logs this, but for summary table:
            current_jax_devices = jax.devices()
            if any(d.platform.lower() == 'gpu' or d.device_kind.lower().startswith('gpu') for d in current_jax_devices):
                jax_device_name = "GPU"
            elif any(d.platform.lower() == 'tpu' or d.device_kind.lower().startswith('tpu') for d in current_jax_devices):
                 jax_device_name = "TPU"
            else: # Default to CPU if no GPU/TPU explicitly found
                jax_device_name = "CPU"
        except Exception as e:
            logger.warning(f"Could not determine JAX device kind: {e}")
            jax_device_name = "Unknown"
            
        speed_up = time_numpy / time_jax if time_jax > 0 else float('inf')
        results_summary.append({
            "Backend": "JAX",
            "Device Used": jax_device_name,
            "Avg Time/eval (s)": f"{time_jax:.6f}",
            "Speed-up vs NumPy CPU": f"{speed_up:.2f}x"
        })
    else:
        logger.info("JAX benchmark skipped as JAX is not available.")

    # --- Report Results ---
    print("\n--- Benchmark Results ---")
    # Basic text table formatting
    headers = ["Backend", "Device Used", "Avg Time/eval (s)", "Speed-up vs NumPy CPU"]
    # Calculate column widths
    col_widths = {header: len(header) for header in headers}
    for row in results_summary:
        for header in headers:
            col_widths[header] = max(col_widths[header], len(str(row.get(header, ''))))

    # Print header
    header_line = " | ".join(header.ljust(col_widths[header]) for header in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in results_summary:
        row_line = " | ".join(str(row.get(header, '')).ljust(col_widths[header]) for header in headers)
        print(row_line)
    print("-------------------------\n")

if __name__ == "__main__":
    main()
