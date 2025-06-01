#!/usr/bin/env python
"""
Profile the exact bottlenecks in real data computation.
Focus on algorithmic inefficiencies, not data reduction.
"""

import time
import numpy as np
import logging
import cProfile
import pstats
import io
from dataclasses import asdict
import sys
sys.path.append('gw-siren-pipeline')

from gwsiren import CONFIG
from gwsiren.multi_event_data_manager import prepare_event_data
from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def profile_real_data_bottlenecks():
    """Profile real data to identify exact bottlenecks."""
    
    # Load the actual data
    me_settings = CONFIG.multi_event_analysis
    event_packages = []
    for event_spec in me_settings.events_to_combine:
        package = prepare_event_data(asdict(event_spec))
        event_packages.append(package)
        
        # Log data characteristics
        n_samples = len(package.dl_samples)
        n_galaxies = len(package.candidate_galaxies_df)
        logger.info(f"Event {event_spec.event_id}: {n_samples} GW samples, {n_galaxies} galaxies")
        
        # Analyze redshift errors to see marginalization load
        z_errs = package.candidate_galaxies_df["z_err"].values
        high_z_err = np.sum(z_errs >= CONFIG.redshift_marginalization.z_err_threshold)
        logger.info(f"  Galaxies needing marginalization: {high_z_err}/{n_galaxies} ({high_z_err/n_galaxies*100:.1f}%)")
    
    # Profile each event separately
    for i, package in enumerate(event_packages):
        logger.info(f"\n=== PROFILING EVENT {i} ===")
        
        # Create likelihood object
        ll = get_log_likelihood_h0(
            "auto",
            package.dl_samples,
            package.candidate_galaxies_df["z"].values,
            package.candidate_galaxies_df["mass_proxy"].values,
            package.candidate_galaxies_df["z_err"].values,
            force_non_vectorized=False
        )
        
        # Profile with cProfile for detailed function-level timing
        profiler = cProfile.Profile()
        
        # Run profiled likelihood evaluation
        theta = [70.0, 0.0]
        profiler.enable()
        start_time = time.perf_counter()
        result = ll(theta)
        end_time = time.perf_counter()
        profiler.disable()
        
        total_time = end_time - start_time
        logger.info(f"Total time: {total_time:.3f}s, Result: {result:.3f}")
        
        # Analyze profile results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        logger.info("Top functions by cumulative time:")
        for line in profile_output.split('\n')[5:25]:  # Skip header
            if line.strip():
                logger.info(f"  {line}")


def analyze_computational_complexity():
    """Analyze the theoretical computational complexity."""
    
    me_settings = CONFIG.multi_event_analysis
    total_operations = 0
    
    for event_spec in me_settings.events_to_combine:
        package = prepare_event_data(asdict(event_spec))
        
        n_samples = len(package.dl_samples)
        n_galaxies = len(package.candidate_galaxies_df)
        z_errs = package.candidate_galaxies_df["z_err"].values
        
        # Count galaxies needing marginalization
        n_marginalized = np.sum(z_errs >= CONFIG.redshift_marginalization.z_err_threshold)
        n_simple = n_galaxies - n_marginalized
        
        # Estimate operations
        # Simple galaxies: O(n_samples) for PDF evaluations
        simple_ops = n_simple * n_samples
        
        # Marginalized galaxies: O(n_samples * n_quad_points) for each galaxy
        n_quad = CONFIG.redshift_marginalization.n_quad_points
        marginalized_ops = n_marginalized * n_samples * n_quad
        
        event_ops = simple_ops + marginalized_ops
        total_operations += event_ops
        
        logger.info(f"Event {event_spec.event_id}:")
        logger.info(f"  Simple galaxies: {n_simple} × {n_samples} = {simple_ops:,} operations")
        logger.info(f"  Marginalized: {n_marginalized} × {n_samples} × {n_quad} = {marginalized_ops:,} operations")
        logger.info(f"  Total: {event_ops:,} operations")
    
    logger.info(f"\nTotal computational load: {total_operations:,} operations")
    logger.info(f"At ~1 μs per operation: {total_operations/1e6:.1f} seconds")


if __name__ == "__main__":
    logger.info("Profiling real data bottlenecks...")
    
    # Analyze computational complexity first
    analyze_computational_complexity()
    
    # Profile actual execution
    profile_real_data_bottlenecks()
