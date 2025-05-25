#!/usr/bin/env python
"""
High-performance distance computation caching using H0-specific interpolation tables.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from collections import OrderedDict
import time

logger = logging.getLogger(__name__)

try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Distance caching will be disabled.")


class H0DistanceCache:
    """
    High-performance cache for luminosity distance computations using interpolation tables.
    
    For each H0 value, builds a cubic interpolation table once, then uses fast 
    interpolation for all subsequent distance queries at that H0.
    
    Perfect for MCMC usage pattern:
    - Build interpolator once per H0 value
    - Use many times for galaxy distance computations
    - Same H0 values revisited frequently during sampling
    """
    
    def __init__(
        self, 
        max_h0_cache_size: int = 100,
        z_grid_min: float = 0.001,
        z_grid_max: float = 3.0,
        z_grid_points: int = 2000,
        h0_tolerance: float = 0.01,
        interpolation_kind: str = 'cubic'
    ):
        """
        Initialize the distance cache.
        
        Args:
            max_h0_cache_size: Maximum number of H0 values to cache
            z_grid_min: Minimum redshift for interpolation grid
            z_grid_max: Maximum redshift for interpolation grid  
            z_grid_points: Number of points in interpolation grid
            h0_tolerance: Tolerance for considering H0 values equivalent (km/s/Mpc)
            interpolation_kind: Type of interpolation ('linear', 'cubic', etc.)
        """
        self.max_cache_size = max_h0_cache_size
        self.z_grid_min = z_grid_min
        self.z_grid_max = z_grid_max
        self.z_grid_points = z_grid_points
        self.h0_tolerance = h0_tolerance
        self.interpolation_kind = interpolation_kind
        
        # LRU cache for interpolators: {h0_key: interpolator}
        self.interpolator_cache = OrderedDict()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_build_time = 0.0
        self.total_interpolation_time = 0.0
        
        # Pre-compute redshift grid
        self.z_grid = np.linspace(z_grid_min, z_grid_max, z_grid_points)
        
        # Reference to the distance computation function (set by parent)
        self._distance_compute_func = None
        
        logger.info(f"Initialized H0DistanceCache: {z_grid_points} z-points, {max_h0_cache_size} H0 cache size")
    
    def set_distance_function(self, distance_func):
        """Set the reference to the actual distance computation function."""
        self._distance_compute_func = distance_func
    
    def get_distances(self, z_values, h0_val: float) -> np.ndarray:
        """
        Get luminosity distances for given redshifts and H0.
        
        Uses cached interpolation table if available, builds new one if needed.
        
        Args:
            z_values: Redshift values (scalar or array)
            h0_val: Hubble constant value
            
        Returns:
            Luminosity distances corresponding to z_values
        """
        if not SCIPY_AVAILABLE:
            # Fallback to direct computation if scipy not available
            return self._distance_compute_func(z_values, h0_val)
        
        start_time = time.perf_counter()
        
        # Normalize H0 to cache key
        h0_key = self._get_h0_cache_key(h0_val)
        
        # Check if we have cached interpolator
        if h0_key in self.interpolator_cache:
            # Cache hit - move to end (LRU)
            interpolator = self.interpolator_cache.pop(h0_key)
            self.interpolator_cache[h0_key] = interpolator
            self.cache_hits += 1
        else:
            # Cache miss - build new interpolator
            interpolator = self._build_interpolator(h0_key)
            self._store_interpolator(h0_key, interpolator)
            self.cache_misses += 1
        
        # Use interpolation to get distances
        z_input = np.atleast_1d(z_values)
        distances = interpolator(z_input)
        
        # Handle scalar input
        if np.isscalar(z_values):
            distances = float(distances[0])
        
        self.total_interpolation_time += time.perf_counter() - start_time
        return distances
    
    def _get_h0_cache_key(self, h0_val: float) -> float:
        """
        Convert H0 value to cache key, grouping nearby values.
        
        This allows cache hits for H0 values that are very close
        (within tolerance) to previously computed values.
        """
        return round(h0_val / self.h0_tolerance) * self.h0_tolerance
    
    def _build_interpolator(self, h0_key: float):
        """
        Build cubic interpolation table for given H0 value.
        
        Args:
            h0_key: Normalized H0 value to build interpolator for
            
        Returns:
            scipy.interpolate.interp1d object
        """
        if self._distance_compute_func is None:
            raise RuntimeError("Distance computation function not set. Call set_distance_function() first.")
        
        logger.debug(f"Building distance interpolator for H0={h0_key:.2f}")
        build_start = time.perf_counter()
        
        # Compute distances on the grid
        grid_distances = self._distance_compute_func(self.z_grid, h0_key)
        
        # Create interpolator
        interpolator = interp1d(
            self.z_grid, 
            grid_distances,
            kind=self.interpolation_kind,
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        build_time = time.perf_counter() - build_start
        self.total_build_time += build_time
        
        logger.debug(f"Built H0={h0_key:.2f} interpolator in {build_time:.3f}s")
        return interpolator
    
    def _store_interpolator(self, h0_key: float, interpolator):
        """
        Store interpolator in cache with LRU eviction.
        
        Args:
            h0_key: H0 cache key
            interpolator: scipy interpolation object
        """
        # Add to cache
        self.interpolator_cache[h0_key] = interpolator
        
        # Evict oldest if over limit
        if len(self.interpolator_cache) > self.max_cache_size:
            oldest_key = next(iter(self.interpolator_cache))
            removed = self.interpolator_cache.pop(oldest_key)
            logger.debug(f"Evicted H0={oldest_key:.2f} interpolator from cache")
    
    def validate_accuracy(self, test_z_values, test_h0_values, max_relative_error=1e-4) -> bool:
        """
        Validate interpolation accuracy against direct computation.
        
        Args:
            test_z_values: Array of redshift values to test
            test_h0_values: Array of H0 values to test  
            max_relative_error: Maximum acceptable relative error
            
        Returns:
            True if all interpolated values are within error tolerance
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Cannot validate accuracy - SciPy not available")
            return True
        
        logger.info("Validating distance cache accuracy...")
        
        max_error = 0.0
        total_tests = 0
        
        for h0_val in test_h0_values:
            for z_val in test_z_values:
                # Direct computation
                direct = self._distance_compute_func(z_val, h0_val)
                
                # Cached computation  
                cached = self.get_distances(z_val, h0_val)
                
                # Relative error
                rel_error = abs(cached - direct) / abs(direct)
                max_error = max(max_error, rel_error)
                total_tests += 1
                
                if rel_error > max_relative_error:
                    logger.error(f"Accuracy validation failed: z={z_val:.3f}, H0={h0_val:.1f}, "
                               f"rel_error={rel_error:.2e} > {max_relative_error:.2e}")
                    return False
        
        logger.info(f"Accuracy validation passed: {total_tests} tests, max_rel_error={max_error:.2e}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_h0_values': len(self.interpolator_cache),
            'total_build_time': self.total_build_time,
            'total_interpolation_time': self.total_interpolation_time,
            'avg_build_time': self.total_build_time / max(self.cache_misses, 1),
            'avg_interpolation_time': self.total_interpolation_time / max(total_queries, 1)
        }
    
    def clear_cache(self):
        """Clear all cached interpolators and reset statistics."""
        self.interpolator_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_build_time = 0.0
        self.total_interpolation_time = 0.0
        logger.info("Distance cache cleared")
    
    def log_statistics(self):
        """Log cache performance statistics."""
        stats = self.get_statistics()
        logger.info(f"Distance Cache Stats: {stats['cache_hits']}/{stats['cache_hits'] + stats['cache_misses']} hits "
                   f"({stats['hit_rate']:.1%}), {stats['cached_h0_values']} H0 values cached")
        logger.info(f"Avg build time: {stats['avg_build_time']:.3f}s, "
                   f"avg interpolation time: {stats['avg_interpolation_time']:.6f}s")


def create_distance_cache(
    max_cache_size: int = 100,
    z_range: Tuple[float, float] = (0.001, 3.0),
    z_points: int = 2000,
    h0_tolerance: float = 0.01
) -> H0DistanceCache:
    """
    Factory function to create a distance cache with reasonable defaults.
    
    Args:
        max_cache_size: Maximum number of H0 interpolators to cache
        z_range: (min_z, max_z) for interpolation grid
        z_points: Number of points in redshift grid
        h0_tolerance: H0 tolerance for cache key grouping
        
    Returns:
        Configured H0DistanceCache instance
    """
    return H0DistanceCache(
        max_h0_cache_size=max_cache_size,
        z_grid_min=z_range[0],
        z_grid_max=z_range[1], 
        z_grid_points=z_points,
        h0_tolerance=h0_tolerance
    ) 