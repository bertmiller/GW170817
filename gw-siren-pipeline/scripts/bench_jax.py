import sys
import logging
import numpy as np
import cProfile
import pstats
import io
import timeit
import platform # For reporting system info
# functools.partial is not strictly needed here as we call the method directly or via a simple lambda/def.

# To ensure gwsiren can be imported, you might need to run this script
# from the root of the GW170817 directory, or ensure gw-siren-pipeline
# is in your PYTHONPATH.
# For example: PYTHONPATH=$PYTHONPATH:/path/to/GW170817/gw-siren-pipeline python gw-siren-pipeline/scripts/bench_jax.py
try:
    from gwsiren.backends import get_xp, BackendNotAvailableError
    from gwsiren.h0_mcmc_analyzer import H0LogLikelihood, MEMORY_THRESHOLD_BYTES, BYTES_PER_ELEMENT
except ImportError as e:
    print(f"ImportError: {e}. Please ensure 'gwsiren' is in your PYTHONPATH.")
    print("Example: export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root/gw-siren-pipeline")
    sys.exit(1)

# Configure basic logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Benchmark-specific default parameters to avoid reliance on CONFIG
# These values would typically come from a config file in the main application
BENCHMARK_SIGMA_V_PEC = 200.0  # km/s
BENCHMARK_C_LIGHT = 299792.458 # km/s
BENCHMARK_OMEGA_M = 0.3
BENCHMARK_H0_PRIOR_MIN = 10.0   # km/s/Mpc
BENCHMARK_H0_PRIOR_MAX = 200.0  # km/s/Mpc
BENCHMARK_ALPHA_PRIOR_MIN = -0.5
BENCHMARK_ALPHA_PRIOR_MAX = 0.5

# --- Define Larger Workload ---
NUM_GW_SAMPLES_LARGE = 10000
NUM_HOSTS_LARGE = 1000

def generate_mock_data_np(n_gw_samples, n_hosts):
    logger.info(f"Generating mock data: {n_gw_samples} GW samples, {n_hosts} host galaxies.")
    mock_dL_gw_np = np.random.normal(loc=700, scale=70, size=n_gw_samples)
    mock_host_zs_np = np.random.uniform(0.01, 0.2, size=n_hosts)
    mock_mass_proxy_np = np.random.rand(n_hosts) * 100 + 1.0
    mock_z_err_np = np.random.uniform(0.0001, 0.005, size=n_hosts)
    logger.info("Mock data generation complete.")
    return mock_dL_gw_np, mock_host_zs_np, mock_mass_proxy_np, mock_z_err_np

def profile_likelihood_evaluation(
    log_likelihood_instance,
    theta_values_to_test,
    num_calls_per_theta=5 # Reduced for larger workload
):
    backend_display_name = getattr(log_likelihood_instance, 'backend_name', 'N/A')
    device_name = getattr(log_likelihood_instance, 'device_name', 'cpu' if backend_display_name == 'numpy' else 'N/A')
    logger.info(
        f"Profiling H0LogLikelihood.__call__ for backend: {backend_display_name} on {device_name} "
        f"({num_calls_per_theta} calls per H0 value)."
    )
    if theta_values_to_test:
        logger.info("Performing a warm-up call for profiling...")
        try:
            warmup_result = log_likelihood_instance(theta_values_to_test[0])
            if hasattr(warmup_result, 'block_until_ready'):
                warmup_result.block_until_ready()
            logger.info("Profiling warm-up call completed.")
        except Exception as e:
            logger.warning(f"Profiling warm-up call failed: {e}.")

    pr = cProfile.Profile()
    pr.enable()
    for theta_val in theta_values_to_test:
        for _ in range(num_calls_per_theta):
            result = log_likelihood_instance(theta_val)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
    pr.disable()
    
    profile_stats_stream = io.StringIO()
    pstats.Stats(pr, stream=profile_stats_stream).sort_stats('cumulative').print_stats(20)
    logger.info(f"cProfile results for {backend_display_name} Likelihood (__call__ method):")
    print(profile_stats_stream.getvalue())

def benchmark_likelihood_call(
    log_likelihood_instance,
    theta_for_benchmark,
    num_runs=10, # Reduced for larger workload
    num_repeats=5 # Reduced for larger workload
):
    backend_display_name = getattr(log_likelihood_instance, 'backend_name', 'N/A')
    device_name = getattr(log_likelihood_instance, 'device_name', 'cpu' if backend_display_name == 'numpy' else 'N/A')
    
    logger.info(
        f"Benchmarking H0LogLikelihood.__call__ for backend: {backend_display_name} on {device_name} "
        f"({num_runs} executions, {num_repeats} repeats)."
    )

    def timed_call_wrapper(instance, theta_val_list):
        res = instance(theta_val_list)
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()

    logger.info("Performing a warm-up call for timeit benchmark...")
    try:
        timed_call_wrapper(log_likelihood_instance, theta_for_benchmark)
        logger.info("Timeit warm-up call successful.")
    except Exception as e:
        logger.warning(f"Timeit warm-up call failed: {e}.")

    timeit_globals = {
        "timed_call_wrapper": timed_call_wrapper,
        "instance": log_likelihood_instance,
        "theta": theta_for_benchmark
    }
    
    execution_times = timeit.repeat(
        stmt="timed_call_wrapper(instance, theta)",
        globals=timeit_globals,
        number=num_runs,
        repeat=num_repeats
    )
    
    min_avg_time_per_call = min(execution_times) / num_runs
    logger.info(
        f"Min average time per {backend_display_name} likelihood call: "
        f"{min_avg_time_per_call * 1e6:.2f} microseconds ({min_avg_time_per_call * 1e3:.2f} ms)"
    )
    return min_avg_time_per_call

def main():
    logger.info(f"--- Likelihood Benchmark Script (Large Workload) ---")
    logger.info(f"System: {platform.system()} {platform.release()}, Machine: {platform.machine()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"NumPy: {np.__version__}")
    try:
        import jax
        logger.info(f"JAX: {jax.__version__}, jaxlib: {jax.lib.__version__}")
    except ImportError:
        logger.info("JAX not installed.")

    logger.info(f"Large Workload: GW Samples = {NUM_GW_SAMPLES_LARGE}, Host Galaxies = {NUM_HOSTS_LARGE}")

    results = []

    # Common parameters for tests
    h0_values_for_tests = [65.0, 70.0, 75.0]
    alpha_value_for_tests = 0.0
    thetas_for_tests = [[h0, alpha_value_for_tests] for h0 in h0_values_for_tests]
    theta_for_benchmark_single = thetas_for_tests[1] # Use [70.0, 0.0] for timeit

    # Generate mock data once
    mock_dL_gw, mock_host_zs, mock_mass_proxy, mock_z_err = generate_mock_data_np(
        NUM_GW_SAMPLES_LARGE, NUM_HOSTS_LARGE
    )

    # --- JAX Benchmark ---
    logger.info("\n--- Starting JAX Benchmark ---")
    jax_time = None
    jax_device = "N/A"
    try:
        xp_jax, backend_name_jax, device_name_jax = get_xp("jax")
        jax_device = device_name_jax
        if backend_name_jax != "jax":
            logger.error(f"Requested JAX, but got {backend_name_jax}. Skipping JAX benchmark.")
        else:
            logger.info(f"JAX backend: {backend_name_jax} on device: {device_name_jax}")
            log_like_jax_instance = H0LogLikelihood(
                xp=xp_jax, backend_name=backend_name_jax,
                # device_name=device_name_jax, # Pass if H0LogLikelihood stores it
                dL_gw_samples=mock_dL_gw, host_galaxies_z=mock_host_zs,
                host_galaxies_mass_proxy=mock_mass_proxy, host_galaxies_z_err=mock_z_err,
                sigma_v=BENCHMARK_SIGMA_V_PEC, c_val=BENCHMARK_C_LIGHT, omega_m_val=BENCHMARK_OMEGA_M,
                h0_min=BENCHMARK_H0_PRIOR_MIN, h0_max=BENCHMARK_H0_PRIOR_MAX,
                alpha_min=BENCHMARK_ALPHA_PRIOR_MIN, alpha_max=BENCHMARK_ALPHA_PRIOR_MAX,
                use_vectorized_likelihood=True # JAX is inherently optimized
            )
            setattr(log_like_jax_instance, 'device_name', device_name_jax) # For logging in benchmark funcs

            profile_likelihood_evaluation(log_like_jax_instance, thetas_for_tests)
            jax_time = benchmark_likelihood_call(log_like_jax_instance, theta_for_benchmark_single)
            if jax_time is not None:
                results.append({
                    "Backend": "JAX", "Device": jax_device,
                    "Time (ms)": jax_time * 1e3,
                    "GW Samples": NUM_GW_SAMPLES_LARGE, "Hosts": NUM_HOSTS_LARGE
                })

    except BackendNotAvailableError as e:
        logger.error(f"JAX backend not available: {e}. Skipping JAX benchmark.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during JAX benchmark: {e}")

    # --- NumPy Benchmark ---
    logger.info("\n--- Starting NumPy Benchmark ---")
    numpy_time = None
    numpy_device = "cpu" # NumPy always uses CPU
    try:
        xp_numpy, backend_name_numpy, device_name_numpy = get_xp("numpy")
        if backend_name_numpy != "numpy":
            logger.error(f"Requested NumPy, but got {backend_name_numpy}. Skipping NumPy benchmark.")
        else:
            logger.info(f"NumPy backend: {backend_name_numpy} on device: {device_name_numpy}")

            # Determine if NumPy should use vectorized path for this workload
            # This mimics the logic in get_log_likelihood_h0 from h0_mcmc_analyzer.py
            current_elements = NUM_GW_SAMPLES_LARGE * NUM_HOSTS_LARGE
            max_elements_for_vectorization = MEMORY_THRESHOLD_BYTES / BYTES_PER_ELEMENT
            numpy_use_vectorized = current_elements <= max_elements_for_vectorization
            
            logger.info(f"NumPy: {current_elements} elements vs threshold {max_elements_for_vectorization}. Vectorized: {numpy_use_vectorized}")

            log_like_numpy_instance = H0LogLikelihood(
                xp=xp_numpy, backend_name=backend_name_numpy,
                dL_gw_samples=mock_dL_gw, host_galaxies_z=mock_host_zs,
                host_galaxies_mass_proxy=mock_mass_proxy, host_galaxies_z_err=mock_z_err,
                sigma_v=BENCHMARK_SIGMA_V_PEC, c_val=BENCHMARK_C_LIGHT, omega_m_val=BENCHMARK_OMEGA_M,
                h0_min=BENCHMARK_H0_PRIOR_MIN, h0_max=BENCHMARK_H0_PRIOR_MAX,
                alpha_min=BENCHMARK_ALPHA_PRIOR_MIN, alpha_max=BENCHMARK_ALPHA_PRIOR_MAX,
                use_vectorized_likelihood=numpy_use_vectorized
            )
            setattr(log_like_numpy_instance, 'device_name', device_name_numpy)


            profile_likelihood_evaluation(log_like_numpy_instance, thetas_for_tests)
            numpy_time = benchmark_likelihood_call(log_like_numpy_instance, theta_for_benchmark_single)
            if numpy_time is not None:
                results.append({
                    "Backend": f"NumPy ({'Vectorized' if numpy_use_vectorized else 'Looped'})", 
                    "Device": numpy_device,
                    "Time (ms)": numpy_time * 1e3,
                    "GW Samples": NUM_GW_SAMPLES_LARGE, "Hosts": NUM_HOSTS_LARGE
                })
                
    except BackendNotAvailableError as e: # Should not happen for NumPy unless system is very broken
        logger.error(f"NumPy backend not available: {e}. Skipping NumPy benchmark.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during NumPy benchmark: {e}")

    # --- Print Results Table ---
    logger.info("\n--- Benchmark Results Summary ---")
    if not results:
        logger.info("No benchmark results to display.")
        return

    # Simple text-based table
    header = "| Backend          | Device      | GW Samples | Hosts   | Avg. Time (ms) |"
    separator = "|------------------|-------------|------------|---------|----------------|"
    
    print("\nBenchmark Comparison Table:")
    print(header)
    print(separator)
    for res in results:
        time_str = f"{res['Time (ms)']:.2f}" if res['Time (ms)'] is not None else "N/A"
        print(f"| {res['Backend']:<16} | {res['Device']:<11} | {res['GW Samples']:<10} | {res['Hosts']:<7} | {time_str:>14} |")
    
    if len(results) == 2 and results[0]['Time (ms)'] is not None and results[1]['Time (ms)'] is not None:
        jax_res = next(r for r in results if r['Backend'] == 'JAX')
        numpy_res = next(r for r in results if 'NumPy' in r['Backend'])
        if jax_res['Time (ms)'] > 0 and numpy_res['Time (ms)'] > 0:
            if jax_res['Time (ms)'] < numpy_res['Time (ms)']:
                speedup = numpy_res['Time (ms)'] / jax_res['Time (ms)']
                logger.info(f"\nJAX was approximately {speedup:.2f}x faster than NumPy for this workload.")
            else:
                slowdown = jax_res['Time (ms)'] / numpy_res['Time (ms)']
                logger.info(f"\nNumPy was approximately {slowdown:.2f}x faster than JAX for this workload.")


if __name__ == '__main__':
    main()