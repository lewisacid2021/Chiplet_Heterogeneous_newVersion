import random
import subprocess
import time
import glob
import shutil
import os
import configparser

# Define parameter ranges for SniperSim (CPU)
sniper_parameter_ranges = {
    "frequency": [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6],  # GHz
    "logical_cpus": [1, 2, 4],  # SMT threads
    "l1_icache_size": [32, 64, 128, 256],  # KB
    "l1_dcache_size": [32, 64, 128, 256],  # KB
    "l2_cache_size": [256, 512, 1024, 2048],  # KB
    "l3_cache_size": [8192, 16384, 32768, 65536],  # KB
    "per_controller_bandwidth": [7.6, 15.2, 30.4, 60.8],  # GB/s
    "l1_shared_cores": [1, 2, 4, 8, 16],  # Number of cores sharing L1 cache
    "l2_shared_cores": [1, 2, 4, 8, 16],  # Number of cores sharing L2 cache
    "l3_shared_cores": [1, 2, 4, 8, 16],  # Number of cores sharing L3 cache
}

# Define parameter ranges for GPGPU-Sim (GPU)
gpgpu_parameter_ranges = {
    "gpgpu_n_clusters": range(1, 20),  # Number of GPU clusters
    "gpgpu_n_cores_per_cluster": range(1, 5),  # Number of cores per cluster
    "gpgpu_shader_registers": [16384, 32768, 49152, 65536],  # Number of registers per SM
    "gpgpu_shmem_size": [16384, 32768, 49152, 65536],  # Shared memory size
    "gpgpu_cache_dl1_sets": [16, 32, 48, 64],  # Number of L1 data cache sets
    "gpgpu_cache_dl2_sets": [32, 64, 96, 128],  # Number of L2 cache sets
    "gpgpu_num_sp_units": range(1, 5),  # Number of single-precision floating-point units
    "gpgpu_num_sfu_units": range(1, 5),  # Number of special function units
    "gpgpu_clock_domains_core": range(500, 1000, 100),  # Core clock frequency (MHz)
}

# Define simulator paths and output file paths
INPUT_FILE = "../matmul/matmul.yml"
CHIPLET_SIMULATOR_PATH = "../../interchiplet/bin/interchiplet"  # Simulator executable for both CPU and GPU
SNIPER_CONFIG_PATH = "../../snipersim/config/gainestown.cfg"  # SniperSim config file
NEHALEM_CONFIG_PATH = "../../snipersim/config/nehalem.cfg"  # SniperSim nehalem config file
GPGPU_CONFIG_PATH = "../../gpgpu-sim/configs/tested-cfgs/SM2_GTX480/gpgpusim.config"  # GPGPU-Sim config file
SNIPER_OUTPUT_PATH = "./proc_r1_p1_t3/sim.out"  # SniperSim performance output
POWER_OUTPUT_PATH = "./proc_r1_p1_t3/power.txt"  # SniperSim power output
GPGPU_OUTPUT_PATH = "./proc_r1_p1_t0/gpgpusim.0.1.log"  # GPGPU-Sim performance output
POWER_REPORT_PATTERN = "./proc_r1_p1_t0/gpgpusim_power_report*"  # GPGPU-Sim power report
LOG_FILE_PATH = "random_search_log.csv"  # Optimization log file

# Random Search parameters
ITERATIONS = 50  # Number of iterations for random search

# Extract SniperSim metrics
def extract_sniper_metrics(file_path):
    """
    Extract performance metrics from SniperSim output.
    :param file_path: Path to the simulation output file
    :return: Dictionary containing IPC, Instructions, Cycles, and Time (ns)
    """
    metrics = {"IPC": None, "Instructions": None, "Cycles": None, "Time (ns)": None}
    with open(file_path, 'r') as file:
        for line in file:
            if "IPC" in line:
                metrics["IPC"] = float(line.split("|")[-1].strip())
            if "Instructions" in line:
                metrics["Instructions"] = int(line.split("|")[-1].strip())
            if "Cycles" in line:
                metrics["Cycles"] = int(line.split("|")[-1].strip())
            if "Time (ns)" in line:
                metrics["Time (ns)"] = int(line.split("|")[-1].strip())
    if all(value is not None for value in metrics.values()):
        return metrics
    else:
        raise ValueError("Failed to extract complete performance metrics.")

# Extract SniperSim power
def extract_sniper_power(file_path):
    """
    Extract total power from SniperSim power output file.
    :param file_path: Path to the power output file
    :return: Total power (in W)
    """
    runtime_dynamic_power = None
    subthreshold_leakage = None
    gate_leakage = None

    with open(file_path, 'r') as file:
        for line in file:
            if "Runtime Dynamic" in line and "W" in line:
                runtime_dynamic_power = float(line.split()[-2])
            if "Subthreshold Leakage" in line and "W" in line and "with power gating" not in line:
                subthreshold_leakage = float(line.split()[-2])
            if "Gate Leakage" in line and "W" in line:
                gate_leakage = float(line.split()[-2])
            if runtime_dynamic_power is not None and subthreshold_leakage is not None and gate_leakage is not None:
                break

    if runtime_dynamic_power is not None and subthreshold_leakage is not None and gate_leakage is not None:
        total_power = runtime_dynamic_power + subthreshold_leakage + gate_leakage
        return total_power
    else:
        raise ValueError("Failed to extract complete power information.")

# Extract GPGPU-Sim IPC
def extract_gpgpu_ipc(log_file):
    """
    Extract IPC from GPGPU-Sim log file.
    :param log_file: Path to the log file
    :return: Total IPC
    """
    with open(log_file, "r") as file:
        for line in file:
            if "gpu_tot_ipc" in line:
                return float(line.split("=")[-1].strip())
    raise ValueError("IPC not found in log file.")

# Extract GPGPU-Sim power
def extract_gpgpu_power(power_report_pattern):
    """
    Extract total power from GPGPU-Sim power report.
    :param power_report_pattern: File path pattern
    :return: Total power
    """
    power_files = glob.glob(power_report_pattern)
    if not power_files:
        raise FileNotFoundError("No power report files found.")
    for file_path in power_files:
        with open(file_path, "r") as file:
            for line in file:
                if "gpu_tot_avg_power" in line:
                    return float(line.split("=")[-1].strip())
    raise ValueError("Power not found in power report files.")

# Modify SniperSim configuration
def modify_sniper_config(individual):
    """
    Modify SniperSim configuration files.
    :param individual: Parameter combination
    """
    params = dict(zip(sniper_parameter_ranges.keys(), individual))
    config_gainestown = configparser.ConfigParser()
    config_nehalem = configparser.ConfigParser()
    config_gainestown.read(SNIPER_CONFIG_PATH)
    config_nehalem.read(NEHALEM_CONFIG_PATH)

    # Modify gainestown.cfg
    config_gainestown.set("perf_model/core", "frequency", str(params["frequency"]))
    config_gainestown.set("perf_model/l3_cache", "cache_size", str(params["l3_cache_size"]))
    config_gainestown.set("perf_model/l3_cache", "shared_cores", str(params["l3_shared_cores"]))
    config_gainestown.set("perf_model/dram", "per_controller_bandwidth", str(params["per_controller_bandwidth"]))

    # Modify nehalem.cfg
    config_nehalem.set("perf_model/core", "logical_cpus", str(params["logical_cpus"]))
    config_nehalem.set("perf_model/l1_icache", "cache_size", str(params["l1_icache_size"]))
    config_nehalem.set("perf_model/l1_icache", "shared_cores", str(params["l1_shared_cores"]))
    config_nehalem.set("perf_model/l1_dcache", "cache_size", str(params["l1_dcache_size"]))
    config_nehalem.set("perf_model/l1_dcache", "shared_cores", str(params["l1_shared_cores"]))
    config_nehalem.set("perf_model/l2_cache", "cache_size", str(params["l2_cache_size"]))
    config_nehalem.set("perf_model/l2_cache", "shared_cores", str(params["l2_shared_cores"]))

    # Save modified configuration files
    with open(SNIPER_CONFIG_PATH, "w") as gainestown_file:
        gainestown_file.write("#include nehalem\n")  # Add #include nehalem
        config_gainestown.write(gainestown_file)

    with open(NEHALEM_CONFIG_PATH, "w") as nehalem_file:
        config_nehalem.write(nehalem_file)

# Modify GPGPU-Sim configuration
def modify_gpgpu_config(individual):
    """
    Modify GPGPU-Sim configuration file.
    :param individual: Parameter combination
    """
    config = dict(zip(gpgpu_parameter_ranges.keys(), individual))
    with open(GPGPU_CONFIG_PATH, "r") as config_file:
        lines = config_file.readlines()
    with open(GPGPU_CONFIG_PATH, "w") as config_file:
        for line in lines:
            modified = False
            for key, value in config.items():
                if line.strip().startswith(key):
                    config_file.write(f"{key} = {value}\n")
                    modified = True
                    break
            if not modified:
                config_file.write(line)

# Run simulation
def run_simulation():
    """
    Run the combined simulation (CPU and GPU).
    """
    subprocess.run([CHIPLET_SIMULATOR_PATH, INPUT_FILE], check=True)

# Fitness function
def evaluate_fitness(individual):
    """
    Evaluate the fitness of an individual considering both CPU and GPU IPC and power.
    :param individual: Parameter combination
    :return: Fitness value, CPU IPC, GPU IPC, CPU power, GPU power, and execution time
    """
    # Modify configuration files for both CPU and GPU
    modify_sniper_config(individual)  # Modify CPU configuration
    modify_gpgpu_config(individual)   # Modify GPU configuration

    # Run the simulation
    start_time = time.time()
    run_simulation()  # Run the combined simulation
    execution_time = int((time.time() - start_time) * 1000)  # Execution time (ms)

    # Extract CPU metrics
    try:
        cpu_metrics = extract_sniper_metrics(SNIPER_OUTPUT_PATH)
        cpu_ipc = cpu_metrics["IPC"]
        cpu_power = extract_sniper_power(POWER_OUTPUT_PATH)  # Extract CPU power
    except Exception as e:
        print(f"Error extracting CPU metrics: {e}")
        cpu_ipc, cpu_power = 0, 0

    # Extract GPU metrics
    try:
        gpu_ipc = extract_gpgpu_ipc(GPGPU_OUTPUT_PATH)
        gpu_power = extract_gpgpu_power(POWER_REPORT_PATTERN)
    except Exception as e:
        print(f"Error extracting GPU metrics: {e}")
        gpu_ipc, gpu_power = 0, 0

    # Define weights for IPC and power
    alpha_cpu = 1.0    # Weight for CPU IPC
    alpha_gpu = 48.0   # Weight for GPU IPC (scaled to match CPU IPC contribution)
    beta_cpu = 0.01    # Weight for CPU power
    beta_gpu = 0.0067  # Weight for GPU power (scaled to match CPU power contribution)

    # Calculate fitness
    fitness = (alpha_cpu * cpu_ipc + alpha_gpu * gpu_ipc) - (beta_cpu * cpu_power + beta_gpu * gpu_power)

    # Clean up simulation output files
    matches = glob.glob('proc*')
    for path in matches:
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Error deleting {path}: {e}")

    return fitness, cpu_ipc, gpu_ipc, cpu_power, gpu_power, execution_time

# Random Search main function
def random_search():
    """
    Main function of the random search algorithm.
    :return: Best individual
    """
    # Initialize log file
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("Iteration,Best Individual (CPU),Best Individual (GPU),CPU IPC,GPU IPC,CPU Power (W),GPU Power (W),Fitness,Execution Time (ms)\n")

    # Initialize best fitness and individual
    best_fitness = -float("inf")
    best_individual_cpu = None
    best_individual_gpu = None

    for iteration in range(ITERATIONS):
        print(f"Iteration {iteration + 1}")

        # Generate a random individual
        individual_cpu = [random.choice(values) for values in sniper_parameter_ranges.values()]
        individual_gpu = [random.choice(values) for values in gpgpu_parameter_ranges.values()]

        # Evaluate fitness
        fitness, cpu_ipc, gpu_ipc, cpu_power, gpu_power, exec_time = evaluate_fitness(individual_cpu)

        # Update best individual
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual_cpu = individual_cpu
            best_individual_gpu = individual_gpu

        # Log results
        with open(LOG_FILE_PATH, "a") as log_file:
            log_file.write(
                f"{iteration + 1},{best_individual_cpu},{best_individual_gpu},{cpu_ipc},{gpu_ipc},{cpu_power},{gpu_power},{fitness},{exec_time}\n"
            )

        # Print current best individual
        print(
            f"Best Individual (CPU): {best_individual_cpu}, Best Individual (GPU): {best_individual_gpu}, "
            f"CPU IPC: {cpu_ipc}, GPU IPC: {gpu_ipc}, CPU Power: {cpu_power} W, "
            f"GPU Power: {gpu_power} W, Fitness: {fitness}, Execution Time: {exec_time} ms"
        )

    return best_individual_cpu, best_individual_gpu

# Run random search
best_individual_cpu, best_individual_gpu = random_search()
print("Best CPU parameter combination:", best_individual_cpu)
print("Best GPU parameter combination:", best_individual_gpu)