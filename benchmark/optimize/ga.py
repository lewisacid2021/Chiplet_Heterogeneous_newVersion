import random
import subprocess
import time
import glob
import shutil
import os
import configparser
import csv

# Define combined parameter ranges (CPU + GPU)
combined_parameter_ranges = {
    # CPU Parameters
    "cpu_frequency": [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
    "cpu_logical_cpus": [1, 2, 4],
    "cpu_l1_icache_size": [32, 64, 128, 256],
    "cpu_l1_dcache_size": [32, 64, 128, 256],
    "cpu_l2_cache_size": [256, 512, 1024, 2048],
    "cpu_l3_cache_size": [8192, 16384, 32768, 65536],
    "cpu_per_controller_bandwidth": [7.6, 15.2, 30.4, 60.8],
    "cpu_l1_shared_cores": [1, 2, 4, 8, 16],
    "cpu_l2_shared_cores": [1, 2, 4, 8, 16],
    "cpu_l3_shared_cores": [1, 2, 4, 8, 16],
    
    # GPU Parameters
    "gpu_n_clusters": list(range(1, 20)),
    "gpu_n_cores_per_cluster": list(range(1, 5)),
    "gpu_shader_registers": [16384, 32768, 49152, 65536],
    "gpu_shmem_size": [16384, 32768, 49152, 65536],
    "gpu_cache_dl1_sets": [16, 32, 48, 64],
    "gpu_cache_dl2_sets": [32, 64, 96, 128],
    "gpu_num_sp_units": list(range(1, 5)),
    "gpu_num_sfu_units": list(range(1, 5)),
    "gpu_clock_domains_core": list(range(500, 1000, 100)),
}

# 定义路径配置
INPUT_FILE = "../matmul/matmul.yml"
CHIPLET_SIMULATOR_PATH = "../../interchiplet/bin/interchiplet"
SNIPER_CONFIG_PATH = "../../snipersim/config/gainestown.cfg"
NEHALEM_CONFIG_PATH = "../../snipersim/config/nehalem.cfg"
GPGPU_CONFIG_PATH = "../../gpgpu-sim/configs/tested-cfgs/SM2_GTX480/gpgpusim.config"
SNIPER_OUTPUT_PATH = "./proc_r1_p1_t3/sim.out"
POWER_OUTPUT_PATH = "./proc_r1_p1_t3/power.txt"
GPGPU_OUTPUT_PATH = "./proc_r1_p1_t0/gpgpusim.0.1.log"
POWER_REPORT_PATTERN = "./proc_r1_p1_t0/gpgpusim_power_report*"
LOG_FILE_PATH = "optimization_log.csv"
BEST_INDIVIDUAL_LOG = "generation_best.csv"  # 每代最优个体记录文件

# Genetic algorithm parameters
POPULATION_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.05
MAX_NO_IMPROVEMENT = 5

# Extract SniperSim metrics
def extract_sniper_metrics(file_path):
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
    with open(log_file, "r") as file:
        for line in file:
            if "gpu_tot_ipc" in line:
                return float(line.split("=")[-1].strip())
    raise ValueError("IPC not found in log file.")

# Extract GPGPU-Sim power
def extract_gpgpu_power(power_report_pattern):
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
def modify_sniper_config(cpu_params):
    config_gainestown = configparser.ConfigParser()
    config_nehalem = configparser.ConfigParser()
    config_gainestown.read(SNIPER_CONFIG_PATH)
    config_nehalem.read(NEHALEM_CONFIG_PATH)

    config_gainestown.set("perf_model/core", "frequency", str(cpu_params["cpu_frequency"]))
    config_gainestown.set("perf_model/l3_cache", "cache_size", str(cpu_params["cpu_l3_cache_size"]))
    config_gainestown.set("perf_model/l3_cache", "shared_cores", str(cpu_params["cpu_l3_shared_cores"]))
    config_gainestown.set("perf_model/dram", "per_controller_bandwidth", str(cpu_params["cpu_per_controller_bandwidth"]))

    config_nehalem.set("perf_model/core", "logical_cpus", str(cpu_params["cpu_logical_cpus"]))
    config_nehalem.set("perf_model/l1_icache", "cache_size", str(cpu_params["cpu_l1_icache_size"]))
    config_nehalem.set("perf_model/l1_icache", "shared_cores", str(cpu_params["cpu_l1_shared_cores"]))
    config_nehalem.set("perf_model/l1_dcache", "cache_size", str(cpu_params["cpu_l1_dcache_size"]))
    config_nehalem.set("perf_model/l1_dcache", "shared_cores", str(cpu_params["cpu_l1_shared_cores"]))
    config_nehalem.set("perf_model/l2_cache", "cache_size", str(cpu_params["cpu_l2_cache_size"]))
    config_nehalem.set("perf_model/l2_cache", "shared_cores", str(cpu_params["cpu_l2_shared_cores"]))

    with open(SNIPER_CONFIG_PATH, "w") as f:
        f.write("#include nehalem\n")
        config_gainestown.write(f)
    with open(NEHALEM_CONFIG_PATH, "w") as f:
        config_nehalem.write(f)

# Modify GPGPU-Sim configuration
def modify_gpgpu_config(gpu_params):
    config_mapping = {
        "gpu_n_clusters": "gpgpu_n_clusters",
        "gpu_n_cores_per_cluster": "gpgpu_n_cores_per_cluster",
        "gpu_shader_registers": "gpgpu_shader_registers",
        "gpu_shmem_size": "gpgpu_shmem_size",
        "gpu_cache_dl1_sets": "gpgpu_cache_dl1_sets",
        "gpu_cache_dl2_sets": "gpgpu_cache_dl2_sets",
        "gpu_num_sp_units": "gpgpu_num_sp_units",
        "gpu_num_sfu_units": "gpgpu_num_sfu_units",
        "gpu_clock_domains_core": "gpgpu_clock_domains_core"
    }

    with open(GPGPU_CONFIG_PATH, "r") as f:
        lines = f.readlines()

    with open(GPGPU_CONFIG_PATH, "w") as f:
        for line in lines:
            updated = False
            for param, config_key in config_mapping.items():
                if line.strip().startswith(config_key):
                    f.write(f"{config_key} = {gpu_params[param]}\n")
                    updated = True
                    break
            if not updated:
                f.write(line)

# Run simulation
def run_simulation():
    subprocess.run([CHIPLET_SIMULATOR_PATH, INPUT_FILE], check=True)

# Fitness function
def evaluate_fitness(individual):
    param_keys = list(combined_parameter_ranges.keys())
    cpu_params = dict(zip(
        [k for k in param_keys if k.startswith("cpu_")],
        individual[:len(param_keys)//2]
    ))
    gpu_params = dict(zip(
        [k for k in param_keys if k.startswith("gpu_")],
        individual[len(param_keys)//2:]
    ))

    modify_sniper_config(cpu_params)
    modify_gpgpu_config(gpu_params)

    start_time = time.time()
    run_simulation()
    execution_time = int((time.time() - start_time) * 1000)

    try:
        cpu_metrics = extract_sniper_metrics(SNIPER_OUTPUT_PATH)
        cpu_ipc = cpu_metrics["IPC"]
        cpu_power = extract_sniper_power(POWER_OUTPUT_PATH)
    except Exception as e:
        print(f"Error extracting CPU metrics: {e}")
        cpu_ipc, cpu_power = 0, 0

    try:
        gpu_ipc = extract_gpgpu_ipc(GPGPU_OUTPUT_PATH)
        gpu_power = extract_gpgpu_power(POWER_REPORT_PATTERN)
    except Exception as e:
        print(f"Error extracting GPU metrics: {e}")
        gpu_ipc, gpu_power = 0, 0

    alpha_cpu = 1.0
    alpha_gpu = 48.0
    beta_cpu = 0.01
    beta_gpu = 0.0067

    fitness = (alpha_cpu * cpu_ipc + alpha_gpu * gpu_ipc) - (beta_cpu * cpu_power + beta_gpu * gpu_power)

    matches = glob.glob('proc*')
    for path in matches:
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Error deleting {path}: {e}")

    return {
        "fitness": fitness,
        "cpu_ipc": cpu_ipc,
        "gpu_ipc": gpu_ipc,
        "cpu_power": cpu_power,
        "gpu_power": gpu_power,
        "exec_time": execution_time,
        "params": individual
    }

def log_generation_best(generation, best_info):
    header = [
        "Generation", "Fitness", "CPU_IPC", "GPU_IPC",
        "CPU_Power(W)", "GPU_Power(W)", "Exec_Time(ms)",
        "CPU_Parameters", "GPU_Parameters"
    ]
    
    param_keys = list(combined_parameter_ranges.keys())
    cpu_params = dict(zip(
        [k for k in param_keys if k.startswith("cpu_")],
        best_info["params"][:len(param_keys)//2]
    ))
    gpu_params = dict(zip(
        [k for k in param_keys if k.startswith("gpu_")],
        best_info["params"][len(param_keys)//2:]
    ))

    row = [
        generation,
        best_info["fitness"],
        best_info["cpu_ipc"],
        best_info["gpu_ipc"],
        best_info["cpu_power"],
        best_info["gpu_power"],
        best_info["exec_time"],
        str(cpu_params),
        str(gpu_params)
    ]

    with open(BEST_INDIVIDUAL_LOG, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if generation == 0:
            writer.writerow(header)
        writer.writerow(row)

# 初始化种群
def initialize_population():
    return [
        [random.choice(values) for values in combined_parameter_ranges.values()]
        for _ in range(POPULATION_SIZE)
    ]

# 遗传算法主函数
def genetic_algorithm():
    population = initialize_population()
    best_fitness = -float("inf")
    best_individual = None

    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}")

        fitness_scores = []
        best_info = None
        for individual in population:
            result = evaluate_fitness(individual)
            fitness_scores.append(result["fitness"])
            if result["fitness"] > best_fitness:
                best_fitness = result["fitness"]
                best_individual = individual
                best_info = result

        # 记录每代最优个体
        log_generation_best(generation + 1, best_info)

        # 早停机制
        if (generation > 0 and 
            best_fitness == fitness_scores[0] and 
            any(score == best_fitness for score in fitness_scores)):
            no_improvement_count += 1
            if no_improvement_count >= MAX_NO_IMPROVEMENT:
                print(f"No improvement for {MAX_NO_IMPROVEMENT} generations. Stopping early.")
                break
        else:
            no_improvement_count = 0

        # 遗传操作
        new_population = [best_individual]
        while len(new_population) < POPULATION_SIZE:
            parents = [
                max(random.sample(list(zip(population, fitness_scores)), 3), key=lambda x: x[1])[0]
                for _ in range(2)
            ]
            
            crossover_points = sorted(random.sample(range(len(combined_parameter_ranges)), 2))
            child = (
                parents[0][:crossover_points[0]] +
                parents[1][crossover_points[0]:crossover_points[1]] +
                parents[0][crossover_points[1]:]
            )
            
            mutation_rate = MUTATION_RATE * (1 - generation / GENERATIONS)
            child = [
                random.choice(list(combined_parameter_ranges.values())[i])
                if random.random() < mutation_rate
                else gene
                for i, gene in enumerate(child)
            ]
            
            new_population.append(child)

        population = new_population[:POPULATION_SIZE]

    return best_individual

# 运行优化
best_params = genetic_algorithm()
print("Optimized Parameters:", dict(zip(combined_parameter_ranges.keys(), best_params)))