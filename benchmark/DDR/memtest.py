import yaml
import subprocess
import os
import shutil
import time

# 输入文件路径
input_file = "ddr.yml"

# 需要测试的 .ini 文件列表
ini_files = [
    "../configs/DDR4_8Gb_x8_2400.ini",
    "../configs/DDR3_8Gb_x8_1600.ini",
    "../configs/GDDR5_8Gb_x32.ini",
    "../configs/GDDR5X_8Gb_x32.ini",
    "../configs/GDDR6_8Gb_x16.ini",
    "../configs/HMC2_8GB_4Lx16.ini",
    "../configs/LPDDR3_8Gb_x32_1600.ini",
    "../configs/LPDDR4_8Gb_x16_2400.ini",
    "../configs/HBM2_8Gb_x128.ini",
    # 添加更多 .ini 文件路径
]

# 实验数据目录
data_dir = "./proc_r1_p1_t1"  # 实验程序生成的数据目录
backup_dir = "./backup_data"  # 备份数据的目录

# 加载输入文件
with open(input_file, "r") as f:
    data = yaml.safe_load(f)

# 确保备份目录存在
os.makedirs(backup_dir, exist_ok=True)

# 遍历每个 .ini 文件进行测试
for ini_file in ini_files:
    print(f"Testing with config: {ini_file}")

    # 修改 args 中的 .ini 文件路径
    for phase in data["phase1"]:
        if "ddr_mem" in phase["cmd"]:  # 找到 ddr_mem 的配置
            phase["args"][1] = ini_file  # 修改 .ini 文件路径

    # 将修改后的内容写回输入文件
    with open(input_file, "w") as f:
        yaml.safe_dump(data, f)

    # 记录开始时间
    start_time = time.time()

    # 运行实验程序
    try:
        # 使用 universal_newlines=True 替代 text=True
        result = subprocess.run(
            ["../../interchiplet/bin/interchiplet", input_file],
            check=True,
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,  # 捕获标准错误
            universal_newlines=True  # 以文本模式返回输出
        )
        print("Experiment program output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Experiment program failed with error: {e.stderr}")

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Experiment completed in {elapsed_time:.2f} seconds.")

    # 备份实验数据
    backup_subdir = os.path.join(backup_dir, os.path.basename(ini_file).replace(".ini", ""))
    os.makedirs(backup_subdir, exist_ok=True)
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path):
            shutil.copy2(item_path, backup_subdir)
        elif os.path.isdir(item_path):
            shutil.copytree(item_path, os.path.join(backup_subdir, item))

    # 将测试结果保存到备份子目录中
    log_file = os.path.join(backup_subdir, f"test_result_{os.path.basename(ini_file)}.log")
    with open(log_file, "w") as f:
        f.write(f"Tested with config: {ini_file}\n")
        f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
        f.write("Experiment program output:\n")
        f.write(result.stdout)

    print(f"Experiment data and test result backed up to: {backup_subdir}")

print("All tests completed.")