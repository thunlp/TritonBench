import os
import subprocess
from multiprocessing import Pool, Lock, Value
from tqdm import tqdm

# 配置参数
script_dir = "./tmp"  # 你想运行的 Python 文件所在的目录
log_dir = "./logs"  # 日志目录
gpu_count = 8  # GPU 数量

# 获取所有 Python 文件（严格按顺序）
scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
scripts = [os.path.join(script_dir, script) for script in scripts]  # 转换为完整路径
total_scripts = len(scripts)  # 总任务数

# 进度条计数器
progress = Value('i', 0)
progress_lock = Lock()

def run_script(args):
    """运行单个 Python 脚本，并记录日志，同时更新进度条"""
    gpu_id, script = args

    script_name = os.path.basename(script)
    log_file = os.path.join(log_dir, f"{script_name}.log")
    err_file = os.path.join(log_dir, f"{script_name}.err")

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script}"
    # print(f"Running: {cmd}")

    with open(log_file, "w") as log, open(err_file, "w") as err:
        process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=err)
        process.wait()  # 等待进程完成

    # 更新进度条
    with progress_lock:
        progress.value += 1
        tqdm.write(f"✅ 完成 {progress.value}/{total_scripts}: {script_name}")

# 创建日志文件夹
os.makedirs(log_dir, exist_ok=True)

# 显示进度条
if __name__ == "__main__":
    with Pool(processes=gpu_count) as pool, tqdm(total=total_scripts, desc="任务进度", ncols=80) as pbar:
        # 按顺序分配 GPU
        args_list = [(i % gpu_count, scripts[i]) for i in range(total_scripts)]
        
        for _ in pool.imap(run_script, args_list):  # 严格按照脚本列表顺序执行
            pbar.update(1)  # 每完成一个任务，更新进度条

        pool.close()
        pool.join()  # 等待所有任务完成