import os
import subprocess
from multiprocessing import Pool, Lock, Value
from tqdm import tqdm
import signal

script_dir = "./tmp"
log_dir = "./logs"
gpu_count = 8
timeout = 600

scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
scripts = [os.path.join(script_dir, script) for script in scripts]
total_scripts = len(scripts)

progress = Value('i', 0)
progress_lock = Lock()

def run_script(args):
    gpu_id, script = args

    script_name = os.path.basename(script)
    log_file = os.path.join(log_dir, f"{script_name}.log")
    err_file = os.path.join(log_dir, f"{script_name}.err")

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script}"
    # print(f"Running: {cmd}")

    with open(log_file, "w") as log, open(err_file, "w") as err:
        process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=err)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # process.kill()
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            tqdm.write(f"⏱️ timeout，killed {script_name} ( {timeout} s)")

    with progress_lock:
        progress.value += 1
        tqdm.write(f"✅ finished {progress.value}/{total_scripts}: {script_name}")

os.makedirs(log_dir, exist_ok=True)

if __name__ == "__main__":
    with Pool(processes=gpu_count) as pool, tqdm(total=total_scripts, desc="Process", ncols=80) as pbar:
        args_list = [(i % gpu_count, scripts[i]) for i in range(total_scripts)]
        
        for _ in pool.imap(run_script, args_list):
            pbar.update(1)

        pool.close()
        pool.join()