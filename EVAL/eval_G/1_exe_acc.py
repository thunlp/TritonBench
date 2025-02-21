import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor


gold_folder = "data/TritonBench_G_v1/"
py_interpreter = "/home/lijianling/miniconda3/envs/LLM/bin/python"


def compare_python_files(file1, file2):
    """
    Compare the outputs of two Python files.
    Returns True if outputs are identical, False otherwise.
    """
    result1 = subprocess.run(
        [py_interpreter, file1],
        capture_output=True,
        text=True
    )
    output1 = result1.stdout

    result2 = subprocess.run(
        [py_interpreter, file2],
        capture_output=True,
        text=True
    )
    output2 = result2.stdout
    
    return output1 == output2, file1.split("/")[-1]  # True if identical, False otherwise

def test_close_parallel(llm_folder, gold_folder, gpus):
    files = [f for f in os.listdir(llm_folder) if f.endswith(".py")]
    # Track correct executions
    correct_count = 0
    total_count = len(files)

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []

        for idx, f in enumerate(files):
            file1 = os.path.join(llm_folder, f)
            file2 = os.path.join(gold_folder, f)

            # Set GPU device for each task (distribute across GPUs)
            gpu_id = gpus[idx % len(gpus)]
            futures.append(executor.submit(run_with_gpu, file1, file2, gpu_id))

        # Process the results
        for future in futures:
            is_correct, file_name = future.result()[0], future.result()[1]   # Get the comparison result (True/False)

            if is_correct:
                correct_count += 1
            else:
                file_path = os.path.join(llm_folder, file_name)
                os.remove(file_path)
                print(f"Deleted {file_name}", flush=True)
                # print(f"failed execution file: {file_name}", flush=True)

    # Calculate and print the correct execution rate
    correct_rate = (correct_count / total_count) * 100
    assert total_count == len(files), "error in files"
    print(f"\nCorrect execution rate: {correct_rate:.2f}% = {correct_count} / {total_count}", flush=True)

def run_with_gpu(file1, file2, gpu_id):
    # Set the GPU device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Compare the Python files
    return compare_python_files(file1, file2)  # Returns True/False

def execute_4folders(root_folder, gpus):
    for folder in os.listdir(root_folder):
        llm_folder = os.path.join(root_folder, folder)
        test_close_parallel(llm_folder, gold_folder, gpus)
        print(f"above is the compare execution for {folder}", flush=True)
        print("========"*30, flush=True)

def execute_4folder(llm_folder, gpus):
    test_close_parallel(llm_folder, gold_folder, gpus)
    print(f"above is the compare execution for {llm_folder}", flush=True)
    print("========"*30, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Call Triton-G operator.")
    parser.add_argument('--folder', type=str, required=True, help="root folder contains multiple test folders or just ont folder.")
    parser.add_argument('--GPUs', type=str, required=True, help="number of GPU available.")
    
    args = parser.parse_args()
    assert os.path.isdir(args.folder), ""
    py_files = [f for f in os.listdir(args.folder) if f.endswith('.py')]
    if py_files == 0:
        execute_4folders(args.folder, args.GPUs)
    else:
        execute_4folder(args.folder, args.GPUs)


if __name__ == "__main__":
    main()