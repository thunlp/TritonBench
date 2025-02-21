import os
import json
import subprocess
import ast
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

statis_path = "data/TritonBench_G_v1.json"
py_folder = "data/TritonBench_G_v1"
py_interpreter = "/home/lijianling/miniconda3/envs/LLM/bin/python"

def process_code(code: str):
    if "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")
    
    try:
        tree = ast.parse(code)
        imports = []
        function_definitions = []

        # Traverse the AST to find import statements and function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Collect the import statements
                imports.append(ast.unparse(node))  # Convert the AST node back to code
            elif isinstance(node, ast.FunctionDef):
                # Collect function definitions
                function_code = ast.unparse(node)  # Get the Python code for the function
                function_definitions.append(function_code)

        return "\n".join(imports) + "\n\n" + "\n".join(function_definitions)

    except:
        return code

def get_test(folder: str, files: list) -> list[str]:
    test = []
    for f in files:
        path = os.path.join(folder, f)
        assert os.path.exists(path), f"{f} not exist!"
        
        code = open(path, "r", encoding="utf-8").read().split("#"*146)[-1]
        assert "def test_" in  code, ""
        test.append(code)
    
    assert len(files) == len(test)
    return test

def get_file_order(gold: list[str]):
    data = json.loads(open(statis_path, 'r', encoding='utf-8').read())
    assert len(data) == len(gold), ""

    files = []
    for g in gold:
        g = g.replace("<|im_end|>", "").replace("<|EOT|>", "")
        tmp = False
        for item in data:
            if g in item["output"]:
                files.append(item["file"])
                tmp = item
                break
        if tmp:
            data.remove(tmp)
        elif g[50:220] == 'as tl\n\nif triton.__version__ >= "2.1.0":\n    @triton.jit\n    def _fwd_kernel(\n        Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录':
                files.append("context_attn_nopad.py")
        else:
            assert False, ""

    assert len(data) == 1 and len(files) == len(gold), ""
    return files

def get_code(path: str) -> list[str]:
    assert path.endswith('jsonl'), ""
    data = [json.loads(line) for line in open(path, 'r', encoding='utf-8').readlines()]

    gold, pred = [], []
    for item in data:
        gold.append(item["label"])
        pred.append(item["predict"])
    
    files = get_file_order(gold)
    test = get_test(py_folder, files)

    assert len(pred) == len(test), ""
    return pred, test, files

def run_code(pred, test, files, tmp_dir=""):
    cnt = len(pred)
    os.makedirs(tmp_dir, exist_ok=True)
    
    correct_count = 0  # To count correctly executed tests

    for p, t, f in zip(pred, test, files):
        try:
            temp_path = os.path.join(tmp_dir, f)

            with open(temp_path, "w") as temp_file:
                temp_file.write(p + "\n" + "#" * 146 + "\n" + t)

            # Run the temporary Python file
            result = subprocess.run(
                [py_interpreter, temp_path], 
                capture_output=True, 
                text=True
            )

            # Output the execution results
            print(f"=== Output for {f} ===")
            print(result.stdout)

            print(f"=== Errors for {f} ===")
            print(result.stderr)

            # Check if the execution was successful (no errors in the stderr)
            if result.returncode == 0:
                correct_count += 1  # Increment correct count if no errors

        finally:
            pass
            # if os.path.exists(temp_path):
            #     os.remove(temp_path)

    # Calculate and print the correct execution rate
    correct_rate = (correct_count / cnt) * 100
    print(f"\nCorrect execution rate: {correct_rate:.2f}%")

def run_script_on_gpu(script_content, test_content, file_name, tmp_dir, gpu_id):
    """
    Runs a given Python script on a specified GPU.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    temp_path = os.path.join(tmp_dir, file_name)

    try:
        with open(temp_path, "w") as temp_file:
            temp_file.write(script_content + "\n" + "#" * 146 + "\n" + test_content)

        # Set GPU device for execution
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Run the temporary Python file
        result = subprocess.run(
            [py_interpreter, temp_path], 
            capture_output=True, 
            text=True,
            env=env
        )

        success = result.returncode == 0  # Determine if execution was successful

        # Output execution results
        print(f"=== Output for {file_name} on GPU {gpu_id} ===", flush=True)
        print(result.stdout, flush=True)

        print(f"=== Errors for {file_name} on GPU {gpu_id} ===", flush=True)
        print(result.stderr, flush=True)

        return success, file_name   # Return execution success status

    finally:
        pass
        # Uncomment to remove temp files after execution
        # if os.path.exists(temp_path):
        #     os.remove(temp_path)

def run_code_parallel(pred, test, files, tmp_dir="temp", gpus=[0, 1, 2, 3, 4, 5, 6, 7], delete=False):
    """
    Runs code in parallel across multiple GPUs, ensuring each GPU runs one script at a time.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    total_scripts = len(pred)
    correct_count = 0
    ok_save_files = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        future_to_file = {
            executor.submit(run_script_on_gpu, process_code(p), t, f, tmp_dir, gpus[i % len(gpus)]): f
            for i, (p, t, f) in enumerate(zip(pred, test, files))
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                success = future.result()[0]
                if success:
                    correct_count += 1
                    ok_save_files.append(future.result()[1])
            except Exception as e:
                print(f"Error processing {file_name}: {e}", flush=True)

    if delete:
        for file in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, file)
            
            if file not in ok_save_files:
                try:
                    # Remove the file
                    os.remove(file_path)
                    print(f"Deleted {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
    # Calculate and print the correct execution rate
    correct_rate = (correct_count / total_scripts) * 100
    print(f"\nCorrect execution rate: {correct_rate:.2f}%", flush=True)
    print(ok_save_files)

def call_4folder(folder, tgt_folder, gpus=[0, 1, 2, 3, 4, 5, 6, 7]):
    for f in os.listdir(folder):
        if not f.endswith(".jsonl"):
            continue
        generated_path = os.path.join(folder, f)

        pred, test, files = get_code(generated_path)
        target_path = os.path.join(tgt_folder, f)

        run_code_parallel(pred, test, files, tmp_dir=target_path, delete=True, gpus=gpus)
        print(f"Above is call test for {f}")
        print("===="*20)

def call_4file(path, tgt_path, gpus=[0]):
    pred, test, files = get_code(path)
    run_code_parallel(pred, test, files, tmp_dir=tgt_path, delete=True, gpus=gpus)
    print(f"Above is call test for {path.split("/")[-1].replace(".jsonl", "")}")
    print("===="*40)

def main():
    parser = argparse.ArgumentParser(description="Call Triton-G operator.")
    parser.add_argument('--source', type=str, required=True, help="Source directory or jsonl file for test.")
    parser.add_argument('--target', type=str, required=True, help="Target directory to save the output.")
    parser.add_argument('--GPUs', type=str, required=True, help="number of GPUs available.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.source):
        call_4folder(args.source, args.target, args.GPUs)
    else:
        assert os.path.isfile(args.source), ""
        call_4file(args.source, args.target, args.GPUs)

if __name__ == "__main__":
    main()