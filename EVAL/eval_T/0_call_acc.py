import json
import os,argparse
import ast
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


statis_path = "data/TritonBench_T_v1.json"
py_folder = "data/TritonBench_G_v1/"
py_interpreter = "/home/lijianling/miniconda3/envs/LLM/bin/python"

def extract_functions_and_imports(code):
    # Parse the code into an AST
    tree = ast.parse(code)
    
    functions = []
    imports = []

    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        # Check if the node is a function definition
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        
        # Check if the node is an import statement
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # For 'import' and 'from ... import' statements, we store the original code text
            imports.append(ast.unparse(node))  # Using `unparse` for getting the code text

    return functions, imports

def clear_code(code: str) -> str:
    if "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")
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

def get_corresponding_files(instrus: list) -> list:
    infos = json.loads(open(statis_path, 'r', encoding='utf-8').read())
    files = []
    for instru in instrus:
        f = []
        assert "Functional Description: " in instru and "Wrapper Entry Information:" in instru, ""
        func = instru.split("Functional Description: ")[-1].split("Wrapper Entry Information:")[0].replace("\n", "")
        for item in infos:
            if func in item["description"].replace("\n", ""):
                f.append(item["file"])
        assert len(f) == 1, ""
        files.append(f[0])
    assert len(files) == len(instrus)
    return files

def get_codes_for_test(path):
    assert path.endswith(".jsonl"), ""
    data = [json.loads(line) for line in open(path, 'r', encoding='utf-8').readlines()]
    key = list(data[0].keys())[0]
    print(key)
    files = get_corresponding_files([item[key] for item in data])
    codes = [clear_code(item["predict"]) for item in data]
    tests = get_test(py_folder, files)

    return codes, tests, files

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
            executor.submit(run_script_on_gpu, p, t, f, tmp_dir, gpus[i % len(gpus)]): f
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

        pred, test, files = get_codes_for_test(generated_path)
        target_path = os.path.join(tgt_folder, f)

        run_code_parallel(pred, test, files, tmp_dir=target_path, delete=True, gpus=gpus)
        print(f"Above is call test for {f}")
        print("===="*20)

def call_4file(path, tgt_path, gpus=[0]):
    pred, test, files = get_codes_for_test(path)
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
