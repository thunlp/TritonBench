import os
import json
import argparse

golden_metrics_folder = "./golden_metrics"
golden_metrics_list = os.listdir(golden_metrics_folder)

def write_file(input_folder_path, results_path):
    if os.path.exists('./tmp'):
        os.system('rm -rf ./tmp')
    os.mkdir('./tmp')
    if os.path.exists('./logs'):
        os.system('rm -rf ./logs')
    os.mkdir('./logs')
    if os.path.exists(results_path):
        os.system(f'rm -rf {results_path}')
    os.mkdir(results_path)

    tab = ' ' * 4
    with open('./performance_utils.py', 'r') as f:
        performance_utils = f.readlines()
    # performance_utils = performance_utils.replace('folder_path = "/home/lishangzhan/triton/bench_performance/results"', f'folder_path = "{results_path}"')
    performance_utils_lines = []
    for line in performance_utils:
        if 'folder_path = ' in line:
            line = tab * 2 + f'folder_path = "{results_path}"\n'
        performance_utils_lines.append(line)
    performance_utils = "".join(performance_utils_lines)
    with open('./performance_utils.py', 'w') as f:
        f.write(performance_utils)
    input_file_list = os.listdir(input_folder_path)
    for file in input_file_list:
        if file[-3:] == ".py":
            op = file[:-3]
            perf_file_name = op + "_perf.py"
            assert perf_file_name in golden_metrics_list, f"{perf_file_name} not in golden_metrics_list"
            with open(os.path.join(golden_metrics_folder, perf_file_name), "r") as f:
                # golden_metrics = f.read()
                lines = f.readlines()
                # print(lines)
                updated_lines = []
                for line in lines:
                    if line == "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n":
                        updated_lines.append(f"sys.path.append('{input_folder_path}')\n")
                    line = line.replace("from TritonBench_v1.", "from ")
                    line = line.replace("op_perf.get_do_bench_config()", "op_perf.get_do_bench_config(warmup=100, rep=1000)")
                    line = line.replace('folder_path = "/home/lishangzhan/triton/bench_performance/results"', f'folder_path = "{results_path}"')
                    updated_lines.append(line)
                golden_metrics = "".join(updated_lines)
            
            golden_metrics_lines = golden_metrics.split("\n")
            flag = False
            for i in range(len(golden_metrics_lines)):
                if "input_tensor = self.to_cuda(input_tensor_)" in golden_metrics_lines[i]:
                    index_1 = i
                if "results.append(result)" in golden_metrics_lines[i]:
                    index_2 = i + 1
                    flag = True
            if flag:
                
                for i in range(index_1, index_2):
                    golden_metrics_lines[i] = tab + golden_metrics_lines[i]                
                golden_metrics_lines.insert(index_1, tab*3 + "try:")
                golden_metrics_lines.insert(index_2 + 1, tab*3 + "except Exception as e:")
                golden_metrics_lines.insert(index_2 + 2, tab*4 + 'print(f"Failed to run benchmark for input tensor. Error: {e}")')
                golden_metrics = "\n".join(golden_metrics_lines)
            
            with open(os.path.join('./tmp', perf_file_name), "w") as f:
                f.write(golden_metrics)
            
def parse_args():
    parser = argparse.ArgumentParser(description='write_file')
    parser.add_argument('--input_folder_path', type=str, help='input_folder_path')
    parser.add_argument('--result_folder_path', type=str, help='result_folder_path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input_folder_path = args.input_folder_path
    results_path = args.result_folder_path
    write_file(input_folder_path, results_path)