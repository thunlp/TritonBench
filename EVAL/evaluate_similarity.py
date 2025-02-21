import tempfile
import os
import json
import subprocess
import shutil
import argparse
import re


def eval_similarity(in_file, res_file, bleu_params="0.25,0.25,0.25,0.25"):
    # 1. 获取处理llamafactory生成jsonl数据
    if in_file.endswith(".jsonl"):
        data = [json.loads(line) for line in open(in_file, 'r', encoding='utf-8').readlines()]
    else:
        data = json.load(open(in_file, 'r', encoding='utf-8'))
    
    def select_code(text):
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        code = "\n\n".join(code_blocks)
        if code == '':
            code = text
        return code
        
    goods, preds = [], []
    for item in data:
        goods.append(item['label'])
        preds.append(select_code(item['predict']))

    def write_lines(lines, path, keptn=True):
        with open(path, 'w', encoding='utf-8') as W:
            for line in lines:
                if keptn: line = line.replace('\r', '').replace('\n', '\\n')
                W.write(line + '\n')
    
    # 2.将每行的 `label` 和 `predict` 分别保存为 gold.txt 和 pred.txt，存储在一个临时文件夹。
    def save_py_and_black_and_read(codes: list, folder):
        for i, code in enumerate(codes):
            file_path = folder + f"/{i}.py"
            with open(file_path, 'w', encoding='utf-8') as outfile:
                outfile.write(code.replace('\r', ''))
            subprocess.run(["/home/lijianling/miniconda3/envs/LLM/bin/black", file_path], check=True)

        texts = []
        for i in range(len(codes)):
            file_path = folder + f"/{i}.py"
            text = open(file_path, 'r', encoding='utf-8').read()
            texts.append(text)
        return texts

    temp_dir = tempfile.mkdtemp(suffix="Triton_")

    # if black:
    #     goods = save_py_and_black_and_read(goods, temp_dir)
    #     preds = save_py_and_black_and_read(preds, temp_dir)

    gold_file = os.path.join(temp_dir, "gold.txt")
    pred_file = os.path.join(temp_dir, "pred.txt")
    write_lines(goods, gold_file)
    write_lines(preds, pred_file)
    
    # 3.运行codebleu
    try:
        codebleu_folder = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/'
        command = [
            "/home/lijianling/miniconda3/envs/codebleu/bin/python", codebleu_folder + "/CodeBLEU/calc_code_bleu.py",
            "--refs", gold_file,
            "--hyp", pred_file,
            "--lang", 'python',
            "--params", bleu_params
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        # 输出标准输出和错误
        print(result.stdout)
        # print(result.stderr)

        # 检查是否成功
        if result.returncode == 0:
            write_lines([result.stdout], res_file, keptn=False)
            print(f"脚本执行成功，结果保存在: {res_file}")
        else:
            print(f"脚本执行失败，错误码: {result.returncode}")
        
    except Exception as e:
        print(f"执行脚本时发生错误: {e}")

    # 4. 删除临时文件夹
    shutil.rmtree(temp_dir)
    print("临时文件夹已删除")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="codebleu")
    parser.add_argument(
        "--generate-folder",  # 参数名称
        type=str,  # 参数类型
        help="待评测的文件夹路径"  # 参数说明
    )
    parser.add_argument(
        "--result-folder",  # 参数名称
        type=str,  # 参数类型
        help="待评测的文件夹路径"  # 参数说明
    )
    args = parser.parse_args()
    gen_folder = args.generate_folder
    res_folder = args.result_folder

    for f in os.listdir(gen_folder):
        if f.endswith(".jsonl") or f.endswith(".json"):
            in_file = os.path.join(gen_folder, f)
            res_file = os.path.join(res_folder, f.replace(".jsonl", ".txt"))
            if os.path.exists(res_file):
                print(f"{f} already tested!")
            else:
                eval_similarity(in_file, res_file)


    # in_file = "/home/lijianling/TritonLLM-g/triton_bench_build/inference_output1/inference_results_original.jsonl"
    # res_file = "EVAL/scores/inference_results_original.txt"
    # eval_similarity(in_file, res_file)