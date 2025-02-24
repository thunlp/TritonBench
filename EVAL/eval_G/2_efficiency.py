import os
import json
import argparse

ref_folder = "../../performance_metrics/perf_G/golden_results"


def calculate(path_gen, path_ref):
    get_ms = lambda data: [item["ms"] for item in data]
    get_gbs = lambda data: [item["GB/s"] for item in data]
    get_tflops = lambda data: [item["TFLOPS"] for item in data]
    avg = lambda mss: round(sum(mss[0]) / sum(mss[1]), 4)

    data_gen = json.loads(open(path_gen, 'r', encoding='utf-8').read())
    data_ref = json.loads(open(path_ref, 'r', encoding='utf-8').read())
    assert len(data_gen) == len(data_ref), ""
    
    ms_ref, ms_gen = get_ms(data_ref), get_ms(data_gen)
    ms = avg((ms_ref, ms_gen))


    efficiency = max(round(max(get_gbs(data_gen)) * 100 / 2039, 4), round(max(get_tflops(data_gen)) * 100 / 312, 4))
    efficiency1 = max(round(max(get_gbs(data_ref)) * 100 / 2039, 4), round(max(get_tflops(data_ref)) * 100 / 312, 4))
    if efficiency >= 100 or ms >= 10:
        assert False, f"{path_gen.split('/')[-1]} test failed!"
    if efficiency1 > efficiency:
        print(f"金标好啊好11111: {efficiency} < {efficiency1}")
    else:
        print(f"生成棒棒棒！！！: {efficiency} > {efficiency1}")
    return ms, efficiency

def statis(gen_folder):
    avg = lambda listt: round(sum(listt) / len(listt), 2)
    files = [f for f in os.listdir(gen_folder) if f.endswith(".json")]
    spdups, effcys = [], []
    print("===="*40)
    for f in files:
        path_gen = os.path.join(gen_folder, f)
        path_ref = os.path.join(ref_folder, f)
        
        try:
            ms, efficiency = calculate(path_gen, path_ref)
            print(f"{f}: {ms}")
            print(f"{f}: {efficiency}\n")
            spdups.append(ms)
            effcys.append(efficiency)
        except:
            print(f"{f} failed")

    print(spdups)
    print(f"\n{gen_folder.split('/')[-1]}")
    print(f"speed up: {avg(spdups)}")
    print(f"efficiency: {avg(effcys)}")
    print("===="*40)

def arg_parser():
    parser = argparse.ArgumentParser(description='Efficiency statistics')
    parser.add_argument('--gen_folder', type=str, required=True, help='The generated folder path')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    gen_folder = args.gen_folder
    statis(gen_folder)
    # root = "/home/lijianling/TritonLLM-g/triton_bench_build/EVAL0/gene_perf/"
    # for gen_folder in os.listdir(root):
    # # for gen_folder in ["results_modified_output_claude-3-5-sonnet-20240620_comp",]:
    #     statis(root + gen_folder)

    # gen_folder = "/home/lijianling/TritonLLM-g/triton_bench_build/EVAL0/gene_perf/results_modified_output_o1-2024-12-17_comp/"    
    # statis(gen_folder)

    # calculate(path, ref_folder+"adam_update_triton.json")