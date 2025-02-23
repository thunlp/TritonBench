import os
import json
import argparse

ref_folder = "../../performance_metrics/perf_T/golden_results"


def calculate(path_gen, path_ref):
    get_ms = lambda data: [item["ms"] for item in data]
    avg = lambda mss: round(sum(mss[0]) / sum(mss[1]), 4)

    data_gen = json.loads(open(path_gen, 'r', encoding='utf-8').read())
    data_ref_ = json.loads(open(path_ref, 'r', encoding='utf-8').read())
    if len(data_gen) == len(data_ref_):
        data_ref = data_ref_
    else:
        data_ref = [data for data in data_ref_ if data['input_size'] in [item['input_size'] for item in data_gen]]
    assert len(data_gen) == len(data_ref), ""
    
    ms_ref, ms_gen = get_ms(data_ref), get_ms(data_gen)
    ms = avg((ms_ref, ms_gen))

    # if efficiency >= 100 or ms >= 10:
    if ms >= 10 or ms <= 0.1:
        assert False, f"{path_gen.split('/')[-1]} test failed!"
    return ms

def statis(gen_folder):
    avg = lambda listt: round(sum(listt) / len(listt), 2)
    files = [f for f in os.listdir(gen_folder) if f.endswith(".json")]
    spdups, effcys = [], []
    print("===="*40)
    for f in files:
        path_gen = os.path.join(gen_folder, f)
        path_ref = os.path.join(ref_folder, f)
        data_gen_ = json.loads(open(path_gen, 'r', encoding='utf-8').read())
        if len(data_gen_) == 0:
            continue
        
        try:
            ms = calculate(path_gen, path_ref)
            print(f"{f}: {ms}")
            spdups.append(ms)
        except:
            print(f"{f} failed")

    print(spdups)
    print(f"\n{gen_folder.split('/')[-1]}")
    print(f"speed up: {avg(spdups)}")
    print("===="*40)

def arg_parser():
    parser = argparse.ArgumentParser(description='Efficiency statistics')
    parser.add_argument('--gen_folder', type=str, required=True, help='The generated folder path')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    gen_folder = args.gen_folder
    statis(gen_folder)