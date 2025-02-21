import os, json

gen_fold = "/home/lijianling/TritonLLM-g/triton_bench_build/EVAL1/Tbench_perf/"
load_json = lambda path: json.loads(open(path, 'r', encoding='utf-8').read())
avg = lambda mss: print(round(sum(mss) / len(mss), 2))


def single_speedup(msRef, msGen):
    assert len(msRef) == len(msGen) != 0, ""
    scores = [round(msRef[i] / msGen[i], 4) for i in range(len(msRef))]
    maxx, minn = max(scores), min(scores)
    if maxx > 10:
        scores.remove(maxx)
        print(f"maxx: {maxx}")
    elif minn < 0.1:
        scores.remove(minn)
        print(f"minn: {minn}")

    assert len(scores) > 0, ""
    return round(sum(scores) / len(scores), 4)

def all_speedup(ref_fold):
    files = [f for f in os.listdir(ref_fold) if f.endswith(".json")]
    speeds = []
    for f in files:
        path_ref = os.path.join(ref_fold, f)
        path_gen = os.path.join(gen_fold, f)

        msRef = load_json(path_ref)
        msGen = load_json(path_gen)

        speed = single_speedup(msRef, msGen)
        speeds.append(speed)
    avg(speeds)
    print(ref_fold.split("/")[-1])
    return 

if __name__ == "__main__":
    root = "/home/lijianling/TritonLLM-g/triton_bench_build/EVAL1/torch_perf/"

    for fold in os.listdir(root):
        ref_fold = os.path.join(root, fold)
        if not os.path.isdir(ref_fold):
            continue
        print("--"*20)
        all_speedup(ref_fold)
        
