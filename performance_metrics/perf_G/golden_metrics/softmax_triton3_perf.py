import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.softmax_triton3 import softmax
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('softmax_triton3', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        M = 4096
        for i in range(7, 16):
            N = 2 ** i
            x = torch.randn(M, N, device='cuda', dtype=torch.float32)
            self.input_tensors.append(x)
        
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        x = input_tensor
        return softmax(x)

    def get_gbps(self, input_tensor, runtime):
        x = input_tensor
        runtime_second = runtime / 1000
        gb = 2 * x.numel() * x.element_size() * 1e-9
        gbps = gb / runtime_second
        return gbps

    def get_tflops(self, input_tensor, runtime):
        x = input_tensor
        n_cols = x.shape[1]
        M = x.shape[0]

        flops_per_row = 4 * n_cols - 1

        total_flops = M * flops_per_row

        runtime_second = runtime / 1000.0

        tflops = total_flops / (runtime_second * 1e12)
        
        return tflops
    
    def run_benchmark(self):
        results = []
        for input_tensor_ in self.input_tensors:
            input_tensor = self.to_cuda(input_tensor_)
            # print(input_tensor)
            op = lambda : self.call_op(input_tensor)
            ms = self.get_runtime(op)
            gbps = self.get_gbps(input_tensor, ms)
            tflops = self.get_tflops(input_tensor, ms)
            result = {
                "input_size": [input_tensor.shape],
                "ms": ms,
                "GB/s": gbps,
                "TFLOPS": tflops
            }
            print(result)
            results.append(result)
            input_tensor = None
        folder_path = "/home/lishangzhan/triton/bench_performance/results"
        file_name = self.op_name + ".json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
