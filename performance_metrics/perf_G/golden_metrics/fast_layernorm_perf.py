import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fast_layernorm import fast_layernorm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fast_layernorm', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 15):
            size = 2 ** i
            input_tensor = torch.rand((2048, size), dtype=torch.float32)
            layernorm = torch.nn.LayerNorm(size, elementwise_affine=True)
            self.input_tensors.append((layernorm, input_tensor))

    def to_cuda(self, input_tensor):
        layernorm, tensor = input_tensor
        layernorm = layernorm.cuda()
        tensor = tensor.cuda()
        return (layernorm, tensor)

    def call_op(self, input_tensor):
        layernorm, tensor = input_tensor
        return fast_layernorm(layernorm, tensor)

    def get_gbps(self, input_tensor, runtime):
        _, x = input_tensor
        total_bytes = 2 * x.numel() * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        _, x = input_tensor
        FLOPS = 5 * x.numel()  # Approximate FLOPS for layernorm
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS
    
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
                "input_size": [item.shape for item in input_tensor if type(item)==torch.Tensor ],
                "ms": ms,
                "GB/s": gbps,
                "TFLOPS": tflops
            }
            print(result)
            results.append(result)
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
