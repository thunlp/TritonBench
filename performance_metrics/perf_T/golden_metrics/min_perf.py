import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.min import min
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, dim=0, **kwargs):
        super().__init__('min', dtype=dtype, is_backward=is_backward, **kwargs)
        self.dim = dim

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(6, 20):
            size = 2 ** i
            input_tensor = torch.rand((size, 1024), dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        values, _ = min(input_tensor, dim=self.dim)
        return values
    
    def get_gbps(self, input_tensor, runtime):
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_shape = list(input_tensor.shape)
        dim_size = output_shape.pop(self.dim)
        output_size = 1
        for d in output_shape:
            output_size *= d
        values_bytes = output_size * input_tensor.element_size()
        indices_bytes = output_size * 8
        total_bytes = input_bytes + values_bytes + indices_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        dim_size = input_tensor.shape[self.dim]
        other_dims = list(input_tensor.shape)
        other_dims.pop(self.dim)
        other_size = 1
        for d in other_dims:
            other_size *= d
        flops = other_size * (dim_size - 1)
        TFLOPS = flops / (runtime / 1000) / 1e12
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
                "input_size": [input_tensor.shape],
                "ms": ms,
                "GB/s": gbps,
                "TFLOPS": tflops
            }
            print(result)
            results.append(result)
            input_tensor = None
        folder_path = "/home/lishangzhan/triton/torch_performance/results"
        file_name = self.op_name + ".json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
