import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.quantize_dynamic import dynamic_custom
from performance_utils import Performance_Metrics, do_bench_config

import torch
from torch import nn
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('quantize_dynamic', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):
            size = 2 ** i
            model = nn.Linear(size, size)
            self.input_tensors.append(model)

    def to_cuda(self, model):
        return model

    def call_op(self, model):
        qconfig_spec = {nn.Linear}
        return dynamic_custom(model, qconfig_spec=qconfig_spec)

    def get_gbps(self, model, runtime):
        weight = model.weight
        M, N = weight.shape
        bytes_read = weight.numel() * weight.element_size()
        bytes_written = weight.numel() * 1 + M * 4
        total_bytes = bytes_read + bytes_written
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, model, runtime):
        weight = model.weight
        M, N = weight.shape
        FLOPS = 3 * M * N 
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
                "input_size": [input_tensor.weight.shape],
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
