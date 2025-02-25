import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.autocast import autocast
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('autocast', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):
            size = 128 * i
            a = torch.rand((size, size), dtype=torch.float32)
            b = torch.rand((size, size), dtype=torch.float32)
            self.input_tensors.append((a, b))

    def to_cuda(self, input_tensor):
        a, b = input_tensor
        return (a.cuda(), b.cuda())
    
    def call_op(self, input_tensor):
        a, b = input_tensor
        with autocast('cuda'):
            return torch.mm(a, b)
    
    def get_gbps(self, input_tensor, runtime):
        a, b = input_tensor
        size = a.size(0)
        element_size = a.element_size()
        input_bytes = (a.numel() + b.numel()) * element_size
        output_bytes = size * size * element_size
        total_bytes = input_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        a, b = input_tensor
        size = a.size(0)
        flops = 2 * size ** 3
        return flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
