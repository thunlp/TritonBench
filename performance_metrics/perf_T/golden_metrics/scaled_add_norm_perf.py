import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.scaled_add_norm import scaled_add_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('scaled_add_norm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            dtype = self.dtype or torch.float32
            y = torch.rand(size, dtype=dtype)
            x = torch.rand(size, dtype=dtype)
            alpha = 0.5
            self.input_tensors.append((y, x, alpha))

    def to_cuda(self, input_tensor):
        y, x, alpha = input_tensor
        return (y.cuda(), x.cuda(), alpha)
    
    def call_op(self, input_tensor):
        y, x, alpha = input_tensor
        return scaled_add_norm(y, x, alpha)
    
    def get_gbps(self, input_tensor, runtime):
        y, x, alpha = input_tensor
        total_bytes = 5 * y.numel() * y.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        y, x, alpha = input_tensor
        flops_per_element = 2 + 2
        total_flops = flops_per_element * y.numel()
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
