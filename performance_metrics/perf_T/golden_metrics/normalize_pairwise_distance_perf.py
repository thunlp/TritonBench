import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.normalize_pairwise_distance import normalize_pairwise_distance
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('normalize_pairwise_distance', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):
            B = 16
            N = 2 ** i
            D = 128
            x1 = torch.rand((B, N, D), dtype=self.dtype)
            x2 = torch.rand((B, N, D), dtype=self.dtype)
            self.input_tensors.append((x1, x2))

    def to_cuda(self, input_tensor):
        x1, x2 = input_tensor
        return (x1.cuda(), x2.cuda())
    
    def call_op(self, input_tensor):
        x1, x2 = input_tensor
        return normalize_pairwise_distance(x1, x2)
    
    def get_gbps(self, input_tensor, runtime):
        x1, x2 = input_tensor
        output_numel = x1.numel() // x1.shape[-1]
        total_bytes = (x1.numel() + x2.numel() + output_numel * 7) * x1.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x1, x2 = input_tensor
        flops = 3 * x1.numel() + 4 * (x1.numel() // x1.shape[-1])
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
