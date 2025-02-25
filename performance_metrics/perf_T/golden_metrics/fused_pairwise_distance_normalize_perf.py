import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_pairwise_distance_normalize import fused_pairwise_distance_normalize
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_pairwise_distance_normalize', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(6, 14):
            N = 2 ** i
            D = 128
            x1 = torch.randn(N, D, dtype=self.dtype or torch.float32)
            x2 = torch.randn(N, D, dtype=self.dtype or torch.float32)
            self.input_tensors.append((x1, x2))

    def to_cuda(self, input_tensor):
        x1, x2 = input_tensor
        return (x1.cuda(), x2.cuda())
    
    def call_op(self, input_tensor):
        x1, x2 = input_tensor
        return fused_pairwise_distance_normalize(x1, x2)
    
    def get_gbps(self, input_tensor, runtime):
        x1, x2 = input_tensor
        input_bytes = (x1.numel() + x2.numel()) * x1.element_size()
        output_bytes = (x1.size(0) * x2.size(0)) * x1.element_size()
        total_bytes = input_bytes * 11 + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x1, x2 = input_tensor
        N, D = x1.size(0), x1.size(1)
        M = x2.size(0)
        flops = 3 * N * M * D
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
