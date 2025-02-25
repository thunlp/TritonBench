import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_add_mul_groupnorm import fused_add_mul_groupnorm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_add_mul_groupnorm', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        N = 16
        C = 64
        num_groups = 8
        
        for i in range(6, 14):
            H = 128
            W = 2 ** i
            input1 = torch.randn(N, C, H, W, dtype=self.dtype)
            input2 = torch.randn(N, C, H, W, dtype=self.dtype)
            weight = torch.randn(C, dtype=self.dtype)
            bias = torch.randn(C, dtype=self.dtype)
            self.input_tensors.append((input1, input2, weight, bias, num_groups))
    
    def to_cuda(self, input_tuple):
        input1, input2, weight, bias, num_groups = input_tuple
        return (
            input1.cuda(),
            input2.cuda(),
            weight.cuda(),
            bias.cuda(),
            num_groups,
        )
        
    def call_op(self, input_tuple):
        input1, input2, weight, bias, num_groups = input_tuple
        return fused_add_mul_groupnorm(
            input1, input2, weight, bias, num_groups
        )
    
    def get_gbps(self, input_tuple, runtime):
        input1, input2, _, _, _ = input_tuple
        element_size = input1.element_size()
        numel = input1.numel()
        total_bytes = numel * element_size * 7
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input1, _, _, _, _ = input_tuple
        numel = input1.numel()
        flops = 10 * numel
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
