import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_bmm_dropout_gelu import fused_bmm_dropout_gelu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_bmm_dropout_gelu', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for exp in range(2, 20):
            B = 32
            N, M, P = 128 * exp, 128 * exp, 128 * exp
            input1 = torch.randn(B, N, M, dtype=self.dtype)
            input2 = torch.randn(B, M, P, dtype=self.dtype)
            self.input_tensors.append((input1, input2))
    
    def to_cuda(self, input_tensor):
        input1, input2 = input_tensor
        return (input1.cuda(), input2.cuda())
    
    def call_op(self, input_tensor):
        input1, input2 = input_tensor
        return fused_bmm_dropout_gelu(input1, input2)
    
    def get_gbps(self, input_tensor, runtime):
        input1, input2 = input_tensor
        B, N, M = input1.shape
        B2, M2, P = input2.shape
        assert B == B2 and M == M2, "Input size not match"
        element_size = input1.element_size()
        total_bytes = (input1.numel() + input2.numel() + B * N * P + B * N * P * 4) * element_size
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        input1, input2 = input_tensor
        B, N, M = input1.shape
        B2, M2, P = input2.shape
        assert B == B2 and M == M2, "Input size not match"
        flops = 2 * B * N * M * P
        return flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
