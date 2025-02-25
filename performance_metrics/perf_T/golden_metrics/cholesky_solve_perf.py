import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.cholesky_solve import cholesky_solve
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cholesky_solve', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):  # n=16, 32, 64, 128, 256, 512
            n = 2 ** i
            k = 10
            L = torch.rand(n, n, dtype=self.dtype)
            L = torch.tril(L)
            L.diagonal().add_(1e-3)
            B = torch.rand(n, k, dtype=self.dtype)
            self.input_tensors.append((B, L))
    
    def to_cuda(self, input_tensor):
        B, L = input_tensor
        return (B.cuda(), L.cuda())
        
    def call_op(self, input_tensor):
        B, L = input_tensor
        return cholesky_solve(B, L, upper=False)
    
    def get_gbps(self, input_tensor, runtime):
        B, L = input_tensor
        B_size = B.numel() * B.element_size()
        L_size = L.numel() * L.element_size()
        X_size = B.numel() * B.element_size()
        total_bytes = B_size + L_size + X_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        B, L = input_tensor
        n, k = B.shape[-2], B.shape[-1]
        FLOPS = 2 * (n ** 2) * k
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
