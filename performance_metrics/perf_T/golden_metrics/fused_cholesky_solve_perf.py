import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_cholesky_solve import fused_cholesky_solve
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_cholesky_solve', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        k = 1
        for i in range(2, 10):
            n = 2 ** i
            A = torch.rand(n, n, dtype=torch.float32)
            A = A @ A.T + n * torch.eye(n, dtype=torch.float32)
            b = torch.rand(n, k, dtype=torch.float32)
            self.input_tensors.append((A, b))

    def to_cuda(self, input_tensor):
        A, b = input_tensor
        return (A.cuda(), b.cuda())
    
    def call_op(self, input_tensor):
        A, b = input_tensor
        return fused_cholesky_solve(A, b)
    
    def get_gbps(self, input_tensor, runtime):
        A, b = input_tensor
        element_size = A.element_size()
        total_bytes = (4 * A.numel() + 4 * b.numel()) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        A, b = input_tensor
        n, k = A.shape[0], b.shape[1]
        total_flops = (n**3)/3 + 2*k*(n**2)
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
