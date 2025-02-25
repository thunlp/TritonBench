import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_qr_solve import fused_qr_solve
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_qr_solve', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 13):
            n = 2 ** i
            m = 2 * n
            k = 10
            A = torch.rand((m, n), dtype=self.dtype)
            b = torch.rand((m, k), dtype=self.dtype)
            self.input_tensors.append((A, b))

    def to_cuda(self, input_tensor):
        A, b = input_tensor
        return (A.cuda(), b.cuda())
    
    def call_op(self, input_tensor):
        A, b = input_tensor
        return fused_qr_solve(A, b)
    
    def get_gbps(self, input_tensor, runtime):
        A, b = input_tensor
        m, n = A.shape
        k = b.shape[1]
        element_size = A.element_size()
        
        total_bytes = (A.numel() + b.numel() + n*k) * element_size
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        A, b = input_tensor
        m, n = A.shape
        k = b.shape[1]
        
        flops_qr = 2 * m * n**2 - (2/3) * n**3
        
        flops_matmul = 2 * m * n * k
        
        flops_solve = n**2 * k
        
        total_flops = flops_qr + flops_matmul + flops_solve
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
