import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.least_squares_qr import least_squares_qr
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('least_squares_qr', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 11):
            m = 2 ** i
            n = 2 ** (i - 2)
            A = torch.randn(m, n, dtype=self.dtype)
            b = torch.randn(m, 1, dtype=self.dtype)
            self.input_tensors.append((A, b))

    def to_cuda(self, input_tensor):
        A, b = input_tensor
        return (A.cuda(), b.cuda())
    
    def call_op(self, input_tensor):
        A, b = input_tensor
        return least_squares_qr(A, b)
    
    def get_gbps(self, input_tensor, runtime):
        A, b = input_tensor
        m, n = A.shape
        bytes_input = (A.numel() + b.numel()) * A.element_size()
        bytes_output = n * A.element_size()
        total_bytes = bytes_input + bytes_output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        A, b = input_tensor
        m, n = A.shape
        flops_qr = 2 * m * n**2 - (2/3) * n**3
        flops_qtb = 2 * m * n
        flops_solve = n**2
        total_flops = flops_qr + flops_qtb + flops_solve
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
