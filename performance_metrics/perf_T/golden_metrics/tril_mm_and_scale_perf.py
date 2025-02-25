import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.tril_mm_and_scale import tril_mm_and_scale
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('tril_mm_and_scale', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):
            n = 2 ** i
            
            A = torch.rand((n, n), dtype=self.dtype or torch.float32)
            B = torch.rand((n, p), dtype=self.dtype or torch.float32)
            alpha = 0.5
            beta = 0.5
            self.input_tensors.append((A, B, alpha, beta))

    def to_cuda(self, input_tensor):
        A, B, alpha, beta = input_tensor
        return (A.cuda(), B.cuda(), alpha, beta)

    def call_op(self, input_tensor):
        A, B, alpha, beta = input_tensor
        return tril_mm_and_scale(A, B, alpha, beta)

    def get_gbps(self, input_tensor, runtime):
        A, B, _, _ = input_tensor
        n, p = A.shape[0], B.shape[1]
        element_size = A.element_size()
        total_bytes = (A.numel() + B.numel() + n*p) * element_size
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tensor, runtime):
        A, B, _, _ = input_tensor
        n, p = A.shape[0], B.shape[1]
        total_flops = 2 * n**2 * p + 2 * n * p
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
