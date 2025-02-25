import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.solve_multiple_lu import solve_multiple_lu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('solve_multiple_lu', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 11):
            n = 2 ** i
            A = torch.randn(n, n, dtype=self.dtype or torch.float32)
            Bs = torch.randn(n, n, dtype=self.dtype or torch.float32)
            self.input_tensors.append((A, Bs))

    def to_cuda(self, input_tensor):
        A, Bs = input_tensor
        return (A.cuda(), Bs.cuda())

    def call_op(self, input_tensor):
        A, Bs = input_tensor
        return solve_multiple_lu(A, Bs, pivot=True)

    def get_gbps(self, input_tensor, runtime):
        A, Bs = input_tensor
        element_size = A.element_size()
        X_numel = Bs.numel()
        total_bytes = (A.numel() + Bs.numel() + X_numel) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        A, Bs = input_tensor
        n = A.size(0)
        m = Bs.size(1)
        flops_lu = (2/3) * (n ** 3)
        flops_matmul = 2 * (n ** 2) * m
        flops_solve_l = (n ** 2) * m
        flops_solve_u = (n ** 2) * m
        total_flops = flops_lu + flops_matmul + flops_solve_l + flops_solve_u
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
