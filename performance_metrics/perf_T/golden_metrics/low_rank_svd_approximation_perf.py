import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.low_rank_svd_approximation import low_rank_svd_approximation
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('low_rank_svd_approximation', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):
            size = 128 * i
            A = torch.rand(size, size, dtype=self.dtype or torch.float32)
            k = min(50, size)
            self.input_tensors.append((A, k))

    def to_cuda(self, input_tuple):
        A, k = input_tuple
        return (A.cuda(), k)

    def call_op(self, input_tuple):
        A, k = input_tuple
        return low_rank_svd_approximation(A, k)

    def get_gbps(self, input_tuple, runtime):
        A, k = input_tuple
        total_bytes = A.numel() * A.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        A, k = input_tuple
        m, n = A.shape[-2], A.shape[-1]
        flops_svd = 2 * m * n**2
        flops_matmul = 2 * m * k * (k + n)
        total_flops = flops_svd + flops_matmul
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
