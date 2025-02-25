import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.solve_and_add_scaled_vector import solve_and_add_scaled_vector
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('solve_and_add_scaled_vector.', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for exp in range(5, 16):
            n = 2 ** exp
            A = torch.triu(torch.randn(n, n, dtype=torch.float32))
            b = torch.randn(n, 1, dtype=torch.float32)
            y = torch.randn(n, 1, dtype=torch.float32)
            alpha = 1.0
            self.input_tensors.append((A, b, y, alpha))

    def to_cuda(self, input_tuple):
        A, b, y, alpha = input_tuple
        return (A.cuda(), b.cuda(), y.cuda(), alpha)

    def call_op(self, input_tuple):
        A, b, y, alpha = input_tuple
        return solve_and_add_scaled_vector(A, b, y, alpha)

    def get_gbps(self, input_tuple, runtime):
        A, b, y, alpha = input_tuple
        n = A.size(0)
        total_bytes = (n**2 + 5 * n) * A.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        A, b, y, alpha = input_tuple
        n = A.size(0)
        flops = (n**2 / 2) + n
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
