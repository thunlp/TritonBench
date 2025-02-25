import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.matrix_vector_dot import matrix_vector_dot
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__('matrix_vector_dot', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(6, 16):
            n = m = 2 ** i
            A = torch.rand(n, m, dtype=torch.float16)
            x = torch.rand(m, dtype=torch.float16)
            y = torch.rand(n, dtype=torch.float16)
            self.input_tensors.append((A, x, y))

    def to_cuda(self, input_tensors):
        A, x, y = input_tensors
        return (A.cuda(), x.cuda(), y.cuda())
    
    def call_op(self, input_tensors):
        A, x, y = input_tensors
        return matrix_vector_dot(A, x, y, alpha=1.0, beta=1.0)
    
    def get_gbps(self, input_tensors, runtime):
        A, x, y = input_tensors
        element_size = A.element_size()
        total_bytes = (A.numel() + 2*x.numel() + 3*y.numel()) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensors, runtime):
        A, x, y = input_tensors
        n, m = A.shape
        total_flops = 2*n*m + 3*n + (2*n - 1)
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
