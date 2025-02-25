import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.solve_symmetric_ldl import solve_symmetric_ldl
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('solve_symmetric_ldl', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(3, 10):
            n = 2 ** i
            X = torch.rand(n, n, dtype=torch.float32)
            A = X @ X.T
            A += torch.eye(n, dtype=torch.float32) * 1e-6
            b = torch.rand(n, 1, dtype=torch.float32)
            self.input_tensors.append((A, b))

    def to_cuda(self, input_tensor):
        A, b = input_tensor
        return (A.cuda(), b.cuda())
    
    def call_op(self, input_tensor):
        A, b = input_tensor
        return solve_symmetric_ldl(A, b)
    
    def get_gbps(self, input_tensor, runtime):
        A, b = input_tensor
        input_bytes = (A.numel() + b.numel()) * A.element_size()
        output_bytes = b.numel() * b.element_size()
        total_bytes = (input_bytes + output_bytes)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        A, b = input_tensor
        n = A.size(0)
        flops = (n**3 / 3) + 2 * n**2
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
