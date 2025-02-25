import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.solve import solve
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('solve', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):
            n = 2 ** i
            A = torch.randn(n, n, dtype=self.dtype) + n * torch.eye(n, dtype=self.dtype)
            B = torch.randn(n, 1, dtype=self.dtype)
            self.input_tensors.append((A, B))
    
    def to_cuda(self, input_tensor):
        A, B = input_tensor
        return (A.cuda(), B.cuda())
        
    def call_op(self, input_tensor):
        A, B = input_tensor
        return solve(A, B)

    def get_gbps(self, input_tensor, runtime):
        A, B = input_tensor
        X_shape = B.shape
        total_bytes = (A.numel() + B.numel() + X_shape[0]*X_shape[1]) * A.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        A, B = input_tensor
        n = A.shape[0]
        k = B.shape[1] if len(B.shape) > 1 else 1
        flops = (2/3) * (n**3) + 2 * (n**2) * k
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
