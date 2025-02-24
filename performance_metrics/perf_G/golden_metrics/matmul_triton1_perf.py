import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the matmul function
from TritonBench_v1.matmul_triton1 import matmul
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul_triton1', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 7):
            size = 2 ** i
            M = K = N = size
            A = torch.randn((M, K), dtype=torch.float32)
            B = torch.randn((K, N), dtype=torch.float32)
            self.input_tensors.append((A, B))
    def to_cuda(self, input_tensor):
        x, y = input_tensor
        return (x.cuda(), y.cuda())

    def call_op(self, input_tensor):
        x, y = input_tensor
        return matmul(x, y)

    def get_gbps(self, input_tensor, runtime):
        x, y = input_tensor
        total_bytes = (x.numel() + y.numel() + x.size(0) * y.size(1)) * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, y = input_tensor
        m, k = x.shape
        _, n = y.shape
        FLOPS = 2 * m * n * k  # 2 * M * N * K operations for matrix multiplication
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
