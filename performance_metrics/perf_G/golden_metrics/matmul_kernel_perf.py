import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.matmul_kernel import matmul
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul_kernel', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(6, 32):  # Adjust the range as needed for different sizes
            size = 128 * i
            a = torch.rand(size, size, dtype=torch.float16)
            b = torch.rand(size, size, dtype=torch.float16)
            c = torch.empty(size, size, dtype=torch.float16)
            self.input_tensors.append((c, a, b, size, size, size))

    def to_cuda(self, input_tensor):
        c, a, b, M, N, K = input_tensor
        return (c.cuda(), a.cuda(), b.cuda(), M, N, K)

    def call_op(self, input_tensor):
        c, a, b, M, N, K = input_tensor
        BLOCK_SIZE_M = 128  # Example block size, adjust as needed
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        return c

    def get_gbps(self, input_tensor, runtime):
        c, a, b, M, N, K = input_tensor
        total_bytes = (a.numel() + b.numel() + c.numel()) * a.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        c, a, b, M, N, K = input_tensor
        FLOPS = 2 * M * N * K
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
