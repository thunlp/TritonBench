import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.matrix_reduction import load_reduce
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matrix_reduction', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 10):  # Adjust the range as needed for testing
            BLOCK_M = 2 ** i
            BLOCK_N = 2 ** i
            # x = torch.randn((BLOCK_M, 128), dtype=torch.float16)
            self.input_tensors.append((BLOCK_M, BLOCK_N))

    def to_cuda(self, input_tensor):
        return input_tensor

    def call_op(self, input_tensor):
        BLOCK_M, BLOCK_N = input_tensor
        return load_reduce(BLOCK_M, BLOCK_N, 'float16')

    def get_gbps(self, input_tensor, runtime):
        BLOCK_M, BLOCK_N, = input_tensor
        dtype = torch.float16
        x = torch.randn((BLOCK_M, BLOCK_N), dtype=dtype)
        y = torch.empty((BLOCK_M,), dtype=dtype)
        total_bytes = x.numel() * x.element_size() + y.numel() * y.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        BLOCK_M, BLOCK_N = input_tensor
        dtype = torch.float16
        x = torch.randn((BLOCK_M, BLOCK_N), dtype=dtype)
        y = torch.empty((BLOCK_M,), dtype=dtype)
        FLOPS = BLOCK_M * BLOCK_N  # Assuming one operation per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
