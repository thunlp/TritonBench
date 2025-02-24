import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.batched_vecmat_mult import batched_vecmat
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('batched_vecmat_mult', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):  # Adjust the range as needed for your testing
            M = 16 * i
            N = 16 * i
            K = 16 * i
            block_m = 16  # Example block size, adjust as needed
            block_n = 16
            block_k = 16
            input_tensor = (M, N, K, block_m, block_n, block_k)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        # No need to transfer to CUDA as the batched_vecmat function already initializes tensors on CUDA
        return input_tensor

    def call_op(self, input_tensor):
        M, N, K, block_m, block_n, block_k = input_tensor
        return batched_vecmat(M, N, K, block_m, block_n, block_k)

    def get_gbps(self, input_tensor, runtime):
        M, N, K, block_m, block_n, block_k = input_tensor
        # Calculate the total bytes processed
        total_bytes = (M * K + M * N * K + M * N) * 4  # 4 bytes for float32
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        M, N, K, block_m, block_n, block_k = input_tensor
        # Calculate the total floating-point operations
        FLOPS = 2 * M * N * K
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
