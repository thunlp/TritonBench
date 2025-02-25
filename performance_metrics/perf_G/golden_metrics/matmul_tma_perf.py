import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.matmul_tma import warpper_tma_load_store
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul_tma', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 8):  # Adjust the range as needed for your testing
            M = N = K = 2 ** i
            self.input_tensors.append((M, N, K))

    def to_cuda(self, input_tensor):
        return input_tensor

    def call_op(self, input_tensor):
        M, N, K = input_tensor
        NUM_CTAS = 1
        NUM_WARPS = 8
        TRANS_A = True
        TRANS_B = True
        OUTPUT_F16 = True
        return warpper_tma_load_store(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_F16)

    def get_gbps(self, input_tensor, runtime):
        total_time_seconds = runtime / 1000.0
        M, N, K = input_tensor
        dtype_size = 2
        numel_a = M * K
        numel_b = K * N
        numel_c = M * N
        total_bytes = (numel_a + numel_b + numel_c) * dtype_size
        
        total_bytes += 2 * numel_a * dtype_size
        total_bytes += 2 * numel_b * dtype_size
        gbps = total_bytes / (total_time_seconds * 1e9)
        return gbps
    
    def get_tflops(self, input_tensor, runtime):
        M, N, K = input_tensor
        FLOPS = 2 * M * N * K  # 2 operations per element in matrix multiplication
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
