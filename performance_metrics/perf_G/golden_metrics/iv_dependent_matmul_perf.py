import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.iv_dependent_matmul import iv_dependent_matmul_wrapper
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('iv_dependent_matmul', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 32):  # Adjust the range as needed for testing
            M = N = K = 128 * i
            self.input_tensors.append((M, K, N))
    
    def to_cuda(self, input_tensor):
        # M, K, N = input_tensor
        # a = torch.rand((M, K), dtype=torch.float16)
        # b = torch.rand((K, N), dtype=torch.float16)
        # return a.cuda(), b.cuda()
        return input_tensor
        
    def call_op(self, input_tensor):
        M, K, N = input_tensor
        BLOCK_SIZE_M = 128  # Example block size, adjust as needed
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        return iv_dependent_matmul_wrapper(M, K, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, type="pre_load")
    
    def get_gbps(self, input_tensor, runtime):
        M, K, N = input_tensor
        total_bytes = (M * K + K * N + M * N) * 4  # 4 bytes for float16
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        M, K, N = input_tensor
        FLOPS = 2 * M * N * K  # 2 * M * N * K operations for matrix multiplication
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
