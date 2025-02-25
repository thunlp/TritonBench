import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.int_scaled_matmul import int_scaled_matmul_kernel, Config
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('int_scaled_matmul', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):  # Example sizes from 256 to 16384
            size = 128 * i
            M = N = K = size
            A = torch.randint(-128, 128, (M, K), dtype=torch.int8)
            B = torch.randint(-128, 128, (M, K), dtype=torch.int8)
            scales1 = torch.randn((M, N), dtype=torch.float16)
            C = torch.empty((M, N), dtype=torch.int32)
            BLOCK_CONFIG = 128
            input_tensor = (A, B, scales1, C, BLOCK_CONFIG)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        a, b, scales1, c, BLOCK_CONFIG = input_tensor
        return (a.cuda(), b.cuda(), scales1.cuda(), c.cuda(), BLOCK_CONFIG)

    def call_op(self, input_tensor):
        a, b, scales1, c, BLOCK_CONFIG = input_tensor
        config = Config(
            BLOCK_M=BLOCK_CONFIG,
            BLOCK_N=BLOCK_CONFIG,
            BLOCK_K=BLOCK_CONFIG,
            GROUP_M=8
        )
        return int_scaled_matmul_kernel(a, b, scales1, c, config)

    def get_gbps(self, input_tensor, runtime):
        A, B, scales1, C, BLOCK_CONIFG = input_tensor
        M, K = A.shape
        K, N = B.shape
        total_bytes = (
            A.numel() * A.element_size() +
            B.numel() * B.element_size() +
            scales1.numel() * scales1.element_size() +
            C.numel() * C.element_size()
        )
        runtime_seconds = runtime / 1000.0

        gbps = total_bytes / runtime_seconds / (1024 ** 3)
        return gbps
    
    def get_tflops(self, input_tensor, runtime):
        a, b, scales1, c, BLOCK_CONFIG = input_tensor
        M, K = a.shape
        K, N = b.shape
        FLOPS = 2 * M * N * K
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
