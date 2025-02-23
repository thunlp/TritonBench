import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.bmm_optimized import bmm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bmm_optimized', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):  # Example sizes from 2^6 to 2^11
            size = 32 * i
            A = torch.rand((32, size, size), dtype=torch.float16)
            B = torch.rand((32, size, size), dtype=torch.float16)
            self.input_tensors.append((A, B))

    def to_cuda(self, input_tensor):
        A, B = input_tensor
        return A.cuda(), B.cuda()

    def call_op(self, input_tensor):
        A, B = input_tensor
        return bmm(A, B)

    def get_gbps(self, input_tensor, runtime):
        A, B = input_tensor
        batch, M, K = A.shape
        _, _, N = B.shape
        total_bytes = (A.numel() + B.numel() + batch * M * N) * A.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        A, B = input_tensor
        batch, M, K = A.shape
        _, _, N = B.shape
        FLOPS = 2 * batch * M * N * K
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
