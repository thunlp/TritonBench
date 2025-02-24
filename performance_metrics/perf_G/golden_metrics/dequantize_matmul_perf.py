import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.dequantize_matmul import matmul_dequantize_int8
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('dequantize_matmul', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):  # Adjust the range for appropriate sizes
            size = 128 * i
            M = N = K = size
            A = torch.randn(M, K, dtype=torch.float16)
            B = torch.randn(K, N, dtype=torch.float16)
            scale = torch.randn(N, dtype=torch.float16)
            input_tensor = (A, B, scale)
            self.input_tensors.append(input_tensor)
            
    def to_cuda(self, input_tensor):
        a, b, b_scale = input_tensor
        return (a.cuda(), b.cuda(), b_scale.cuda())

    def call_op(self, input_tensor):
        a, b, b_scale = input_tensor
        return matmul_dequantize_int8(a, b, b_scale)

    def get_gbps(self, input_tensor, runtime):
        a, b, b_scale = input_tensor
        total_bytes = a.numel() * a.element_size() + b.numel() * b.element_size() + b_scale.numel() * b_scale.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        a, b, b_scale = input_tensor
        M, K = a.shape
        K, N = b.shape
        FLOPS = 2 * M * N * K  # Each element in the result matrix requires K multiplications and K-1 additions
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
