import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.matmul_dequantize_int4 import matmul_dequantize_int4_gptq, quantize_int4
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul_dequantize_int4', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(2, 33):  # Adjust the range as needed for testing
            size = 128 * i
            M = K = N = size
            a = torch.randn((M, K), dtype=torch.float16)
            b = torch.randn((K, N), dtype=torch.float16)
            group_size = 128
            int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=group_size)
            input_tensor = (a, int_b, b_scale, b_zero_point, group_size)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        x, qweight, scales, qzeros, group_size = input_tensor
        return (x.cuda(), qweight.cuda(), scales.cuda(), qzeros.cuda(), group_size)

    def call_op(self, input_tensor):
        x, qweight, scales, qzeros, group_size = input_tensor
        return matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, group_size)

    def get_gbps(self, input_tensor, runtime):
        a, int_b, b_scale, b_zero_point, group_size = input_tensor
        # Calculate memory access size in bytes
        mem_access = (
            a.numel() * a.element_size() +  # Input matrix A
            int_b.numel() * int_b.element_size() +  # Quantized weight matrix int_b
            b_scale.numel() * b_scale.element_size() +  # Scale matrix b_scale
            b_zero_point.numel() * b_zero_point.element_size() +  # Zero-point matrix b_zero_point
            a.shape[0] * int_b.shape[1] * 2  # Output matrix C, assuming float16 (2 bytes per element)
        )
        # Convert memory access to GB
        mem_access_gb = mem_access / (1024 ** 3)
        # Runtime in seconds
        runtime_s = runtime / 1000.0
        # GBPS calculation
        return mem_access_gb / runtime_s
    
    def get_tflops(self, input_tensor, runtime):
        x, qweight, scales, qzeros, group_size = input_tensor
        M, K = x.shape
        N = qweight.shape[1]
        FLOPS = 2 * M * N * K
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
