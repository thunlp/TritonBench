import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.matmul_dequant_int4 import matmul_dequantize_int4_s1, quantize_int4
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul_dequant_int4', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):  # Adjust the range for different sizes
            size = 128 * i
            M = K = N = size
            a = torch.randn((M, K), dtype=torch.float16)
            b = torch.randn((K, N), dtype=torch.float16)
            group_size = 128
            int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=group_size)
            input_tensor = (a, int_b, b_scale, b_zero_point, group_size)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        a, b, b_scale, b_zero_point, group_size = input_tensor
        return (a.cuda(), b.cuda(), b_scale.cuda(), b_zero_point.cuda(), group_size)

    def call_op(self, input_tensor):
        a, b, b_scale, b_zero_point, group_size = input_tensor
        return matmul_dequantize_int4_s1(a, b, b_scale, b_zero_point, group_size)

    def get_gbps(self, input_tensor, runtime):
        a, int_b, b_scale, b_zero_point, group_size = input_tensor
        
        data_size_a = a.numel() * a.element_size()
        data_size_b = int_b.numel() * int_b.element_size()
        data_size_out = a.shape[0] * int_b.shape[1] * 2
        
        total_data_bytes = data_size_a + data_size_b + data_size_out
        
        total_data_gb = total_data_bytes / 1e9
        
        runtime_seconds = runtime / 1000
        gbps = total_data_gb / runtime_seconds
        return gbps
    
    def get_tflops(self, input_tensor, runtime):
        a, b, b_scale, b_zero_point, group_size = input_tensor
        M, K = a.shape
        N = b.shape[1]
        # Calculate total FLOPS
        FLOPS = 2 * M * N * K
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
