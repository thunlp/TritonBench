import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.matmul_dequantize import matmul_dequantize_int4_gptq
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul_dequantize', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 32):  # Example sizes, adjust as needed
            M = 128 * i
            K = 128 * i
            N = 128 * i
            x = torch.rand((M, K), dtype=torch.float16)
            qweight = torch.randint(0, 16, (K // 8, N), dtype=torch.int32)
            scales = torch.rand((K // 128, N), dtype=torch.float16)
            qzeros = torch.randint(0, 16, (K // 128, N // 8), dtype=torch.int32)
            group_size = 128
            self.input_tensors.append((x, qweight, scales, qzeros, group_size))

    def to_cuda(self, input_tensor):
        x, qweight, scales, qzeros, group_size = input_tensor
        return (x.cuda(), qweight.cuda(), scales.cuda(), qzeros.cuda(), group_size)

    def call_op(self, input_tensor):
        x, qweight, scales, qzeros, group_size = input_tensor
        return matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, group_size)

    def get_gbps(self, input_tensor, runtime):
        x, qweight, scales, qzeros, group_size = input_tensor
        M, K = x.shape
        N = qweight.shape[1]
        total_bytes = (x.numel() * x.element_size() +
                       qweight.numel() * qweight.element_size() +
                       scales.numel() * scales.element_size() +
                       qzeros.numel() * qzeros.element_size() +
                       M * N * 2)  # Output size in bytes (float16)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, qweight, scales, qzeros, group_size = input_tensor
        M, K = x.shape
        N = qweight.shape[1]
        FLOPS = 2 * M * N * K  # Each multiplication and addition counts as two FLOPS
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
