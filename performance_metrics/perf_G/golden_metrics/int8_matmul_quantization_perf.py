import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.int8_matmul_quantization import matmul_quantize_int8, quantize_int8
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('int8_matmul_quantization', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 32):  # Adjust the range as needed for different sizes
            M = 128 * i
            K = 128 * i
            N = 128 * i
            fpa = torch.rand((M, K), dtype=torch.float16)
            b = torch.rand((K, N), dtype=torch.float16)
            b, b_scale, _ = quantize_int8(b)
            self.input_tensors.append((fpa, b, b_scale))

    def to_cuda(self, input_tensor):
        fpa, b, b_scale = input_tensor
        return fpa.cuda(), b.cuda(), b_scale.cuda()

    def call_op(self, input_tensor):
        fpa, b, b_scale = input_tensor
        return matmul_quantize_int8(fpa, b, b_scale)

    def get_gbps(self, input_tensor, runtime):
        fpa, b, _ = input_tensor
        M, K = fpa.shape
        _, N = b.shape
        total_bytes = (M * K + K * N + M * N) * 4  # Assuming float16 for fpa and b, float16 for output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        fpa, b, _ = input_tensor
        M, K = fpa.shape
        _, N = b.shape
        FLOPS = M * N * K  # int8 matmul, FLOPS 2 more than float16 matmul
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
