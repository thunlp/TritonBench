import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.int8_dequant_matmul import int8_matmul_rowwise_dequantize
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('int8_dequant_matmul', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(2, 33):  # Adjust the range as needed for your testing
            size = 128 * i
            M = K = N = size
            a = torch.randint(-127, 127, (M, K), dtype=torch.int8)
            b = torch.randint(-127, 127, (K, N), dtype=torch.int8)
            state_x = torch.rand(M, dtype=torch.float16)
            state_w = torch.rand(N, dtype=torch.float16)
            bias = torch.rand(N, dtype=torch.float16)
            input_tensor = (a, b, state_x, state_w, bias)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        a, b, state_x, state_w, bias = input_tensor
        return (a.cuda(), b.cuda(), state_x.cuda(), state_w.cuda(), bias.cuda())

    def call_op(self, input_tensor):
        a, b, state_x, state_w, bias = input_tensor
        return int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias)

    def get_gbps(self, input_tensor, runtime):
        a, b, _, _, _ = input_tensor
        total_bytes = (a.numel() + b.numel()) * a.element_size() + a.size(0) * a.element_size() + b.size(1) * b.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        a, b, _, _, _ = input_tensor
        M, K = a.shape
        _, N = b.shape
        FLOPS = 2 * M * N * K
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
