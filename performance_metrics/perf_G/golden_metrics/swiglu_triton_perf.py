import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.swiglu_triton import swiglu_forward, swiglu_backward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('swiglu_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 17):
            size = 2 ** i
            a = torch.rand((128, size), dtype=torch.float32)
            b = torch.rand((128, size), dtype=torch.float32)
            self.input_tensors.append((a, b))

    def to_cuda(self, input_tensor):
        a, b = input_tensor
        return a.cuda(), b.cuda()

    def call_op(self, input_tensor):
        a, b = input_tensor
        return swiglu_forward(a, b)

    def get_gbps(self, input_tensor, runtime):
        a, b = input_tensor
        total_bytes = 3 * a.numel() * a.element_size()  # a, b, and output c
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        a, b = input_tensor
        # Assuming each element requires a few operations: multiply, add, sigmoid
        # Let's assume 5 operations per element as a rough estimate
        FLOPS = 5 * a.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
