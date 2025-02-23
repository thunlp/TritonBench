import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.rmsnorm_fused_llama import rmsnorm_forward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rmsnorm_fused_llama', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):
            size = 2 ** i
            x = torch.rand((4096, size), dtype=torch.float16)
            weight = torch.rand(size, dtype=torch.float16)
            eps = 1e-5
            self.input_tensors.append((x, weight, eps))

    def to_cuda(self, input_tensor):
        x, weight, eps = input_tensor
        return (x.cuda(), weight.cuda(), eps)

    def call_op(self, input_tensor):
        x, weight, eps = input_tensor
        return rmsnorm_forward(x, weight, eps)

    def get_gbps(self, input_tensor, runtime):
        x, weight, _ = input_tensor
        total_bytes = (x.numel() + weight.numel() + x.numel()) * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _, _ = input_tensor
        FLOPS = 2 * x.numel()  # Assuming 2 FLOPS per element for normalization and scaling
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
