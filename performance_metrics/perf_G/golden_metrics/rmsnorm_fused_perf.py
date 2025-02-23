import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rmsnorm_fused import TritonLlamaRMSNorm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rmsnorm_fused', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 15):  # Adjust the range based on your needs
            size = 2 ** i
            weight = torch.rand(size, dtype=torch.float32)
            input_tensor = torch.rand((2048, size), dtype=torch.float32)
            self.input_tensors.append((input_tensor, weight))

    def to_cuda(self, input_tensor):
        x, weight = input_tensor
        return x.cuda(), weight.cuda()

    def call_op(self, input_tensor):
        x, weight = input_tensor
        norm_layer = TritonLlamaRMSNorm(weight)
        return norm_layer(x)

    def get_gbps(self, input_tensor, runtime):
        x, _ = input_tensor
        total_bytes = 2 * x.numel() * x.element_size()  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _ = input_tensor
        FLOPS = 2 * x.numel()  # Assuming 2 FLOPS per element for normalization and scaling
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
