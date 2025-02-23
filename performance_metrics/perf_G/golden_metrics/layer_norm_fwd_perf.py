import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.layer_norm_fwd import _layer_norm_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('layer_norm_fwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 15):  # Adjust the range as needed for your testing
            M = 2 ** i
            N = 2 ** i
            x = torch.rand((M, N), dtype=torch.float32)
            weight = torch.rand((N,), dtype=torch.float32)
            bias = torch.rand((N,), dtype=torch.float32)
            self.input_tensors.append((x, weight, bias))

    def to_cuda(self, input_tensor):
        x, weight, bias = input_tensor
        return (x.cuda(), weight.cuda(), bias.cuda())

    def call_op(self, input_tensor):
        x, weight, bias = input_tensor
        eps = 1e-5
        return _layer_norm_fwd(x, weight, bias, eps)

    def get_gbps(self, input_tensor, runtime):
        x, weight, bias = input_tensor
        total_bytes = (x.numel() + weight.numel()) * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, weight, bias = input_tensor
        # Assuming FLOPS involves normalization and scaling operations
        FLOPS = 5 * x.numel()  # This is a rough estimate, adjust based on actual operations
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
