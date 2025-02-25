import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.combined_activation import combined_activation
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('combined_activation', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):
            N = 2 ** (i // 2)
            D_in = 2 ** (i // 2)
            D_out = 2 ** (i // 2)
            input = torch.randn((N, D_in), dtype=self.dtype)
            weight1 = torch.randn((D_in, D_out), dtype=self.dtype)
            weight2 = torch.randn((D_out,), dtype=self.dtype)
            bias = torch.randn((D_out,), dtype=self.dtype)
            self.input_tensors.append((input, weight1, weight2, bias))

    def to_cuda(self, input_tuple):
        return tuple(tensor.cuda() for tensor in input_tuple)
    
    def call_op(self, input_tuple):
        input, weight1, weight2, bias = input_tuple
        return combined_activation(input, weight1, weight2, bias)
    
    def get_gbps(self, input_tuple, runtime):
        input, weight1, weight2, bias = input_tuple
        input_bytes = (input.numel() + weight1.numel() + weight2.numel() + bias.numel()) * input.element_size()
        output_numel = input.shape[0] * weight1.shape[1]
        output_bytes = output_numel * input.element_size()
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input, weight1, weight2, bias = input_tuple
        N = input.shape[0]
        D_in = input.shape[1]
        D_out = weight1.shape[1]

        flops_mm = 2 * N * D_in * D_out
        
        flops_activations = 8 * N * D_out
        
        flops_elementwise = 2 * N * D_out
        
        total_flops = flops_mm + flops_activations + flops_elementwise
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
