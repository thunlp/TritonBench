import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_embedding_add_tanh import fused_embedding_add_tanh
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_embedding_add_tanh', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        V = 1024
        for i in range(8, 28):
            B = 2 ** (i // 3)
            L = 2 ** ((i + 1) // 3)
            D = 2 ** ((i + 2) // 3)
            input_indices = torch.randint(0, V, (B, L), dtype=torch.int64)
            weight = torch.randn(V, D, dtype=torch.float32)
            other = torch.randn(B, L, D, dtype=torch.float32)
            self.input_tensors.append((input_indices, weight, other))

    def to_cuda(self, input_tuple):
        input_indices, weight, other = input_tuple
        return (input_indices.cuda(), weight.cuda(), other.cuda())
    
    def call_op(self, input_tuple):
        input_indices, weight, other = input_tuple
        return fused_embedding_add_tanh(input_indices, weight, other)
    
    def get_gbps(self, input_tuple, runtime):
        input_indices, weight, other = input_tuple
        elements_input = input_indices.numel()
        D = weight.shape[1]
        bytes_input_indices = elements_input * input_indices.element_size()
        bytes_weight_accessed = elements_input * D * weight.element_size() * 4
        bytes_other = other.numel() * other.element_size()
        bytes_output = elements_input * D * 4
        total_bytes = bytes_input_indices + bytes_weight_accessed + bytes_other + bytes_output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_indices, weight, other = input_tuple
        elements_output = input_indices.numel() * weight.shape[1]  # B*L*D
        FLOPS = 2 * elements_output
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
