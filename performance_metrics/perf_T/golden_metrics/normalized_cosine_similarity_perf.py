import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.normalized_cosine_similarity import normalized_cosine_similarity
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('normalized_cosine_similarity', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(6, 16):
            size = 2 ** i
            x1 = torch.rand((size, 1024), dtype=self.dtype)
            x2 = torch.rand((size, 1024), dtype=self.dtype)
            self.input_tensors.append((x1, x2))

    def to_cuda(self, input_tensor):
        x1, x2 = input_tensor
        return (x1.cuda(), x2.cuda())
    
    def call_op(self, input_tensor):
        x1, x2 = input_tensor
        return normalized_cosine_similarity(x1, x2, dim=1)
    
    def get_gbps(self, input_tensor, runtime):
        x1, x2 = input_tensor
        input_bytes = (x1.numel() + x2.numel()) * x1.element_size()
        output_bytes = x1.shape[0] * x1.element_size()
        total_bytes = input_bytes * 3 + output_bytes
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        x1, x2 = input_tensor
        N, D = x1.shape
        flops = 8 * N * D
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
