import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_index_select_eq import fused_index_select_eq
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_index_select_eq', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):
            input_size = 2 ** i
            input_shape = (input_size, 128)
            input_tensor = torch.randn(input_shape, dtype=self.dtype or torch.float32)
            
            dim = 0
            index_length = max(1, input_size // 8)
            index = torch.randint(0, input_size, (index_length,), dtype=torch.long)
            
            other = 0.0
            
            selected = torch.index_select(input_tensor, dim, index)
            selected_numel = selected.numel()
            
            self.input_tensors.append((input_tensor, dim, index, other, selected_numel))

    def to_cuda(self, input_tuple):
        input_tensor, dim, index, other, selected_numel = input_tuple
        return (
            input_tensor.cuda(),
            dim,
            index.cuda(),
            other.cuda() if isinstance(other, torch.Tensor) else other,
            selected_numel
        )
    
    def call_op(self, input_tuple):
        input_tensor, dim, index, other, _ = input_tuple
        return fused_index_select_eq(input_tensor, dim, index, other)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, _, _, _, selected_numel = input_tuple
        input_bytes = selected_numel * input_tensor.element_size()
        output_bytes = selected_numel * 1
        total_bytes = input_bytes * 3 + output_bytes
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tuple, runtime):
        _, _, _, _, selected_numel = input_tuple
        return selected_numel / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
