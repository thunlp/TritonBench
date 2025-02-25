import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_gather_masked_fill import fused_gather_masked_fill
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_gather_masked_fill', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 13):
            S = 2 ** i
            input_tensor = torch.randn((S, 1024), dtype=torch.float32)
            index = torch.randint(0, S, (S, 1024), dtype=torch.int64)
            mask = torch.rand((S, 1024)) > 0.5
            value = 0.5
            dim = 0
            self.input_tensors.append((input_tensor, dim, index, mask, value))

    def to_cuda(self, input_tuple):
        input_tensor, dim, index, mask, value = input_tuple
        return (
            input_tensor.cuda(),
            dim,
            index.cuda(),
            mask.cuda(),
            value
        )
    
    def call_op(self, input_tuple):
        return fused_gather_masked_fill(*input_tuple)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, _, index, mask, _ = input_tuple
        
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        index_bytes = index.numel() * index.element_size()
        mask_bytes = mask.numel() * mask.element_size()
        output_bytes = index.numel() * input_tensor.element_size()
        
        total_bytes = input_bytes + index_bytes + mask_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tuple, runtime):
        _, _, index, _, _ = input_tuple
        flops = index.numel()
        return flops / (runtime / 1000) / 1e12
    
if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
