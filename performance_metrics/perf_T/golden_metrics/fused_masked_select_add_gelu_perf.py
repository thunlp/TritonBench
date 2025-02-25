import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_masked_select_add_gelu import fused_masked_select_add_gelu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_masked_select_add_gelu', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            mask = torch.rand(size) > 0.5
            mask_count = mask.sum().item()
            other = 0.5
            self.input_tensors.append((input_tensor, mask, other, mask_count))

    def to_cuda(self, input_tuple):
        input_tensor, mask, other, mask_count = input_tuple
        return (input_tensor.cuda(), mask.cuda(), other, mask_count)

    def call_op(self, input_tuple):
        input_tensor, mask, other, _ = input_tuple
        return fused_masked_select_add_gelu(input_tensor, mask, other, alpha=1, approximate='none')

    def get_gbps(self, input_tuple, runtime):
        input_tensor, mask, _, mask_count = input_tuple
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        mask_bytes = mask.numel() * mask.element_size()
        element_size = input_tensor.element_size()
        
        total_bytes = (input_bytes + mask_bytes + 
                      5 * mask_count * element_size + 4 * input_bytes)  # 5 = Z(r+w) + S(r+w) + Y(w)
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        _, _, _, mask_count = input_tuple
        total_flops = 9 * mask_count  
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
