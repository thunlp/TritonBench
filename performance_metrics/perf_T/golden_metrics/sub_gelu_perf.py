import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.sub_gelu import sub_gelu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, alpha=1, approximate='none', **kwargs):
        super().__init__('sub_gelu', dtype=dtype, is_backward=is_backward, **kwargs)
        self.alpha = alpha
        self.approximate = approximate

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 26):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            other_tensor = torch.rand(size, dtype=self.dtype)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tensor_tuple):
        input_tensor, other_tensor = input_tensor_tuple
        return (input_tensor.cuda(), other_tensor.cuda())
    
    def call_op(self, input_tensor_tuple):
        input_tensor, other_tensor = input_tensor_tuple
        return sub_gelu(input_tensor, other_tensor, alpha=self.alpha, approximate=self.approximate)

    def get_gbps(self, input_tensor_tuple, runtime):
        input_tensor, other_tensor = input_tensor_tuple
        element_size = input_tensor.element_size()
        num_elements = input_tensor.numel()
        total_bytes = (num_elements + num_elements + num_elements) * element_size * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor_tuple, runtime):
        input_tensor, other_tensor = input_tensor_tuple
        num_elements = input_tensor.numel()
        
        if self.approximate == 'tanh':
            ops_per_element = 11
        else:
            ops_per_element = 6
        
        total_flops = num_elements * ops_per_element
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
