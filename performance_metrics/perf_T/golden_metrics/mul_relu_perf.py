import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.mul_relu import mul_relu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('mul_relu', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            other_tensor = torch.rand(size, dtype=self.dtype)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tensor_tuple):
        input_cuda = input_tensor_tuple[0].cuda()
        other_cuda = input_tensor_tuple[1].cuda()
        return (input_cuda, other_cuda)
    
    def call_op(self, input_tensor_tuple):
        input, other = input_tensor_tuple
        return mul_relu(input, other)
    
    def get_gbps(self, input_tensor_tuple, runtime):
        input, other = input_tensor_tuple
        numel = input.numel()
        element_size = input.element_size()
        total_bytes = 5 * numel * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor_tuple, runtime):
        input, other = input_tensor_tuple
        numel = input.numel()
        FLOPS = numel * 2
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
