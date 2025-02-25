import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.sub import sub
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sub', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            other_tensor = torch.rand(size, dtype=self.dtype)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tensor_pair):
        input_tensor, other_tensor = input_tensor_pair
        return (input_tensor.cuda(), other_tensor.cuda())

    def call_op(self, input_tensor_pair):
        input, other = input_tensor_pair
        return sub(input, other, alpha=1)

    def get_gbps(self, input_tensor_pair, runtime):
        input, other = input_tensor_pair
        total_bytes = (input.numel() + other.numel() + input.numel()) * input.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor_pair, runtime):
        input, _ = input_tensor_pair
        FLOPS = input.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
