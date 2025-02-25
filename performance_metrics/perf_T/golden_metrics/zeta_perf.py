import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.zeta import zeta
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('zeta', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            other_tensor = torch.rand(size, dtype=self.dtype) + 1.0
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tuple):
        input_tensor, other_tensor = input_tuple
        return (input_tensor.cuda(), other_tensor.cuda())
    
    def call_op(self, input_tuple):
        input_tensor, other_tensor = input_tuple
        return zeta(input_tensor, other_tensor)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, other_tensor = input_tuple
        numel = input_tensor.numel()
        element_size = input_tensor.element_size()
        total_bytes = 3 * numel * element_size
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        input_tensor, _ = input_tuple
        flops = input_tensor.numel() * 10
        return flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
