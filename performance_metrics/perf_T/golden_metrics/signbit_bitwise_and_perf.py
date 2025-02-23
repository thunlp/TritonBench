import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.signbit_bitwise_and import signbit_bitwise_and
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('signbit_bitwise_and', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)
            other_tensor = torch.randint(0, 2, (size,), dtype=torch.int8)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tuple):
        input_tensor, other_tensor = input_tuple
        return (input_tensor.cuda(), other_tensor.cuda())
    
    def call_op(self, input_tuple):
        input_tensor, other_tensor = input_tuple
        return signbit_bitwise_and(input_tensor, other_tensor)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, other_tensor = input_tuple
        size = input_tensor.numel()
        input_bytes = input_tensor.element_size() * size * 2 + other_tensor.element_size() * size
        output_bytes = 2 * size * 1  # 1 byte per element for both outputs
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, other_tensor = input_tuple
        flops = 2 * input_tensor.numel()
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
