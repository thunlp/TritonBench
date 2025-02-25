import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.bitwise_and import bitwise_and
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bitwise_and', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            other_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda())
    
    def call_op(self, input_tensor):
        return bitwise_and(input_tensor[0], input_tensor[1])
    
    def get_gbps(self, input_tensor, runtime):
        numel = input_tensor[0].numel()
        element_size = input_tensor[0].element_size()
        total_bytes = 3 * numel * element_size  # 3 * num_elements * bytes_per_element
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        numel = input_tensor[0].numel()
        TFLOPS = numel / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
