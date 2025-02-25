import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.index_fill_ import index_fill_
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('index_fill_', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):
            size = 2 ** i
            x = torch.randn(size, dtype=self.dtype)
            dim = 0
            index = torch.randint(0, size, (size // 2,), dtype=torch.long)
            value = 1.0
            self.input_tensors.append((dim, x, index, value))

    def to_cuda(self, input_tuple):
        dim, x, index, value = input_tuple
        return (dim, x.cuda(), index.cuda(), value)
    
    def call_op(self, input_tuple):
        dim, x, index, value = input_tuple
        return index_fill_(dim, x, index, value)
    
    def get_gbps(self, input_tuple, runtime):
        dim, x, index, value = input_tuple
        modified_elements = index.numel()
        total_bytes = modified_elements * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        dim, x, index, value = input_tuple
        operations = index.numel()
        TFLOPS = operations / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
