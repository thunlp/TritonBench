import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.dropout_triton import dropout
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('dropout_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 30):
            size = 2 ** i
            x = torch.rand(size, dtype=torch.float16)
            x_keep = torch.rand(size, dtype=torch.float16) > 0.5  # Random mask
            self.input_tensors.append((x, x_keep))

    def to_cuda(self, input_tensor):
        x, x_keep = input_tensor
        return x.cuda(), x_keep.cuda()

    def call_op(self, input_tensor):
        x, x_keep = input_tensor
        p = 0.5  # Example dropout probability
        return dropout(x, x_keep, p)

    def get_gbps(self, input_tensor, runtime):
        x, x_keep = input_tensor
        total_bytes = x.numel() * x.element_size() * 2 + x_keep.numel() * x_keep.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _ = input_tensor
        FLOPS = x.numel()  # Each element involves one division and one conditional operation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
