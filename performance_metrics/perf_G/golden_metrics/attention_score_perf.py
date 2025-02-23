import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.attention_score import get_score
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attention_score', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):  # Adjust the range as needed for testing
            size = 2 ** i
            q = torch.rand((1, 1, size, 64), dtype=torch.float32)  # Example shape
            k = torch.rand((1, 1, size, 64), dtype=torch.float32)
            m = torch.rand((1, 1, size), dtype=torch.float32)
            self.input_tensors.append((q, k, m))

    def to_cuda(self, input_tensor):
        q, k, m = input_tensor
        return q.cuda(), k.cuda(), m.cuda()

    def call_op(self, input_tensor):
        q, k, m = input_tensor
        return get_score(q, k, m, sliding_window=None, complement_sliding_window=False)

    def get_gbps(self, input_tensor, runtime):
        q, k, m = input_tensor
        total_bytes = (q.numel() + k.numel() + m.numel()) * q.element_size() + q.size(0) * q.size(1) * k.size(2) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, m = input_tensor
        FLOPS = 2 * q.size(0) * q.size(1) * q.size(2) * k.size(2) * q.size(3)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
