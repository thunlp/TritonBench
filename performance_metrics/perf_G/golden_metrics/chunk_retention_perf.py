import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_retention import chunk_retention
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_retention', dtype=dtype, is_backward=is_backward, **kwargs)
    
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range as needed
            B, H, T, D = 32, 128, 2 ** i, 128  # Example dimensions
            q = torch.rand(B, H, T, D, dtype=torch.float32)
            k = torch.rand(B, H, T, D, dtype=torch.float32)
            v = torch.rand(B, H, T, D, dtype=torch.float32)
            initial_state = torch.rand(B, H, T, D, dtype=torch.float32)
            self.input_tensors.append((q, k, v, initial_state))

    def to_cuda(self, input_tensor):
        q, k, v, initial_state = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), initial_state.cuda())

    def call_op(self, input_tensor):
        q, k, v, initial_state = input_tensor
        return chunk_retention(q, k, v, initial_state, output_final_state=True)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, _ = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, _ = input_tensor
        # Assuming a simple FLOP count for demonstration purposes
        FLOPS = 2 * q.numel() * k.size(-1) * v.size(-1)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
