import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fused_recurrent_delta import fused_recurrent_delta_rule
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_recurrent_delta', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 11):  # Adjust the range for different sizes
            B, H, T, K, V = 4, 8, 2**i, 64, 64  # Example dimensions
            q = torch.rand(B, H, T, K, dtype=torch.float32)
            k = torch.rand(B, H, T, K, dtype=torch.float32)
            v = torch.rand(B, H, T, V, dtype=torch.float32)
            beta = torch.rand(B, H, T, dtype=torch.float32)
            self.input_tensors.append((q, k, v, beta))

    def to_cuda(self, input_tensor):
        q, k, v, beta = input_tensor
        return q.cuda(), k.cuda(), v.cuda(), beta.cuda()

    def call_op(self, input_tensor):
        q, k, v, beta = input_tensor
        return fused_recurrent_delta_rule(q, k, v, beta)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, beta = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + beta.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, beta = input_tensor
        # Assuming the operation involves a number of FLOPs proportional to the size of q, k, v
        FLOPS = 2 * q.numel() * v.size(-1)  # Example calculation, adjust based on actual operation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
