import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_linear_attn import chunk_linear_attn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_linear_attn', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range as needed
            B, H, T, K, V = 32, 128, 2 ** i, 64, 64  # Example dimensions
            q = torch.rand(B, H, T, K, dtype=torch.float32)
            k = torch.rand(B, H, T, K, dtype=torch.float32)
            v = torch.rand(B, H, T, V, dtype=torch.float32)
            self.input_tensors.append((q, k, v))

    def to_cuda(self, input_tensor):
        q, k, v = input_tensor
        return q.cuda(), k.cuda(), v.cuda()

    def call_op(self, input_tensor):
        q, k, v = input_tensor
        scale = q.shape[-1] ** -0.5
        return chunk_linear_attn(q, k, v, scale=scale)

    def get_gbps(self, input_tensor, runtime):
        q, k, v = input_tensor
        B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3]
        total_bytes = (q.numel() + k.numel() + v.numel() + B * H * T * V) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v = input_tensor
        B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3]
        FLOPS = 2 * B * H * T * K * V  # Simplified FLOPS calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
