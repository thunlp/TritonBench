import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_gla_simple import chunk_fwd_o_fn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_gla_simple', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):  # Adjust the range as needed for your testing
            B = 4
            H = 8
            T = 2 ** i
            K = 32
            V = 32
            BT = 64  # Block size for T
            scale = 1.0 / (K ** 0.5)
            q = torch.rand((B, H, T, K), dtype=torch.float32)
            k = torch.rand((B, H, K, T), dtype=torch.float32)
            v = torch.rand((B, H, T, V), dtype=torch.float32)
            h = torch.rand((B, H, K, V), dtype=torch.float32)
            g = torch.rand((B, H, T), dtype=torch.float32)
            self.input_tensors.append((h, q, k, v, g, BT, scale))

    def to_cuda(self, input_tensor):
        h, q, k, v, g, BT, scale = input_tensor
        return (h.cuda(), q.cuda(), k.cuda(), v.cuda(), g.cuda(), BT, scale)

    def call_op(self, input_tensor):
        h, q, k, v, g, BT, scale = input_tensor
        return chunk_fwd_o_fn(h, q, k, v, g, BT, scale)

    def get_gbps(self, input_tensor, runtime):
        h, q, k, v, g, BT, scale = input_tensor
        B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3]
        total_bytes = (q.numel() + k.numel() + v.numel() + h.numel() + g.numel()) * q.element_size() + v.numel() * v.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        h, q, k, v, g, BT, scale = input_tensor
        B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3]
        FLOPS = 2 * B * H * T * K * V  # Assuming 2 FLOPS per multiply-add operation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
