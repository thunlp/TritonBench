import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_gated_attention import fwd_pre, fwd_inner
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_gated_attention', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        B, H = 32, 128  # Batch size and number of heads
        for i in range(2, 10):  # Example range for tensor sizes
            T = 2 ** i
            S = T
            K = 32
            V = 32
            BT = 64
            BK = 32
            BV = 32
            q = torch.rand((B, H, T, K), dtype=torch.float16)
            k = torch.rand((B, H, S, K), dtype=torch.float16)
            v = torch.rand((B, H, S, V), dtype=torch.float16)
            g = torch.rand((B, H, T, S), dtype=torch.float16)
            self.input_tensors.append((q, k, v, g, B, H, T, K, V, BT, BK, BV))

    def to_cuda(self, input_tensor):
        q, k, v, g, B, H, T, K, V, BT, BK, BV = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), g.cuda(), B, H, T, K, V, BT, BK, BV)

    def call_op(self, input_tensor):
        q, k, v, g, B, H, T, K, V, BT, BK, BV = input_tensor
        g_pre = fwd_pre(g, B, H, T, S=T, BT=BT)
        return fwd_inner(q, k, v, g_pre, B, H, T, K, V, BT, BK, BV)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, g, B, H, T, K, V, BT, BK, BV = input_tensor
        g_pre = fwd_pre(g, B, H, T, S=T, BT=BT)
        total_bytes = (q.numel() + k.numel() + v.numel() + g.numel()) * q.element_size() + g_pre.numel() * g_pre.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, g, B, H, T, K, V, BT, BK, BV = input_tensor
        FLOPS = 2 * B * H * T * K * V  # Example calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
