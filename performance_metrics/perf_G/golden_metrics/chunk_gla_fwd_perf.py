import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_gla_fwd import chunk_fwd_intra_gated_gk_fn, chunk_fwd_o_gated_gk_fn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_gla_fwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range as needed for your testing
            B, H, T, K, V = 8, 8, 2**i, 32, 32  # Example dimensions
            q = torch.rand(B, H, T, K, dtype=torch.float32)
            k = torch.rand(B, H, K, T, dtype=torch.float32)
            g = torch.rand(B, H, T, K, dtype=torch.float32)
            v = torch.rand(B, H, T, V, dtype=torch.float32)
            h = torch.rand(B, H, K, V, dtype=torch.float32)
            scale = 1.0 / (K ** 0.5)
            BT = 64  # Example block size
            self.input_tensors.append((q, k, g, v, h, scale, BT))

    def to_cuda(self, input_tensor):
        q, k, g, v, h, scale, BT = input_tensor
        return (q.cuda(), k.cuda(), g.cuda(), v.cuda(), h.cuda(), scale, BT)

    def call_op(self, input_tensor):
        q, k, g, v, h, scale, BT = input_tensor
        A = chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT)
        o = chunk_fwd_o_gated_gk_fn(q, v, g, A, h, BT, scale)
        return o

    def get_gbps(self, input_tensor, runtime):
        q, k, g, v, h, scale, BT = input_tensor
        total_bytes = (q.numel() + k.numel() + g.numel() + v.numel() + h.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, g, v, h, scale, BT = input_tensor
        B, H, T, K = q.shape
        V = v.shape[-1]
        # Assuming each operation involves a multiply and an add
        FLOPS = 2 * B * H * T * K * V
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
