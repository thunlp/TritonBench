import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_bwd_dqkg import chunk_bwd_dqkg_fn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_bwd_dqkg', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):  # Adjust the range as needed for your testing
            B = 2
            H = 2
            T = 2 ** i
            K = 64
            V = 64
            scale = 1.0 / (K ** 0.5)
            q = torch.rand((B, H, T, K), dtype=torch.float32)
            k = torch.rand((B, H, T, K), dtype=torch.float32)
            v = torch.rand((B, H, T, V), dtype=torch.float32)
            g = torch.rand((B, H, T), dtype=torch.float32)
            h = torch.rand((B, H, V, K), dtype=torch.float32)
            do = torch.rand((B, H, T, V), dtype=torch.float32)
            dh = torch.rand((B, H, V, K), dtype=torch.float32)
            self.input_tensors.append((do, q, k, v, g, h, dh, scale))

    def to_cuda(self, input_tensor):
        return tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in input_tensor)

    def call_op(self, input_tensor):
        do, q, k, v, g, h, dh, scale = input_tensor
        return chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale)

    def get_gbps(self, input_tensor, runtime):
        do, q, k, v, g, h, dh, scale = input_tensor
        total_bytes = (do.numel() + q.numel() + k.numel() + v.numel() + g.numel() + h.numel() + dh.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        do, q, k, v, g, h, dh, scale = input_tensor
        B, H, T, K = q.shape
        V = v.shape[-1]
        # Estimate FLOPS based on operations in the kernel
        FLOPS = 2 * B * H * T * K * V
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
