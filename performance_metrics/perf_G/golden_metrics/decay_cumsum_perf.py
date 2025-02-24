import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Triton kernel launch function
from TritonBench_v1.decay_cumsum import launch_fwd_decay_cumsum
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('decay_cumsum', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 9):
            size = 2 ** i
            B, H, T, DK = 32, 8, size, 128  # Example dimensions
            g = torch.rand((B, H, T, DK), dtype=torch.float16)
            g_o = torch.zeros_like(g)
            scale = 1.0
            BT, BK = 64, 64  # Example block sizes
            self.input_tensors.append((g, g_o, B, H, T, scale, BT, BK, DK))

    def to_cuda(self, input_tensor):
        g, g_o, B, H, T, scale, BT, BK, DK = input_tensor
        return (g.cuda(), g_o.cuda(), B, H, T, scale, BT, BK, DK)

    def call_op(self, input_tensor):
        g, g_o, B, H, T, scale, BT, BK, DK = input_tensor
        launch_fwd_decay_cumsum(g, g_o, B, H, T, scale, BT, BK, DK)
        return g_o

    def get_gbps(self, input_tensor, runtime):
        g, _, _, _, _, _, _, _, DK = input_tensor
        total_bytes = 2 * g.numel() * g.element_size()  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        g, _, _, _, T, _, _, _, DK = input_tensor
        FLOPS = 2 * T * DK  # Example FLOPS calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
