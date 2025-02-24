import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fused_recurrent_hgrn import fused_recurrent_hgrn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_recurrent_hgrn', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 15):  # Example sizes for T and D
            T = 2 ** i
            D = 2 ** i
            B = 4  # Batch size
            H = 2  # Number of heads
            x = torch.rand(B, H, T, D, dtype=torch.float32)
            g = torch.rand(B, H, T, D, dtype=torch.float32)
            initial_state = torch.rand(B, H, D, dtype=torch.float32)
            self.input_tensors.append((x, g, initial_state))

    def to_cuda(self, input_tensor):
        x, g, initial_state = input_tensor
        return (x.cuda(), g.cuda(), initial_state.cuda())

    def call_op(self, input_tensor):
        x, g, initial_state = input_tensor
        return fused_recurrent_hgrn(x, g, initial_state, output_final_state=True)

    def get_gbps(self, input_tensor, runtime):
        x, g, _ = input_tensor
        total_bytes = (x.numel() + g.numel() + x.numel()) * x.element_size()  # x, g, and output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _, _ = input_tensor
        B, H, T, D = x.shape
        FLOPS = 2 * B * H * T * D  # Assuming 2 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
