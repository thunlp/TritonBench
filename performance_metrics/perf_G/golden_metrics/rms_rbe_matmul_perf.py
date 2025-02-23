import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rms_rbe_matmul import rms_matmul_rbe_wrapper
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rms_rbe_matmul', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):  # Adjust the range as needed for testing
            batch_size = 1
            M = 2 ** i
            K = 2 ** i
            N = 2 ** i
            x = torch.rand((batch_size, M, K), dtype=torch.float16)
            weight = torch.rand((K, N), dtype=torch.float16)
            rms_w = torch.rand((K,), dtype=torch.float16)
            self.input_tensors.append((x, weight, rms_w, True, 0, 1, N))

    def to_cuda(self, input_tensor):
        x, weight, rms_w, use_rbe, start_pos, n_heads, head_dim = input_tensor
        return (x.cuda(), weight.cuda(), rms_w.cuda(), use_rbe, start_pos, n_heads, head_dim)

    def call_op(self, input_tensor):
        x, weight, rms_w, use_rbe, start_pos, n_heads, head_dim = input_tensor
        return rms_matmul_rbe_wrapper(x, weight, rms_w, use_rbe, start_pos, n_heads, head_dim)

    def get_gbps(self, input_tensor, runtime):
        x, weight, rms_w, _, _, _, _ = input_tensor
        total_bytes = (x.numel() + weight.numel() + rms_w.numel()) * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, weight, _, _, _, _, _ = input_tensor
        FLOPS = 2 * x.size(0) * x.size(1) * weight.size(1) * weight.size(0)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
