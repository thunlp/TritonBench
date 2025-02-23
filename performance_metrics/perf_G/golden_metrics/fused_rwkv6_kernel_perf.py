import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fused_rwkv6_kernel import fused_recurrent_rwkv6
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_rwkv6_kernel', dtype=dtype, is_backward=is_backward, **kwargs)
        self.input_tensors = []

    def get_input_tensors(self):
        # Define different sizes for the input tensors
        for i in range(12, 18):  # Adjust the range as needed
            B = 2 ** (i - 10)
            H = 2 ** (i - 10)
            T = 2 ** (i - 10)
            K = 2 ** (i - 10)
            V = 2 ** (i - 10)
            r = torch.rand(B, H, T, K, dtype=torch.float32)
            k = torch.rand(B, H, T, K, dtype=torch.float32)
            v = torch.rand(B, H, T, V, dtype=torch.float32)
            w = torch.rand(B, H, T, K, dtype=torch.float32)
            u = torch.rand(H, K, dtype=torch.float32)
            self.input_tensors.append((r, k, v, w, u))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        r, k, v, w, u = input_tensor
        return fused_recurrent_rwkv6(r, k, v, w, u)

    def get_gbps(self, input_tensor, runtime):
        r, k, v, w, u = input_tensor
        total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in input_tensor)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        r, k, v, w, u = input_tensor
        # Assuming the main computation involves operations proportional to the size of r, k, v, w
        FLOPS = 2 * r.numel() * k.size(-1)  # Simplified estimation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
